import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dinov2_model(device):
    """Load the Dinov2_small model for faster feature extraction."""
    logger.info(f"Loading DINOv2 model on {device}")
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(images):
    """Preprocess a batch of images for Dinov2 model, standardizing bounding boxes."""
    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # Dinov2 expects 518x518 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Standardize aspect ratio by padding to square
    standardized_images = []
    for img in images:
        w, h = img.size
        if w <= 0 or h <= 0:
            logger.warning("Invalid image dimensions, skipping image")
            continue
        max_side = max(w, h)
        padded_img = Image.new('RGB', (max_side, max_side), (123, 117, 104))  # Mean RGB color
        padded_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        standardized_images.append(padded_img)
    if not standardized_images:
        logger.warning("No valid images to preprocess")
        return None
    return torch.stack([transform(img) for img in standardized_images])

def extract_dinov2_features(model, images, device):
    """Extract features for a batch of images using Dinov2_small model."""
    try:
        image_tensor = preprocess_image(images)
        if image_tensor is None:
            logger.warning("No valid images to process, returning None")
            return None
        image_tensor = image_tensor.to(device)
        logger.debug(f"Processing {image_tensor.shape[0]} images on {device}")
        with torch.no_grad():
            features = model(image_tensor)
        return features.cpu().numpy()
    except RuntimeError as e:
        logger.error(f"CUDA error during feature extraction: {e}")
        if "out of memory" in str(e).lower():
            logger.warning("Attempting to clear CUDA memory")
            torch.cuda.empty_cache()
        return None

def estimate_direction(track_data):
    """Estimate vehicle direction based on bounding box movement."""
    if len(track_data) < 2 or 'utc_time' not in track_data.columns:
        logger.debug("Insufficient data for direction estimation, returning 'unknown'")
        return 'unknown'
    track_data = track_data.sort_values('utc_time')
    # Compute bounding box centers: ((u1+u2)/2, (v1+v2)/2)
    centers = track_data[['u1', 'u2', 'v1', 'v2']].copy()
    centers['x'] = (centers['u1'] + centers['u2']) / 2
    centers['y'] = (centers['v1'] + centers['v2']) / 2
    centers = centers[['x', 'y']].values  # Shape: (n_frames, 2)
    # Compute displacement between consecutive frames
    displacements = np.diff(centers, axis=0)  # Shape: (n_frames-1, 2)
    if len(displacements) == 0:
        logger.debug("No displacements calculated, returning 'unknown'")
        return 'unknown'
    # Average displacement in x and y
    avg_dx, avg_dy = np.mean(displacements, axis=0)
    # Determine primary direction
    if abs(avg_dx) > abs(avg_dy):
        return 'left' if avg_dx < 0 else 'right'
    else:
        return 'up' if avg_dy < 0 else 'down'

def sample_frames(tracks_df, time_interval=0.5):
    """Sample frames per track at specified time intervals (in seconds)."""
    sampled_indices = []
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        if 'utc_time' not in track_data.columns:
            logger.warning(f"'utc_time' not in DataFrame, processing all frames for track {track_id}")
            sampled_indices.extend(track_data.index)
            continue
        # Convert utc_time (ms) to seconds and find frames at ~time_interval intervals
        track_data = track_data.sort_values('utc_time')
        start_time = track_data['utc_time'].iloc[0] / 1000
        last_selected = start_time
        for idx, row in track_data.iterrows():
            current_time = row['utc_time'] / 1000
            if current_time >= last_selected + time_interval:
                sampled_indices.append(idx)
                last_selected = current_time
    return tracks_df.loc[sampled_indices].copy()

def aggregate_features(feature_vectors, direction, camera_directions, video_ids, total_cameras):
    """Aggregate features from available views with direction-based weighting, handling blind spots."""
    if len(feature_vectors) == 0:
        logger.warning("No feature vectors to aggregate")
        return None
    # Validate feature vectors
    feature_dim = feature_vectors[0].shape[0]
    if not all(f.shape[0] == feature_dim for f in feature_vectors):
        logger.warning("Inconsistent feature dimensions, skipping aggregation")
        return None
    # Initialize weights based on direction alignment
    weights = []
    for vid in video_ids:
        camera_dir = camera_directions.get(vid, 'unknown')
        # Assign higher weight if camera direction aligns with vehicle direction
        if camera_dir == direction or camera_dir == 'unknown':
            weights.append(1.0)
        else:
            weights.append(0.5)  # Lower weight for non-aligned views
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    # Weighted average of available features
    weighted_features = np.average(feature_vectors, axis=0, weights=weights)
    # Normalize to unit length for cosine similarity
    weighted_features = weighted_features / (np.linalg.norm(weighted_features) + 1e-8)
    # Pad with zeros for missing views
    padded_features = [f / (np.linalg.norm(f) + 1e-8) for f in feature_vectors]  # Normalize each feature
    for _ in range(total_cameras - len(feature_vectors)):
        padded_features.append(np.zeros(feature_dim))
    # Concatenate weighted average with padded features
    all_features = np.concatenate([weighted_features] + padded_features, axis=0)
    return all_features

def process_tracks_for_video_group(track_dfs, video_files, video_ids, camera_directions, batch_size=16, time_interval=0.5):
    """
    Process tracks across multiple videos (cameras) with direction-aware feature aggregation.

    Args:
        track_dfs (dict): Dictionary of video_id to DataFrame with track data
        video_files (dict): Dictionary mapping video_id to video file paths
        video_ids (list): List of video IDs (camera IDs)
        camera_directions (dict): Mapping of video_id to primary camera direction
        batch_size (int): Number of images to process in a batch
        time_interval (float): Time interval (seconds) for frame sampling

    Returns:
        pd.DataFrame: Processed DataFrame with direction-aware feature vectors
    """
    start_time = time.time()
    logger.info(f"Starting processing for video group {video_ids}...")

    # Sample frames for each video
    processed_dfs = {}
    for video_id in video_ids:
        if video_id not in track_dfs:
            logger.warning(f"No track data for video_id {video_id}, treating as blind spot")
            continue
        processed_dfs[video_id] = sample_frames(track_dfs[video_id], time_interval) if time_interval > 0 else track_dfs[video_id].copy()
        processed_dfs[video_id]['new_feature_vector'] = [None] * len(processed_dfs[video_id])
        logger.info(f"Video {video_id}: Processing {len(processed_dfs[video_id])} frames (original: {len(track_dfs[video_id])})")

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        logger.warning("CUDA not available, falling back to CPU")

    # Load Dinov2 model
    dinov2_model = load_dinov2_model(device)

    # Combine all tracks for synchronization
    combined_df = pd.concat([df.assign(video_id=vid) for vid, df in processed_dfs.items()], ignore_index=True)
    if combined_df.empty:
        logger.error("No valid track data to process")
        return pd.DataFrame()

    # Create a mapping from combined_df index to original processed_dfs index
    index_mapping = {}
    offset = 0
    for video_id, df in processed_dfs.items():
        for idx in df.index:
            index_mapping[offset] = (video_id, idx)
            offset += 1

    # Group by track_id and utc_time to process multi-view frames
    grouped = combined_df.groupby(['track_id', 'utc_time'])

    total_cameras = len(video_ids)
    for (track_id, utc_time), group in tqdm(grouped, desc="Processing multi-view tracks"):
        images = []
        indices = []
        video_ids_present = []

        # Estimate direction for this track
        track_data = combined_df[combined_df['track_id'] == track_id]
        direction = estimate_direction(track_data)

        # Process each available view (video)
        for _, row in group.iterrows():
            video_id = row['video_id']
            frame_index = row['frame_index']
            u1, u2, v1, v2 = row['u1'], row['u2'], row['v1'], row['v2']

            if video_id not in video_files:
                logger.warning(f"No video file for video_id {video_id}")
                continue
            cap = cv2.VideoCapture(video_files[video_id])
            if not cap.isOpened():
                logger.error(f"Could not open video {video_files[video_id]} for video_id {video_id}")
                continue

            # Set video to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                logger.warning(f"Could not read frame {frame_index} for track {track_id}, video_id {video_id}")
                continue

            # Extract bounding box with validation
            x1, y1, x2, y2 = int(u1), int(v1), int(u2), int(v2)
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                logger.warning(f"Invalid bounding box ({x1},{y1},{x2},{y2}) for track {track_id}, frame {frame_index}, video_id {video_id}")
                continue

            bbox_image = frame[y1:y2, x1:x2]
            if bbox_image.size == 0:
                logger.warning(f"Empty bounding box for track {track_id}, frame {frame_index}, video_id {video_id}")
                continue

            # Convert to RGB and PIL Image
            bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(bbox_image_rgb)

            images.append(pil_image)
            # Store the original index from processed_dfs using the index_mapping
            combined_idx = row.name
            if combined_idx in index_mapping:
                vid, orig_idx = index_mapping[combined_idx]
                indices.append((vid, orig_idx))
                video_ids_present.append(video_id)
            else:
                logger.warning(f"Index {combined_idx} not found in index_mapping for track {track_id}, video_id {video_id}")

        # Extract and aggregate features
        if images:
            feature_vectors = extract_dinov2_features(dinov2_model, images, device)
            if feature_vectors is None:
                logger.warning(f"Feature extraction failed for track {track_id}, utc_time {utc_time}")
                continue
            aggregated_feature = aggregate_features(feature_vectors, direction, camera_directions, video_ids_present, total_cameras)
            if aggregated_feature is None:
                logger.warning(f"Feature aggregation failed for track {track_id}, utc_time {utc_time}")
                continue
            # Assign aggregated feature to all rows in this group
            for vid, idx in indices:
                processed_dfs[vid].at[idx, 'new_feature_vector'] = aggregated_feature

    # Combine and clean results
    final_dfs = []
    for video_id, df in processed_dfs.items():
        df = df[df['new_feature_vector'].notnull()]
        df['video_id'] = video_id
        final_dfs.append(df)

    combined_df = pd.concat(final_dfs, ignore_index=True) if final_dfs else pd.DataFrame()

    end_time = time.time()
    logger.info(f"Finished processing video group in {end_time - start_time:.2f} seconds")
    return combined_df

def process_all_tracks(track_dataframes, video_files, group_dir, batch_size=16, time_interval=0.5):
    """
    Process tracks for all videos with direction-aware feature aggregation for fixed paths.

    Args:
        track_dataframes (dict): Dictionary of video_id to DataFrame from classifier
        video_files (dict): Dictionary mapping video_id to video file paths
        group_dir (str): Directory for saving output
        batch_size (int): Number of images to process in a batch
        time_interval (float): Time interval (seconds) for frame sampling

    Returns:
        pd.DataFrame: Combined DataFrame with direction-aware feature vectors
    """
    # Define camera directions (adjust based on your setup)
    camera_directions = {
        824714: 'front',  # Camera 1 faces front of vehicle
        824719: 'side',   # Camera 2 faces side of vehicle
        # Add more video_ids and their primary directions
    }

    all_dfs = []
    video_ids = list(track_dataframes.keys())
    if not video_ids:
        logger.error("No video IDs to process")
        return pd.DataFrame()

    # Process all videos as a group
    combined_df = process_tracks_for_video_group(
        track_dfs=track_dataframes,
        video_files=video_files,
        video_ids=video_ids,
        camera_directions=camera_directions,
        batch_size=batch_size,
        time_interval=time_interval
    )

    if not combined_df.empty:
        all_dfs.append(combined_df)

    # Combine all DataFrames
    if not all_dfs:
        logger.warning("No data processed, returning empty DataFrame")
        return pd.DataFrame()
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Save per video_id
    for video_id in video_ids:
        video_df = final_df[final_df['video_id'] == video_id]
        if video_df.empty:
            logger.warning(f"No data for video_id {video_id}, skipping parquet save")
            continue
        output_path = f"{group_dir}/processed_tracks_{video_id}.parquet"
        video_df.to_parquet(output_path)
        logger.info(f"Saved {len(video_df)} rows to {output_path}")

    return final_df

# Example usage
if __name__ == "__main__":
    from car_classifier import car_classifier
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group_dir", type=str, required=True,
        help="Path to the directory containing the group of video and track files"
    )
    args = parser.parse_args()
    args.group_dir = args.group_dir.rstrip("/")

    # Load classifier output
    track_dataframes, video_ids = car_classifier(args.group_dir)

    # Map video_id to video file paths
    video_files = {vid: f"{args.group_dir}/video_{vid}.mp4" for vid in video_ids}

    # Process tracks with direction-aware feature extraction
    processed_df = process_all_tracks(track_dataframes, video_files, args.group_dir, batch_size=32, time_interval=0.5)