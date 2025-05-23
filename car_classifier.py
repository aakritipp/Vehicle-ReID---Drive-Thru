import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from helpers import load_track_dataframes, load_tracks, load_true_global_id_maps, assign_global_ids


# 1. Load data from a group directory
def load_group_data(group_dir):
    """Load all track dataframes from a group directory"""
    # print(f"Loading data from {group_dir}...")
    track_dataframes, video_ids = load_track_dataframes(group_dir)
    return track_dataframes, video_ids


# 2. Parse ground truth data for car classification
def parse_ground_truth(ground_truth_maps):
    """
    Parse ground truth data to identify car vs. non-car
    Tracks present in the ground truth are cars, those not present are non-cars
    """
    car_tracks = {}
    for video_id, tracks in ground_truth_maps.items():
        video_id = int(video_id)
        car_tracks[video_id] = set()
        for track_id in tracks.keys():
            car_tracks[video_id].add(int(track_id))

    return car_tracks


# 3. Extract features for each track using the provided feature vectors
def extract_features_from_group(track_dataframes, car_tracks):
    """Extract features from all tracks in all videos using provided feature vectors"""
    all_features = []
    all_track_info = []  # (video_id, track_id) pairs
    all_labels = []  # 1 for car, 0 for non-car

    for video_id, df in track_dataframes.items():
        # print(f"Processing video {video_id}, with {len(df)} rows and {df['track_id'].nunique()} tracks")

        # Convert video_id to int if needed
        video_id = int(video_id)

        # Process each track
        for track_id, group in df.groupby('track_id'):
            # Skip if too few data points
            if len(group) < 3:
                continue

            # Check if this track is a car according to ground truth
            is_car = 0
            if video_id in car_tracks and int(track_id) in car_tracks[video_id]:
                is_car = 1

            # Sort by frame index
            group = group.sort_values('frame_index')

            # Use the provided feature vectors (average them for each track)
            feature_vectors = np.array(group['feature_vector'].tolist())
            avg_feature_vector = np.mean(feature_vectors, axis=0)

            # Add some additional motion features
            # Movement features
            x_positions = group['world_x'].values
            y_positions = group['world_y'].values

            # Calculate trajectory features
            if len(x_positions) > 1:
                # X and Y range of movement
                x_range = np.max(x_positions) - np.min(x_positions)
                y_range = np.max(y_positions) - np.min(y_positions)
            else:
                x_range = y_range = 0

            # Bounding box features
            bbox_heights = group['bbox_height'].values
            bbox_widths = group['bbox_width'].values

            # Additional features to append to the feature vector
            additional_features = [
                x_range,  # X range of movement
                y_range,  # Y range of movement
                len(group),  # Number of detections
                np.mean(bbox_heights),  # Average height
                np.mean(bbox_widths),  # Average width
                np.mean(bbox_widths / bbox_heights)  # Average aspect ratio
            ]

            # Combine the visual feature vector with additional features
            combined_features = np.concatenate([avg_feature_vector, additional_features])

            all_features.append(combined_features)
            all_track_info.append((video_id, track_id))
            all_labels.append(is_car)

    # Convert to numpy arrays
    features = np.array(all_features)
    labels = np.array(all_labels)

    return features, all_track_info, labels


# 4. Build model
def build_model(input_shape):
    """Build a neural network for binary classification"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


# 5. Filter noise from all videos
def filter_noise_from_group(track_dataframes, model, features, track_info, threshold=0.1):
    """Filter noise from all videos in the group"""
    # Get predictions
    predictions = model.predict(features)
    # print(predictions)

    # Create a dictionary to store predictions by (video_id, track_id)
    pred_dict = {}
    for i, (video_id, track_id) in enumerate(track_info):
        pred_dict[(video_id, track_id)] = predictions[i][0]
    # print(pred_dict)

    # Filter each dataframe
    filtered_dataframes = {}
    total_original = 0
    total_filtered = 0

    for video_id, df in track_dataframes.items():
        video_id = int(video_id)
        original_rows = len(df)
        total_original += original_rows

        # Get tracks that are predicted as cars
        car_track_ids = [track_id for (vid, track_id), pred in pred_dict.items()
                         if vid == video_id and pred >= threshold]
        # print(car_track_ids)

        # Filter dataframe
        filtered_df = df[df['track_id'].isin(car_track_ids)]
        filtered_dataframes[video_id] = filtered_df
        filtered_rows = len(filtered_df)
        total_filtered += filtered_rows

    return filtered_dataframes


# 6. Save filtered dataframes
def save_filtered_dataframes(filtered_dataframes, output_dir):
    """Save all filtered dataframes"""
    os.makedirs(output_dir, exist_ok=True)

    for video_id, df in filtered_dataframes.items():
        output_path = os.path.join(output_dir, f"filtered_tracks_{video_id}.parquet")
        df.to_parquet(output_path)
        # print(f"Saved filtered data for video {video_id} to {output_path}")

    video_ids = os.path.basename(output_dir).split("_")
    video_ids = [int(video_id) for video_id in video_ids]
    track_dataframes = {}
    for video_id in video_ids:
        parquet_path = os.path.join(output_dir, f"filtered_tracks_{video_id}.parquet")
        video_path = os.path.join(output_dir, f"video_{video_id}.mp4")
        track_dataframes[video_id] = load_tracks(parquet_path, video_path)
    return track_dataframes, video_ids


# 8. Main function
def car_classifier(group_dir):
    """Main function to process a group directory"""
    # Load all data
    track_dataframes, video_ids = load_group_data(group_dir)

    # Load ground truth data
    true_global_id_maps = load_true_global_id_maps(group_dir)

    if not true_global_id_maps:
        print("No ground truth data found! Using unsupervised approach.")
        # Implement alternative if no ground truth is available
        # This could be your original approach with heuristics
        return

    # Parse ground truth to identify car tracks
    car_tracks = parse_ground_truth(true_global_id_maps)

    # Extract features and get labels from ground truth
    features, track_info, labels = extract_features_from_group(track_dataframes, car_tracks)

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build and train model
    print("Training model...")
    model = build_model(scaled_features.shape[1])

    # Use class weights to handle imbalance if needed
    class_weights = None
    if np.sum(labels) / len(labels) < 0.3:  # If less than 30% are cars
        weight_for_cars = (len(labels) - np.sum(labels)) / np.sum(labels)
        class_weights = {0: 1.0, 1: min(weight_for_cars, 5.0)}  # Cap the weight at 5.0

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        verbose=1,
        class_weight=class_weights
    )

    # Evaluate
    evaluation = model.evaluate(X_test, y_test)
    filtered_dataframes = filter_noise_from_group(track_dataframes, model, scaled_features, track_info)
    return save_filtered_dataframes(filtered_dataframes, group_dir)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Filter noise from vehicle tracking data')
    parser.add_argument('--group_dir', type=str, required=True,
                        help='Path to the directory containing the group of video and track files')

    args = parser.parse_args()

    filtered_dataframes = car_classifier(args.group_dir)