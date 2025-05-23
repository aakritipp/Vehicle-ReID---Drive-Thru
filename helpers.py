"""
Helper functions that handle data loading and global ID assignment.
"""
import argparse
import os
import json
import cv2
import pandas as pd
import numpy as np
from typing import Dict, Literal


def remove_small_detections(
        tracks: pd.DataFrame, 
        min_height_fraction: float = 0.07, 
        min_width_fraction: float = 0.07
        ) -> pd.DataFrame:
    """
    Remove detections that are too small.

    :param tracks: Pandas DataFrame containing track data
    :param min_height_fraction: Minimum height fraction of the bounding box relative to the video height
    :param min_width_fraction: Minimum width fraction of the bounding box relative to the video width
    :return: Pandas DataFrame containing track data with small detections removed
    """
    tracks = tracks[tracks["bbox_height"] > min_height_fraction * tracks["video_height"]]
    tracks = tracks[tracks["bbox_width"] > min_width_fraction * tracks["video_width"]]
    return tracks


def load_tracks(tracks_path: str, video_path: str) -> pd.DataFrame:
    """
    Load track data from a parquet file and add video height and width, and bounding box height and width
    to the DataFrame. Additionally, remove small detections.

    This file has the following columns:
    - track_id: unique identifier for the track. Note that a vehicle can have multiple tracks within the same video
        since tracking isn't perfect.
    - frame_index: index of the frame in the video
    - u1, u2, v1, v2: bounding box coordinates of the vehicle (in pixels)
    - utc_time: timestamp of the track (in milliseconds since epoch)
    - world_x, world_y: world coordinates of the vehicle (in meters)
    - feature_vector: embedding vector representing the visual features of the vehicle. Vehicles that look
        similar should have feature vectors that are closer to each other in vector space than vehicles
        that look very different.

    The following columns will be added to the DataFrame:
    - bbox_height: height of the bounding box in pixels
    - bbox_width: width of the bounding box in pixels
    - video_height: height of the video in pixels
    - video_width: width of the video in pixels
    
    :param tracks_path: Path to the parquet file containing track data
    :param video_path: Path to the video file
    :return: Pandas DataFrame containing track data
    """
    tracks = pd.read_parquet(tracks_path)
    video = cv2.VideoCapture(video_path)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video.release()
    tracks["bbox_height"] = tracks["v2"] - tracks["v1"]
    tracks["bbox_width"] = tracks["u2"] - tracks["u1"]
    tracks["video_height"] = video_height
    tracks["video_width"] = video_width
    tracks = remove_small_detections(tracks)
    return tracks


def load_track_dataframes(group_dir: str):
    """
    Load track dataframes for all videos in a group directory.
    
    :param group_dir: Path to the directory containing track parquet files
    :return: Tuple of (track_dataframes, video_ids) where track_dataframes is a dictionary mapping 
             video IDs to track dataframes and video_ids is a list of video IDs
    """
    video_ids = os.path.basename(group_dir).split("_")
    video_ids = [int(video_id) for video_id in video_ids]
    track_dataframes = {}
    for video_id in video_ids:
        # parquet_path = os.path.join(group_dir, f"tracks_{video_id}.parquet")
        parquet_path = os.path.join(group_dir, f"processed_tracks_{video_id}.parquet")
        # parquet_path = os.path.join(group_dir, f"tracks_{video_id}_dinov2.parquet")
        video_path = os.path.join(group_dir, f"video_{video_id}.mp4")
        track_dataframes[video_id] = load_tracks(parquet_path, video_path)
    return track_dataframes, video_ids


def load_true_global_id_maps(group_dir: str) -> Dict[int, Dict[int, int]]:
    """
    Load ground truth global ID maps from a JSON file.
    
    :param group_dir: Path to the directory containing the ground_truth_global_id_maps.json file
    :return: Dictionary mapping video IDs to dictionaries mapping track IDs to global IDs,
             or None if the file doesn't exist
    
    Example returned format:
    {
        824719: {0: 0, 1: 0},  # video 824719's track 0 -> global ID 0, track 1 -> global ID 0
        824714: {0: 1, 1: 0}   # video 824714's track 0 -> global ID 1, track 1 -> global ID 0
    }
    """
    json_path = os.path.join(group_dir, "ground_truth_global_id_maps.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
        return {
            int(video_id): {
                int(track_id): int(global_id) for track_id, global_id in global_id_map.items()
            } for video_id, global_id_map in data.items()
        }


def assign_global_ids(track_dataframes: Dict[int, pd.DataFrame], global_id_maps: Dict[int, Dict[int, int]], data_type: Literal["predicted", "ground_truth"]):
    """
    Assign global IDs to tracks based on provided mapping.
    
    :param track_dataframes: Dictionary mapping video IDs to track dataframes
    :param global_id_maps: Dictionary mapping video IDs to dictionaries mapping track IDs to global IDs
    :param data_type: Either "predicted" or "ground_truth" to determine column name
    :return: Dictionary mapping video IDs to track dataframes with global IDs assigned
    
    Example:
    ```
    track_dataframes = {
        824719: pd.DataFrame(...),  # tracks from video 824719
        824714: pd.DataFrame(...)   # tracks from video 824714
    }
    global_id_maps = {
        824719: {0: 0, 1: 0},  # video 824719's track 0 -> global ID 0, track 1 -> global ID 0
        824714: {0: 1, 1: 0}   # video 824714's track 0 -> global ID 1, track 1 -> global ID 0
    }
    track_dataframes = assign_global_ids(track_dataframes, global_id_maps, "predicted")
    ```
    """
    if data_type == "predicted":
        column_name = "predicted_global_id"
    else:
        column_name = "true_global_id"
    global_id_maps = {int(video_id): global_id_map for video_id, global_id_map in global_id_maps.items()}
    for video_id, tracks in track_dataframes.items():
        if video_id in global_id_maps:
            tracks[column_name] = tracks["track_id"].map(global_id_maps[video_id])
        else:
            tracks[column_name] = np.nan
    return track_dataframes


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group_dir", type=str, required=True, 
        help="Path to the directory containing the group of video and track files"
    )
    args = parser.parse_args()
    args.group_dir = args.group_dir.rstrip("/")

    track_dataframes, video_ids = load_track_dataframes(args.group_dir)
    print("track_dataframes", track_dataframes)
    print("video_ids", video_ids)

    true_global_id_maps = load_true_global_id_maps(args.group_dir)
    print("true_global_id_maps", true_global_id_maps)

    track_dataframes = assign_global_ids(track_dataframes, true_global_id_maps, "ground_truth")
    print("track_dataframes", track_dataframes)
