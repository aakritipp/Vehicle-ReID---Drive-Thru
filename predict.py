import argparse
from helpers import assign_global_ids
from scipy.spatial.distance import euclidean, cityblock, minkowski, correlation
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
from typing import Dict
import helpers


def compute_feature_distance(vec1, vec2, metric='euclidean', **kwargs):
    """
    Compute distance between two feature vectors using the specified metric.

    Args:
        vec1, vec2: Feature vectors (numpy arrays)
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski', 'correlation', 'cosine')
        **kwargs: Additional parameters (e.g., p for Minkowski, cov_matrix for Mahalanobis)

    Returns:
        Distance value
    """
    if metric == 'euclidean':
        return euclidean(vec1, vec2)
    elif metric == 'manhattan':
        return cityblock(vec1, vec2)
    elif metric == 'minkowski':
        p = kwargs.get('p', 2)  # Default to Euclidean (p=2) if p not provided
        return minkowski(vec1, vec2, p)
    elif metric == 'correlation':
        return correlation(vec1, vec2)
    elif metric == 'cosine':
        return cosine(vec1, vec2)
    elif metric == 'mahalanobis':
        cov_matrix = kwargs.get('cov_matrix', np.identity(len(vec1)))
        diff = vec1 - vec2
        return np.sqrt(diff.T @ np.linalg.inv(cov_matrix) @ diff)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def estimate_velocity(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate velocity for each track based on world coordinates and time.

    :param tracks: DataFrame with track data including utc_time, world_x, world_y
    :return: DataFrame with added velocity_x, velocity_y columns (in meters/second)
    """
    tracks = tracks.sort_values(['track_id', 'utc_time'])
    tracks['time_diff'] = tracks.groupby('track_id')['utc_time'].diff() / 1000.0  # desa # Convert ms to seconds
    tracks['dx'] = tracks.groupby('track_id')['world_x'].diff()
    tracks['dy'] = tracks.groupby('track_id')['world_y'].diff()
    tracks['velocity_x'] = tracks['dx'] / tracks['time_diff']
    tracks['velocity_y'] = tracks['dy'] / tracks['time_diff']
    tracks['slope'] = tracks['dx'] / tracks['dy']
    # Fill NaN values (first entry of each track) with 0
    tracks[['velocity_x', 'velocity_y', 'slope']] = tracks[['velocity_x', 'velocity_y', 'slope']].fillna(0)
    return tracks


class SimpleKalmanFilter:
    """
    A simple Kalman filter implementation using just NumPy.
    Tracks position and velocity in 2D space.
    """

    def __init__(self, x, y, vx=0, vy=0, dt=1.0):
        # State: [x, y, vx, vy]
        self.x = np.array([x, y, vx, vy])

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],  # x = x + vx*dt
            [0, 1, 0, dt],  # y = y + vy*dt
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]  # vy = vy
        ])

        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Covariance matrices
        self.R = np.eye(2) * 0.1  # Measurement noise
        self.Q = np.eye(4) * 0.01  # Process noise

        # State covariance
        self.P = np.eye(4) * 100  # High initial uncertainty

        # Identity matrix
        self.I = np.eye(4)

    def predict(self, dt=None):
        """
        Predict the next state based on current state and model.

        Args:
            dt: Time step (optional, to update time in model)

        Returns:
            Predicted x, y position
        """
        if dt is not None:
            # Update time step in state transition matrix
            self.F[0, 2] = dt
            self.F[1, 3] = dt

        # Predict next state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[0], self.x[1]

    def update(self, z_x, z_y):
        """
        Update the filter with a new measurement.

        Args:
            z_x, z_y: Measured position

        Returns:
            Updated state estimate
        """
        # Measurement
        z = np.array([z_x, z_y])

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update state covariance
        self.P = (self.I - K @ self.H) @ self.P

        return self.x[0], self.x[1]


def group_tracks_in_time_window_with_kalman(tracks: pd.DataFrame, time_window_ms: float, max_distance_m: float,
                                            feature_similarity_threshold: float) -> Dict[int, int]:
    """
    Group tracks by first identifying the last occurrence of each track_id and checking
    forward within a 2-second window for another track with similar appearance and feasible
    spatial proximity.

    :param tracks: DataFrame with track data
    :param time_window_ms: Time window in milliseconds for looking forward
    :param max_distance_m: Maximum distance a vehicle can travel in the time window
    :param feature_similarity_threshold: Maximum cosine distance for feature vectors
    :return: Dictionary mapping track_id to global_id
    """

    # Ensure tracks are sorted by time
    tracks = tracks.sort_values(['track_id', 'utc_time'])

    # Initialize global ID assignments
    global_id_map = {}
    current_global_id = -1

    # Estimate velocities
    tracks = estimate_velocity(tracks)

    # Group tracks by track_id and get the last occurrence
    last_occurrences = tracks.groupby('track_id').last().reset_index()

    # Sort last occurrences by utc_time to process chronologically
    last_occurrences = last_occurrences.sort_values('utc_time')

    for _, last_track in last_occurrences.iterrows():
        track_id = last_track['track_id']
        # print(track_id, last_track['utc_time'])

        # Define the forward-looking time window
        start_time = last_track['utc_time']
        end_time = start_time + time_window_ms

        # Find candidate tracks within the time window
        candidate_tracks = tracks[
            (tracks['utc_time'] >= start_time) &
            (tracks['utc_time'] < end_time)
            ]

        candidate_tracks = candidate_tracks[candidate_tracks['track_id'] != track_id]

        if candidate_tracks.empty:
            if track_id not in global_id_map:
                current_global_id += 1
                global_id_map[track_id] = current_global_id
            continue

        # Check each candidate track for spatial and appearance similarity
        for candidate_id in candidate_tracks['track_id'].unique():
            candidate_data = candidate_tracks[candidate_tracks['track_id'] == candidate_id]
            candidate = candidate_data.iloc[0]  # Take the earliest detection in the window

            # Calculate spatial distance
            distance = np.sqrt(
                (candidate['world_x'] - last_track['world_x']) ** 2 +
                (candidate['world_y'] - last_track['world_y']) ** 2
            )

            # Estimate max possible distance based on velocity
            avg_velocity = np.sqrt(
                (last_track['velocity_x'] ** 2 + last_track['velocity_y'] ** 2)
            )
            time_diff_sec = (end_time - start_time) / 1000.0
            velocity_distance = avg_velocity * time_diff_sec

            # Use the larger of max_distance_m or velocity-based distance
            effective_max_distance = max(max_distance_m, velocity_distance)

            # Calculate feature similarity
            feature_vec1 = np.array(eval(last_track['new_feature_vector'])
                                    if isinstance(last_track['new_feature_vector'], str)
                                    else last_track['new_feature_vector'])
            feature_vec2 = np.array(eval(candidate['new_feature_vector'])
                                    if isinstance(candidate['new_feature_vector'], str)
                                    else candidate['new_feature_vector'])
            feature_distance = cosine(feature_vec1, feature_vec2)

            # print(track_id, candidate_id, distance, feature_distance, effective_max_distance)

            # If spatially close and features are similar, assign same global ID
            if (distance <= max_distance_m and
                    feature_distance <= feature_similarity_threshold):
                # print(candidate_id in global_id_map)
                if track_id in global_id_map or candidate_id in global_id_map:
                    if track_id in global_id_map:
                        global_id_map[candidate_id] = global_id_map[track_id]
                    else:
                        global_id_map[track_id] = global_id_map[candidate_id]
                else:
                    current_global_id += 1
                    global_id_map[track_id] = current_global_id
                    global_id_map[candidate_id] = current_global_id
                # print(global_id_map)
                # processed_track_ids.add(candidate_id)
            else:
                if track_id not in global_id_map:
                    current_global_id += 1
                    global_id_map[track_id] = current_global_id

    # Assign global IDs to any remaining unprocessed tracks
    for track_id in tracks['track_id'].unique():
        if track_id not in global_id_map:
            global_id_map[track_id] = current_global_id
            current_global_id += 1

    return global_id_map


def merge_global_ids_with_kalman(
        track_dataframes: Dict[int, pd.DataFrame],
        global_id_maps: Dict[int, Dict[int, int]],
        time_window_ms: float,
        max_distance_m: float,
        feature_similarity_threshold: float,
        overlap: bool,
        spatial_weight: float,
        feature_weight: float
) -> Dict[int, Dict[int, int]]:
    """
    Merge global IDs across different cameras using Kalman filter prediction
    to account for motion between camera views.

    Args:
        track_dataframes: Dictionary mapping video IDs to track DataFrames
        global_id_maps: Dictionary mapping video IDs to track ID to global ID mappings
        time_window_ms: Maximum time difference for considering timestamps to be close
        max_distance_m: Maximum distance in meters for spatial proximity
        feature_similarity_threshold: Maximum cosine distance for feature vectors

    Returns:
        Updated global_id_maps with consistent IDs across cameras
    """
    # Create a unified DataFrame with all tracks, including video_id
    all_tracks = []
    for video_id, tracks in track_dataframes.items():
        tracks = tracks.copy()
        tracks['video_id'] = video_id
        all_tracks.append(tracks)
    all_tracks = pd.concat(all_tracks, ignore_index=True)

    # Add current global_id to tracks based on global_id_maps
    all_tracks['global_id'] = all_tracks.apply(
        lambda row: global_id_maps[row['video_id']][row['track_id']], axis=1
    )

    # Initialize Kalman filters for each track in each video
    kalman_filters = {}  # (video_id, track_id) -> KalmanFilter

    for video_id, video_tracks in all_tracks.groupby('video_id'):
        for track_id, track_data in video_tracks.groupby('track_id'):
            track_data = track_data.sort_values('utc_time')

            # Initialize with first position
            first_row = track_data.iloc[0]
            x, y = first_row['world_x'], first_row['world_y']

            # Calculate initial velocity if possible
            vx, vy = 0, 0
            if len(track_data) > 1:
                second_row = track_data.iloc[1]
                dt = (second_row['utc_time'] - first_row['utc_time']) / 1000.0
                if dt > 0:
                    vx = (second_row['world_x'] - x) / dt
                    vy = (second_row['world_y'] - y) / dt

            # Create Kalman filter
            kf = SimpleKalmanFilter(x, y, vx, vy)

            # Update with all measurements
            prev_time = None
            for _, row in track_data.iterrows():
                if prev_time is not None:
                    dt = (row['utc_time'] - prev_time) / 1000.0
                    kf.predict(dt)
                kf.update(row['world_x'], row['world_y'])
                prev_time = row['utc_time']

            # Store the filter
            kalman_filters[(video_id, track_id)] = kf

    # Initialize a mapping to track merged global IDs
    global_id_remap = {}

    # Get list of video IDs
    video_ids = list(track_dataframes.keys())

    # Create a list to store all potential matches
    all_matches = []

    # Calculate scores for all possible pairs
    for i, video_id1 in enumerate(video_ids):
        for video_id2 in video_ids[i + 1:]:
            # Get tracks from each video
            tracks1 = all_tracks[all_tracks['video_id'] == video_id1]
            tracks2 = all_tracks[all_tracks['video_id'] == video_id2]

            # For each global_id in video 1
            for global_id1, track1_with_same_global in tracks1.groupby('global_id'):
                # For each global_id in video 2
                for global_id2, track2_with_same_global in tracks2.groupby('global_id'):

                    # For each track with the same global_id in video 1
                    for track_id1, track1_data in track1_with_same_global.groupby('track_id'):
                        # Get last position and time
                        track1_data = track1_data.sort_values('utc_time')
                        last_row = track1_data.iloc[-1]
                        last_time = last_row['utc_time']

                        # Get Kalman filter for this track
                        kf1 = kalman_filters[(video_id1, track_id1)]

                        # Extract feature vector
                        feature_vec_str = last_row['new_feature_vector']
                        last_feature = np.array(
                            eval(feature_vec_str) if isinstance(feature_vec_str, str) else feature_vec_str)

                        # For each track with the same global_id in video 2
                        for track_id2, track2_data in track2_with_same_global.groupby('track_id'):
                            # Get first position and time
                            track2_data = track2_data.sort_values('utc_time')
                            first_row = track2_data.iloc[0]
                            first_time = first_row['utc_time']

                            # Display track and time info
                            # print(track_id1, track_id2, first_time - last_time)

                            # Check time window
                            time_diff = first_time - last_time
                            if overlap:
                                time_diff = abs(time_diff)
                            if not (0 <= time_diff <= time_window_ms):
                                continue

                            # Get first feature vector
                            feature_vec_str = first_row['new_feature_vector']
                            first_feature = np.array(
                                eval(feature_vec_str) if isinstance(feature_vec_str, str) else feature_vec_str)

                            # Calculate feature similarity
                            feature_distance = cosine(last_feature, first_feature)

                            # Use Kalman filter to predict position
                            dt = time_diff / 1000.0  # Convert ms to seconds

                            # Create a temporary Kalman filter with the same state as kf1
                            kf_temp = SimpleKalmanFilter(kf1.x[0], kf1.x[1], kf1.x[2], kf1.x[3])
                            kf_temp.P = kf1.P.copy()

                            # Predict using the temporary filter
                            predicted_x, predicted_y = kf_temp.predict(dt)

                            # Calculate spatial distance
                            predicted_distance = np.sqrt(
                                (first_row['world_x'] - predicted_x) ** 2 +
                                (first_row['world_y'] - predicted_y) ** 2
                            )

                            combined_score = (spatial_weight * predicted_distance / max_distance_m +
                                              feature_weight * feature_distance / feature_similarity_threshold)

                            # Display detailed score info
                            # print(track_id1, track_id2, predicted_distance, feature_distance, combined_score)

                            # Check if this is a valid match
                            if (
                                    predicted_distance <= max_distance_m and feature_distance <= feature_similarity_threshold):
                                # Store the match details
                                match_info = {
                                    'video_id1': video_id1,
                                    'video_id2': video_id2,
                                    'global_id1': global_id1,
                                    'global_id2': global_id2,
                                    'track_id1': track_id1,
                                    'track_id2': track_id2,
                                    'combined_score': combined_score,
                                    # 'predicted_distance': predicted_distance,
                                    # 'feature_distance': feature_distance
                                }
                                all_matches.append(match_info)

    # Sort matches by score (best matches first)
    all_matches.sort(key=lambda x: x['combined_score'])

    # print("all matches")
    # print(all_matches)

    # Process matches in order of score
    global_id_remap = {}  # Final global ID mapping
    orphans = []  # Track orphaned tracks
    processed_global_ids = set()  # Keep track of processed global IDs

    for match in all_matches:
        video_id1 = match['video_id1']
        video_id2 = match['video_id2']
        global_id1 = match['global_id1']
        global_id2 = match['global_id2']
        track_id1 = match['track_id1']
        track_id2 = match['track_id2']

        # Skip if either global ID has already been processed
        if (video_id1, global_id1) in processed_global_ids or (video_id2, global_id2) in processed_global_ids:
            # Record orphaned track info
            if (video_id1, global_id1) not in processed_global_ids:
                orphans.append((video_id1, global_id1))
                # print(f"Track (video {video_id1}, global {global_id1}) orphaned as a better match was found")
            elif (video_id2, global_id2) not in processed_global_ids:
                orphans.append((video_id2, global_id2))
                # print(f"Track (video {video_id2}, global {global_id2}) orphaned as a better match was found")
            continue

        # Create a new merged ID
        new_global_id = min(global_id1, global_id2) * 1000 + max(global_id1, global_id2)

        # Update the mapping
        global_id_remap[(video_id1, global_id1)] = new_global_id
        global_id_remap[(video_id2, global_id2)] = new_global_id

        # Mark these global IDs as processed
        processed_global_ids.add((video_id1, global_id1))
        processed_global_ids.add((video_id2, global_id2))

        # print(f"Matched tracks {track_id1} (video {video_id1}) and {track_id2} (video {video_id2})")
        # print(f"  Global IDs: {global_id1} and {global_id2} -> {new_global_id}")
        # print(f"  Score: {match['combined_score']:.4f}")

    # Update global_id_maps with merged IDs
    updated_global_id_maps = {}
    for video_id in global_id_maps:
        updated_global_id_maps[video_id] = {}
        for track_id, global_id in global_id_maps[video_id].items():
            key = (video_id, global_id)
            if key in global_id_remap:
                updated_global_id_maps[video_id][track_id] = global_id_remap[key]
            else:
                updated_global_id_maps[video_id][track_id] = global_id

    return updated_global_id_maps


def predict_global_ids(track_dataframes: Dict[int, pd.DataFrame], facility_id) -> Dict[int, Dict[int, int]]:
    """
    Predict global IDs for tracks across all videos using Kalman filtering.

    Args:
        track_dataframes: Dictionary mapping video IDs to track DataFrames
        facility_id: Facility ID for parameter selection

    Returns:
        Dictionary mapping video IDs to track ID to global ID mappings
    """
    # Set parameters based on facility ID
    if facility_id == 29:
        time_window_ms = 5000.0
        max_distance_m = 5
        feature_similarity_threshold = 0.35
        overlap = False
        time_window_ms_across = 5000000  # Longer time window for cross-camera
        max_distance_m_across = 100  # Larger distance threshold for cross-camera
        spatial_weight = 0.35
        feature_weight = 0.65
    elif facility_id == 34:
        time_window_ms = 3000.0
        max_distance_m = 1.5
        feature_similarity_threshold = 0.35
        overlap = False
        time_window_ms_across = 100000  # Longer time window for cross-camera
        max_distance_m_across = 50
        spatial_weight = 0.6
        feature_weight = 0.4
    elif facility_id == 41:
        time_window_ms = 5000.0
        max_distance_m = 3
        feature_similarity_threshold = 0.35
        overlap = True
        time_window_ms_across = 100000  # Longer time window for cross-camera
        max_distance_m_across = 50
        spatial_weight = 0.6
        feature_weight = 0.4
    elif facility_id == 53:
        time_window_ms = 3000.0
        max_distance_m = 4
        feature_similarity_threshold = 0.35
        overlap = False
        time_window_ms_across = 100000  # Longer time window for cross-camera
        max_distance_m_across = 50
        spatial_weight = 0.6
        feature_weight = 0.4
    else:
        time_window_ms = 3000.0
        max_distance_m = 1.5
        feature_similarity_threshold = 0.35
        overlap = True
        time_window_ms_across = 200000  # Longer time window for cross-camera
        max_distance_m_across = 25
        spatial_weight = 0.6
        feature_weight = 0.4

    # Process each video individually first
    global_id_maps = {}
    for video_id, tracks in track_dataframes.items():
        # Group tracks within this video using Kalman filter
        global_id_map = group_tracks_in_time_window_with_kalman(
            tracks, time_window_ms, max_distance_m, feature_similarity_threshold
        )
        global_id_maps[video_id] = global_id_map

    # print("Initial global ID maps:", global_id_maps)

    # Merge global IDs across cameras using Kalman filter predictions
    predicted_global_id_maps = merge_global_ids_with_kalman(
        track_dataframes,
        global_id_maps,
        time_window_ms=time_window_ms_across,  # Longer time window for cross-camera
        max_distance_m=max_distance_m_across,  # Larger distance threshold for cross-camera
        feature_similarity_threshold=1,  # Stricter feature similarity for cross-camera
        overlap=overlap,
        spatial_weight=spatial_weight,
        feature_weight=feature_weight
    )

    return predicted_global_id_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group_dir", type=str, required=True,
        help="Path to the directory containing the group of video and track files"
    )
    args = parser.parse_args()
    facility_id = args.group_dir.split("/")[0]
    facility_id = int(facility_id.split("_")[1])
    args.group_dir = args.group_dir.rstrip("/")

    # print(facility_id)

    # Load track data
    track_dataframes, video_ids = helpers.load_track_dataframes(args.group_dir)

    # Predict global IDs
    predicted_global_id_maps = predict_global_ids(track_dataframes, facility_id)
    # print(predicted_global_id_maps)

    # Assign predicted global IDs to DataFrames
    track_dataframes = assign_global_ids(
        track_dataframes, predicted_global_id_maps, "predicted"
    )


if __name__ == "__main__":
    main()
