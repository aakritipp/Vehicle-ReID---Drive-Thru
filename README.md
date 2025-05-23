# Trace Candidate Project Vehicle Re-ID

System for tracking vehicles across multiple camera views using DINOv2 feature extraction and direction-aware feature aggregation, Kalman filtering.

## Overview

This project provides a complete solution for tracking and identifying vehicles across multiple non-overlapping camera views. The system uses state-of-the-art computer vision techniques to extract visual features from vehicle images, predict their trajectories, and maintain consistent vehicle identities across different camera perspectives.

Key features:
- Vehicle classification using a neural network model
- Feature extraction with DINOv2 (Vision Transformer)
- Direction-aware feature aggregation to handle different viewing angles
- Kalman filtering for trajectory prediction and cross-camera matching
- Evaluation framework with precision, recall, and end-to-end accuracy metrics

## System Requirements

- Python 3.8+
- PyTorch 1.8+
- TensorFlow 2.4+
- CUDA-capable GPU (recommended)
- OpenCV 4.5+

## Usage

### Basic Usage

Process a group of videos with:

```bash
python dinov2_multiview.py --group_dir /path/to/group_directory
```
To evalute the results:
```bash
python evaluate.py --group_dir /path/to/group_directory
```

Where a group directory contains:
- `video_{video_id}.mp4` files - Raw video files
- `tracks_{video_id}.parquet` files - Initial track data
- `ground_truth_global_id_maps.json` - Ground truth for evaluation



### Complete Workflow

For a complete tracking workflow:

1. **Pre-process and classify vehicles**:
   ```bash
   python car_classifier.py --group_dir /path/to/group_directory
   ```

2. **Extract features with DINOv2**:
   ```bash
   python dinov2_multiview.py --group_dir /path/to/group_directory
   ```

3. **Generate global IDs for tracks across cameras**:
   ```bash
   python predict.py --group_dir /path/to/group_directory
   ```

4. **Evaluate tracking performance** (if ground truth available):
   ```bash
   python evaluate.py --group_dir /path/to/group_directory
   ```

## System Architecture

The system follows this processing pipeline:

1. **Track Extraction**: Initial vehicle tracks are extracted from video feeds
2. **Car Classification**: Neural network separates cars from other objects
3. **Feature Extraction**: DINOv2 extracts robust visual features from vehicle images
4. **Direction Estimation**: Vehicle direction is estimated from bounding box movement
5. **Feature Aggregation**: Features from multiple views are aggregated with direction weighting
6. **Within-Camera Tracking**: Tracks within each camera are linked using time and feature similarity
7. **Cross-Camera Tracking**: Kalman filter prediction links tracks across different cameras
8. **Evaluation**: System performance is evaluated against ground truth data

## File Descriptions

### Core Modules

- **dinov2_multiview.py**: Implements feature extraction using DINOv2 Vision Transformer with direction-aware feature aggregation
- **car_classifier.py**: Neural network classifier to separate cars from other objects
- **predict.py**: Implements Kalman filtering for track prediction and global ID assignment
- **evaluate.py**: Evaluation metrics for tracking performance
- **helpers.py**: Utility functions for data loading and preprocessing

## Data Format

### Input Data

- **Video Files**: MP4 video files from different camera perspectives
- **Track Files**: Parquet files containing detection and tracking data with the following columns:
  - `track_id`: Unique identifier for each track
  - `frame_index`: Index of the frame in the video
  - `u1, u2, v1, v2`: Bounding box coordinates
  - `utc_time`: Timestamp in milliseconds
  - `world_x, world_y`: World coordinates in meters
  - `feature_vector`: Initial visual feature vector

### Output Data

- **Processed Track Files**: Enhanced track files with:
  - `new_feature_vector`: DINOv2 feature vector with direction-aware processing
  - `predicted_global_id`: System-assigned global identifier
  - Additional motion and appearance information

## Key Components

### DINOv2 Feature Extraction

The system uses DINOv2, a state-of-the-art Vision Transformer model, to extract robust visual features from vehicle images. These features are invariant to many visual transformations, making them ideal for re-identification across different viewpoints.

```python
# Extract features with DINOv2
features = extract_dinov2_features(model, images, device)
```

### Direction-Aware Feature Aggregation

To handle different viewing angles, the system estimates vehicle direction and weights features accordingly:
This has been set as unknown currently and have not been evaluated extensively for different facility making the confidence on features low.

```python
direction = estimate_direction(track_data)
aggregated_feature = aggregate_features(feature_vectors, direction, camera_directions, 
                                        video_ids_present, total_cameras)
```

### Kalman Filtering for Tracking

The system implements a Kalman filter to predict vehicle trajectories, enabling reliable tracking across cameras:

```python
kf = SimpleKalmanFilter(x, y, vx, vy)
predicted_x, predicted_y = kf.predict(dt)
```

### Within-Camera Re-identification
Time-window based tracking: Identifies the last occurrence of each track and looks for potential continuations within a configurable time window
Spatial proximity analysis: Calculates feasible distance thresholds based on vehicle velocity and expected movement patterns
Feature similarity comparison: Compares visual features using cosine similarity to ensure consistent vehicle appearance even when track IDs change

```python
within_camera_global_id = group_tracks_in_time_window_with_kalman(tracks, time_window_ms, max_distance_m,
                                            feature_similarity_threshold)
```

### Across-Camera Re-identification
For maintaining vehicle identity across different camera views, the system employs:

Cross-camera matching: Compares tracks between camera pairs, using Kalman prediction to bridge spatial-temporal gaps
Combined similarity scoring: Integrates spatial proximity (weighted by prediction confidence) and feature similarity into a unified matching score
Prioritized processing: Processes potential matches in order of best matching score to ensure optimal assignments
Global ID consistency: Creates a consistent mapping of track IDs to global IDs across all cameras

Note: Facility 53 is not working currently
```python
final_global_id = merge_global_ids_with_kalman(track_dataframes, global_id_maps, time_window_ms, max_distance_m, 
                                               feature_similarity_threshold, overlap, spatial_weight, feature_weight)
```

## Evaluation Metrics

The system calculates these key metrics:
- **Precision**: Proportion of predicted groups where all tracks belong to the same ground truth object
- **Recall**: Proportion of ground truth groups where all tracks are assigned the same predicted global ID
- **End-to-End Accuracy**: Proportion of ground truth objects appearing in all videos that are consistently grouped

## How Evaluation Works
The evaluation algorithm focuses on grouping consistency rather than exact ID matching:

For each ground truth global ID (which identifies a set of tracks that should be grouped together), it checks if those exact same tracks are grouped together in the predictions. If they are, the grouping is considered correct.
If tracks that should be together are split across different predicted IDs, the tracks in the majority predicted group are considered correct (True Positives) and the rest are marked as incorrect (False Negatives).
From the inverse perspective: for each predicted global ID, it verifies if it contains tracks that all belong to the same ground truth global ID. The tracks that don't belong to the majority ground truth ID are considered incorrect (False Positives).

This approach ensures that even though the system generates different ID values than the ground truth, the evaluation still accurately measures if the system is correctly identifying the same vehicle across multiple cameras.