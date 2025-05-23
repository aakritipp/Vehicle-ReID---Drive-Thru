"""
Evaluate the accuracy of predicted global IDs against ground truth global IDs, focusing on grouping consistency.
"""
import argparse
import pandas as pd
from typing import Dict

import helpers
import predict


def evaluate_predictions(track_dataframes: Dict[int, pd.DataFrame]) -> Dict[str, float]:
    """
    Evaluate the accuracy of track grouping consistency between predicted and ground truth global IDs.

    This function evaluates whether tracks that should be grouped together (according to ground truth)
    are actually grouped together in the predictions, regardless of the specific global ID values used.

    The metrics returned are:
    - precision: Proportion of predicted groups where all tracks belong to the same ground truth object.
    - recall: Proportion of ground truth groups where all tracks are assigned the same predicted global ID.
    - end_to_end_accuracy: Proportion of ground truth objects that appear in all videos and are consistently
      grouped under a single predicted global ID across all videos.

    :param track_dataframes: Dictionary mapping video IDs to track dataframes
    :return: Dictionary containing precision, recall, and end-to-end accuracy metrics
    """
    # Concatenate all tracks from different videos
    concat_tracks = pd.concat([df.assign(video_id=video_id) for video_id, df in track_dataframes.items()])

    # Check if necessary columns exist
    if "predicted_global_id" not in concat_tracks.columns or "true_global_id" not in concat_tracks.columns:
        print("Missing required columns: 'predicted_global_id' or 'true_global_id'")
        return {"precision": 0, "recall": 0, "end_to_end_accuracy": 0}

    # Filter out rows without ground truth global_id
    concat_tracks = concat_tracks.dropna(subset=["true_global_id"])

    if concat_tracks.empty:
        print("No ground truth global IDs found for evaluation")
        return {"precision": 0, "recall": 0, "end_to_end_accuracy": 0}

    # Create mapping dictionaries to track which tracks belong to which groups
    # This helps us focus on grouping consistency rather than exact ID matching
    true_id_to_group = {}
    pred_id_to_group = {}

    # Assign group IDs to each unique true_global_id
    for idx, true_id in enumerate(concat_tracks["true_global_id"].unique()):
        true_id_to_group[true_id] = idx

    # Assign group IDs to each unique predicted_global_id
    for idx, pred_id in enumerate(concat_tracks["predicted_global_id"].dropna().unique()):
        pred_id_to_group[pred_id] = idx

    # Map the original IDs to group IDs
    concat_tracks["true_group"] = concat_tracks["true_global_id"].map(true_id_to_group)
    concat_tracks["pred_group"] = concat_tracks["predicted_global_id"].map(pred_id_to_group)

    TP = 0  # True Positives: Correctly grouped tracks
    FP = 0  # False Positives: Predicted groups that mix multiple ground truth objects
    FN = 0  # False Negatives: Ground truth groups that are split across multiple predicted IDs
    end_to_end_correct = 0
    end_to_end_total = 0

    # Evaluate Recall: For each ground truth group, check if all tracks are assigned the same predicted group
    for true_group, group_data in concat_tracks.groupby("true_group"):
        # Get the predicted groups for this ground truth group
        pred_groups = group_data["pred_group"].dropna()
        if pred_groups.empty:
            FN += len(group_data)
            continue

        # Count unique predicted groups in this ground truth group
        unique_pred_groups = pred_groups.nunique()
        total_tracks = len(pred_groups)

        if unique_pred_groups == 1:
            # All tracks in this ground truth group have the same predicted group (correct grouping)
            TP += total_tracks
        else:
            # Tracks are split across multiple predicted groups (incorrect grouping)
            # For each predicted group in this true group, count how many tracks don't belong to the majority
            pred_group_counts = pred_groups.value_counts()
            majority_pred_group = pred_group_counts.idxmax()
            majority_count = pred_group_counts.max()
            TP += majority_count  # Tracks in the majority group are considered correct
            FN += (total_tracks - majority_count)  # Tracks not in the majority group are false negatives

    # Evaluate Precision: For each predicted group, check if all tracks belong to the same ground truth group
    for pred_group, group_data in concat_tracks.groupby("pred_group"):
        # Get the ground truth groups for this predicted group
        true_groups = group_data["true_group"].dropna()
        if true_groups.empty:
            continue

        # Count unique ground truth groups in this predicted group
        unique_true_groups = true_groups.nunique()
        total_tracks = len(true_groups)

        if unique_true_groups == 1:
            # All tracks in this predicted group belong to the same ground truth group (correct grouping)
            # These tracks are already counted in TP from the recall calculation
            pass
        else:
            # This predicted group mixes tracks from multiple ground truth groups (incorrect grouping)
            true_group_counts = true_groups.value_counts()
            majority_true_group = true_group_counts.idxmax()
            majority_count = true_group_counts.max()
            FP += (total_tracks - majority_count)  # Tracks not in the majority true group are false positives

    # Evaluate End-to-End Accuracy: Check if ground truth objects that span all videos are consistently grouped
    for true_group, group_data in concat_tracks.groupby("true_group"):
        # Check if this ground truth object appears in all videos
        if group_data["video_id"].nunique() == len(track_dataframes):
            end_to_end_total += 1
            # Get the predicted groups across all videos
            pred_groups_per_video = {}
            for video_id in track_dataframes.keys():
                video_group = group_data[group_data["video_id"] == video_id]
                pred_groups = set(video_group["pred_group"].dropna())
                pred_groups_per_video[video_id] = pred_groups

            # Check if the predicted group is consistent across all videos
            consistent = True
            for video_id, pred_groups in pred_groups_per_video.items():
                if not pred_groups:
                    consistent = False
                    break
                if len(pred_groups) > 1:
                    consistent = False
                    break

            if consistent:
                # Get the single predicted group ID from each video
                pred_group_ids = [list(groups)[0] for groups in pred_groups_per_video.values()]
                # Check if all videos have the same predicted group ID
                if len(set(pred_group_ids)) == 1:
                    # Ensure this predicted group isn't also assigned to tracks from another ground truth object
                    pred_group_id = pred_group_ids[0]
                    pred_tracks = concat_tracks[concat_tracks["pred_group"] == pred_group_id]
                    if pred_tracks["true_group"].nunique() == 1:
                        end_to_end_correct += 1

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    end_to_end_accuracy = end_to_end_correct / end_to_end_total if end_to_end_total > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "end_to_end_accuracy": end_to_end_accuracy,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_total": end_to_end_total,
    }


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_dir", type=str, required=True)
    args = parser.parse_args()
    facility_id = args.group_dir.split("/")[0]
    facility_id = int(facility_id.split("_")[1])
    args.group_dir = args.group_dir.rstrip("/")

    track_dataframes, video_ids = helpers.load_track_dataframes(args.group_dir)
    true_global_id_maps = helpers.load_true_global_id_maps(args.group_dir)
    track_dataframes = helpers.assign_global_ids(track_dataframes, true_global_id_maps, "ground_truth")
    predicted_global_id_maps = predict.predict_global_ids(track_dataframes, facility_id)
    # print(predicted_global_id_maps)
    track_dataframes = helpers.assign_global_ids(track_dataframes, predicted_global_id_maps, "predicted")
    # print(track_dataframes)
    metrics = evaluate_predictions(track_dataframes)
    print("metrics", metrics)