"""
MediaPipe keypoint extraction from video files.
Extracts: 40 lip landmarks, 21 left hand, 21 right hand, 5 pose points.
Output shape per video: (N_frames, 87, 3) where 3 = (x, y, z).
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Same landmark indices as in the original project
LIPS_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308,
])
LEFT_HAND_IDXS = np.arange(21)   # MediaPipe hand has 21 landmarks
RIGHT_HAND_IDXS = np.arange(21)
# Pose: left/right shoulder, elbow, wrist (indices 11-16 in mediapipe pose)
POSE_IDXS = np.array([11, 12, 13, 14, 15])  # left shoulder, right shoulder, left elbow, right elbow, left wrist

N_LANDMARKS = 40 + 21 + 21 + 5  # 87 total


def extract_keypoints_from_video(video_path: str, max_frames: int = 256) -> np.ndarray:
    """
    Extract MediaPipe holistic keypoints from a video file.

    Returns:
        keypoints: np.ndarray of shape (N_frames, 87, 3)
        If extraction fails, returns array of zeros.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((1, N_LANDMARKS, 3), dtype=np.float32)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If video has more frames than max_frames, sample uniformly
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)

    all_keypoints = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        frame_idx = 0
        sample_idx = 0

        while cap.isOpened() and sample_idx < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == frame_indices[sample_idx]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                kp = _results_to_array(results)
                all_keypoints.append(kp)
                sample_idx += 1

            frame_idx += 1

    cap.release()

    if len(all_keypoints) == 0:
        return np.zeros((1, N_LANDMARKS, 3), dtype=np.float32)

    return np.stack(all_keypoints, axis=0).astype(np.float32)


def _results_to_array(results) -> np.ndarray:
    """Convert MediaPipe Holistic results to (87, 3) array."""
    kp = np.zeros((N_LANDMARKS, 3), dtype=np.float32)

    # Lips from face mesh (40 points)
    if results.face_landmarks:
        for i, idx in enumerate(LIPS_IDXS):
            lm = results.face_landmarks.landmark[idx]
            kp[i] = [lm.x, lm.y, lm.z]

    # Left hand (21 points)
    offset = 40
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            kp[offset + i] = [lm.x, lm.y, lm.z]

    # Right hand (21 points)
    offset = 61
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            kp[offset + i] = [lm.x, lm.y, lm.z]

    # Pose (5 points)
    offset = 82
    if results.pose_landmarks:
        for i, idx in enumerate(POSE_IDXS):
            lm = results.pose_landmarks.landmark[idx]
            kp[offset + i] = [lm.x, lm.y, lm.z]

    return kp


def batch_extract_keypoints(
    video_dir: str,
    output_dir: str,
    num_workers: int = 4,
    max_frames: int = 256,
):
    """
    Extract keypoints from all videos in a directory tree.
    Saves each as {attachment_id}.npy in output_dir.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(video_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} videos in {video_dir}")

    # Single-process because MediaPipe doesn't play well with multiprocessing
    for vf in tqdm(video_files, desc="Extracting keypoints"):
        out_path = output_dir / f"{vf.stem}.npy"
        if out_path.exists():
            continue
        kp = extract_keypoints_from_video(str(vf), max_frames=max_frames)
        np.save(str(out_path), kp)

    print(f"Done. Saved keypoints to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Path to SLOVO video directory")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy keypoints")
    parser.add_argument("--max_frames", type=int, default=256)
    args = parser.parse_args()
    batch_extract_keypoints(args.video_dir, args.output_dir, max_frames=args.max_frames)
