"""
SOTA keypoint extraction using RTMW-x (COCO-WholeBody format).

RTMW-x achieves 0.702 Whole AP, 0.664 Hand AP on COCO-WholeBody —
significantly better than MediaPipe, especially for rotated hands.

Uses `rtmlib` — lightweight wrapper, no mmpose/mmcv dependency needed.
    pip install rtmlib onnxruntime-gpu

COCO-WholeBody 133 keypoints layout:
    [0:17]    Body (17 points)
    [17:23]   Feet (6 points)
    [23:91]   Face (68 points)  — includes lips [71:91] = outer [71:83] + inner [83:91]
    [91:112]  Left hand (21 points)
    [112:133] Right hand (21 points)

We extract and save ALL 133 keypoints + confidence scores.
The model selects relevant subsets during training.

Output per video: keypoints (N_frames, 133, 2), scores (N_frames, 133)
Saved as a dict: {"keypoints": ..., "scores": ...}
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---- COCO-WholeBody index map (for reference and model usage) ----
# Body
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Relevant subsets for Sign Language Recognition
UPPER_BODY_IDXS = np.array([5, 6, 7, 8, 9, 10])          # shoulders, elbows, wrists (6)
FACE_OFFSET = 23                                            # face starts at index 23
EYEBROW_IDXS = np.arange(23 + 17, 23 + 27)                # left+right eyebrows (10)
LIP_OUTER_IDXS = np.arange(23 + 48, 23 + 60)              # outer lip contour (12)
LIP_INNER_IDXS = np.arange(23 + 60, 23 + 68)              # inner lip contour (8)
LIPS_IDXS = np.concatenate([LIP_OUTER_IDXS, LIP_INNER_IDXS])  # all lips (20)
FACE_SLR_IDXS = np.concatenate([EYEBROW_IDXS, LIPS_IDXS])     # eyebrows + lips (30)
LEFT_HAND_IDXS = np.arange(91, 112)                        # left hand (21)
RIGHT_HAND_IDXS = np.arange(112, 133)                      # right hand (21)

# Combined subset for SLR: 30 face + 21 left hand + 21 right hand + 6 upper body = 78
SLR_IDXS = np.concatenate([FACE_SLR_IDXS, LEFT_HAND_IDXS, RIGHT_HAND_IDXS, UPPER_BODY_IDXS])
N_SLR_LANDMARKS = len(SLR_IDXS)  # 78

# Ranges within the SLR subset (for the model)
SLR_FACE_RANGE = (0, 30)            # eyebrows (10) + lips (20)
SLR_LEFT_HAND_RANGE = (30, 51)      # 21 points
SLR_RIGHT_HAND_RANGE = (51, 72)     # 21 points
SLR_POSE_RANGE = (72, 78)           # 6 points


def _init_wholebody(device="cuda"):
    """Initialize RTMW wholebody model via rtmlib."""
    from rtmlib import Wholebody

    wholebody = Wholebody(
        to_openpose=False,
        mode="balanced",      # balanced = best accuracy/speed tradeoff
        backend="onnxruntime",
        device=device,
    )
    return wholebody


def extract_keypoints_from_video(
    video_path: str,
    wholebody,
    max_frames: int = 256,
    save_full: bool = False,
) -> dict:
    """
    Extract RTMW keypoints from a video file.

    Args:
        video_path: path to .mp4 file
        wholebody: initialized rtmlib Wholebody instance
        max_frames: max frames to sample
        save_full: if True, save all 133 keypoints; if False, save SLR subset (78)

    Returns:
        dict with "keypoints" (N, K, 2) and "scores" (N, K)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        n_kp = 133 if save_full else N_SLR_LANDMARKS
        return {
            "keypoints": np.zeros((1, n_kp, 2), dtype=np.float32),
            "scores": np.zeros((1, n_kp), dtype=np.float32),
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    # Uniform temporal sampling
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)

    all_keypoints = []
    all_scores = []

    frame_idx = 0
    sample_idx = 0

    while cap.isOpened() and sample_idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == frame_indices[sample_idx]:
            # rtmlib returns (keypoints, scores) for detected persons
            # keypoints: (N_persons, 133, 2), scores: (N_persons, 133)
            kps, scores = wholebody(frame)

            if kps is not None and len(kps) > 0:
                # Take the person with highest mean confidence
                if len(kps) > 1:
                    best_idx = scores.mean(axis=1).argmax()
                    kp = kps[best_idx]       # (133, 2)
                    sc = scores[best_idx]    # (133,)
                else:
                    kp = kps[0]
                    sc = scores[0]

                # Normalize to [0, 1] by frame dimensions
                h, w = frame.shape[:2]
                kp[:, 0] /= w
                kp[:, 1] /= h
            else:
                kp = np.zeros((133, 2), dtype=np.float32)
                sc = np.zeros(133, dtype=np.float32)

            if not save_full:
                kp = kp[SLR_IDXS]    # (78, 2)
                sc = sc[SLR_IDXS]    # (78,)

            all_keypoints.append(kp)
            all_scores.append(sc)
            sample_idx += 1

        frame_idx += 1

    cap.release()

    if len(all_keypoints) == 0:
        n_kp = 133 if save_full else N_SLR_LANDMARKS
        return {
            "keypoints": np.zeros((1, n_kp, 2), dtype=np.float32),
            "scores": np.zeros((1, n_kp), dtype=np.float32),
        }

    return {
        "keypoints": np.stack(all_keypoints, axis=0).astype(np.float32),
        "scores": np.stack(all_scores, axis=0).astype(np.float32),
    }


def batch_extract_keypoints(
    video_dir: str,
    output_dir: str,
    max_frames: int = 256,
    device: str = "cuda",
    save_full: bool = False,
):
    """
    Extract keypoints from all videos in a directory tree.
    Saves each as {attachment_id}.npz in output_dir.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} videos in {video_dir}")

    # Init model once
    print(f"Loading RTMW wholebody model on {device}...")
    wholebody = _init_wholebody(device=device)
    print("Model loaded.")

    n_kp = 133 if save_full else N_SLR_LANDMARKS
    print(f"Extracting {n_kp} keypoints per frame, max {max_frames} frames per video.")

    for vf in tqdm(video_files, desc="Extracting keypoints"):
        out_path = output_dir / f"{vf.stem}.npz"
        if out_path.exists():
            continue

        result = extract_keypoints_from_video(
            str(vf), wholebody, max_frames=max_frames, save_full=save_full
        )
        np.savez_compressed(
            str(out_path),
            keypoints=result["keypoints"],
            scores=result["scores"],
        )

    print(f"Done. Saved keypoints to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract RTMW keypoints from SLOVO videos")
    parser.add_argument("--video_dir", required=True, help="Path to SLOVO video directory")
    parser.add_argument("--output_dir", required=True, help="Where to save .npz keypoints")
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_full", action="store_true", help="Save all 133 keypoints (default: SLR subset of 78)")
    args = parser.parse_args()

    batch_extract_keypoints(
        args.video_dir, args.output_dir,
        max_frames=args.max_frames,
        device=args.device,
        save_full=args.save_full,
    )
