"""Video probing and frame extraction using OpenCV."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def probe(path: str) -> dict:
    """
    Extract video metadata without full decode.

    Returns dict with: duration_seconds, width, height, fps, codec, frame_count
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4)) if codec_int else "unknown"

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "frame_count": frame_count,
    }


def extract_frames(
    path: str,
    fps: float = 0.5,
    max_frames: int = 60,
    max_resolution: int = 768,
) -> list[dict]:
    """
    Extract frames from video at given sampling rate.

    Args:
        path: Path to video file
        fps: Target frames per second to extract
        max_frames: Maximum number of frames to return
        max_resolution: Longest side in pixels for downscaling

    Returns:
        List of dicts: {index, timestamp, timestamp_str, image (np.ndarray)}
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 30.0

    # Calculate stride that evenly samples across the ENTIRE video duration
    # This prevents only sampling the first N frames of a long video
    fps_interval = int(video_fps / fps) if fps > 0 else 1
    fps_interval = max(1, fps_interval)
    total_at_fps = total_frames // fps_interval

    if total_at_fps > max_frames:
        # Spread max_frames evenly across the full video
        stride = max(1, total_frames // max_frames)
    else:
        stride = fps_interval

    frames = []
    sample_idx = 0

    for frame_idx in range(0, total_frames, stride):
        if len(frames) >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        if not ret:
            continue

        timestamp = frame_idx / video_fps
        timestamp_str = _format_timestamp(timestamp)

        # Downscale
        image = _downscale(image, max_resolution)

        frames.append({
            "index": sample_idx,
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
            "image": image,
        })
        sample_idx += 1

    cap.release()
    return frames


def extract_frames_by_indices(
    path: str,
    indices: list[int],
    max_resolution: int = 768,
) -> list[dict]:
    """Extract specific frame indices from video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for i, idx in enumerate(sorted(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, image = cap.read()
        if not ret:
            continue

        timestamp = idx / video_fps if video_fps > 0 else 0
        timestamp_str = _format_timestamp(timestamp)
        image = _downscale(image, max_resolution)

        frames.append({
            "index": i,
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
            "image": image,
        })

    cap.release()
    return frames


def _downscale(image: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Resize image so longest side is max_side, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def estimate_processing_cost(metadata: dict, mode: str = "balanced") -> str:
    """Estimate processing cost based on video metadata and mode."""
    duration = metadata.get("duration_seconds", 0)
    resolution = metadata.get("width", 0) * metadata.get("height", 0)

    if mode == "fast":
        limit = 120  # 2 min
    elif mode == "detailed":
        limit = 30   # 30 sec
    else:
        limit = 60   # 1 min

    if duration <= limit and resolution <= 1280 * 720:
        return "low"
    elif duration <= limit * 3:
        return "medium"
    else:
        return "high"
