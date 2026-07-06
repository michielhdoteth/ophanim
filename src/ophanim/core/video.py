"""Video probing and frame extraction using OpenCV and ffmpeg."""
import cv2
import numpy as np
import subprocess
import tempfile
import os
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
            "reason": "uniform",
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


def extract_at_timestamps(
    path: str,
    timestamps: list[float],
    max_resolution: int = 768,
) -> list[dict]:
    """Extract a single frame at each specified timestamp. These are 'pinned' frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for i, ts in enumerate(sorted(timestamps)):
        frame_idx = int(ts * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        if not ret:
            continue

        image = _downscale(image, max_resolution)
        frames.append({
            "index": i,
            "timestamp": ts,
            "timestamp_str": _format_timestamp(ts),
            "image": image,
            "reason": "timestamp-cue",
            "pinned": True,
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


def parse_time(s: str) -> float:
    """Parse a timestamp string into float seconds.

    Accepts: 'SS', 'MM:SS', 'HH:MM:SS', 'HH:MM:SS.mmm'
    Examples: '45' -> 45.0, '1:30' -> 90.0, '1:30:00' -> 5400.0
    """
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp: {s}")


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


# ---------------------------------------------------------------------------
# ffmpeg-based extraction (for large/long videos where OpenCV is too slow)
# ---------------------------------------------------------------------------

def extract_scene_frames_ffmpeg(
    path: str,
    threshold: float = 0.20,
    max_frames: int = 100,
    max_resolution: int = 768,
    min_scene_gap: float = 0.5,
) -> list[dict]:
    """
    Extract frames at scene changes using ffmpeg's scene detection filter.

    Args:
        path: Video file path
        threshold: Scene change sensitivity 0-1 (lower = more scenes detected)
        max_frames: Maximum frames to return
        max_resolution: Longest side in pixels
        min_scene_gap: Minimum seconds between scene frames

    Returns:
        List of frame dicts: {index, timestamp, timestamp_str, image}
    """
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use ffmpeg scene detection with select filter
        cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr",
            "-frame_pts", "1",
            os.path.join(tmpdir, "frame_%06d.jpg"),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg scene detection timed out (10 min)")

        # Parse showinfo output to get timestamps
        timestamps = []
        for line in result.stderr.split("\n"):
            if "showinfo" in line and "pts_time:" in line:
                try:
                    pts_part = line.split("pts_time:")[1].split()[0]
                    timestamps.append(float(pts_part))
                except (IndexError, ValueError):
                    continue

        # Read extracted frames
        frames = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        for i, (fpath, ts) in enumerate(zip(frame_files, timestamps)):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, max_resolution)
            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _format_timestamp(ts),
                "image": img,
                "reason": "scene-change",
            })

    return frames


def extract_keyframes_ffmpeg(
    path: str,
    max_frames: int = 100,
    max_resolution: int = 768,
) -> list[dict]:
    """
    Extract I-frames (keyframes) using ffmpeg's skip_frame filter.

    Keyframes are the most informative frames in a video - they contain
    full image data without prediction from other frames.
    """
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    # Get video fps for timestamp calculation
    meta = probe(path)
    video_fps = meta.get("fps", 30.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", "select='eq(pict_type,I)'",
            "-vsync", "vfr",
            os.path.join(tmpdir, "keyframe_%06d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg keyframe extraction timed out")

        frames = []
        for i, fpath in enumerate(sorted(Path(tmpdir).glob("keyframe_*.jpg"))):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, max_resolution)
            # Estimate timestamp from frame index (approximate)
            ts = i * (meta["duration_seconds"] / max(1, len(list(Path(tmpdir).glob("keyframe_*.jpg")))))
            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _format_timestamp(ts),
                "image": img,
                "reason": "keyframe",
            })

    return frames


def dedupe_frames_ffmpeg(
    frames: list[dict],
    threshold: float = 2.0,
) -> list[dict]:
    """
    Lightweight perceptual deduplication using ffmpeg 16x16 thumbnails.

    This is much faster than full histogram comparison. It:
    1. Scales each frame to 16x16 grayscale via ffmpeg
    2. Computes mean absolute pixel difference between consecutive frames
    3. Keeps frames where difference > threshold

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        threshold: Minimum pixel difference to keep (default 2.0)

    Returns:
        Deduplicated list of frames
    """
    if len(frames) <= 1:
        return frames

    result = [frames[0]]
    prev_thumb = _make_thumbnail(frames[0]["image"])

    for frame in frames[1:]:
        thumb = _make_thumbnail(frame["image"])
        diff = np.mean(np.abs(prev_thumb.astype(float) - thumb.astype(float)))
        if diff > threshold:
            result.append(frame)
            prev_thumb = thumb

    return result


def dedupe_frames_sliding(
    frames: list[dict],
    threshold: float = 2.0,
    window: int = 3,
) -> list[dict]:
    """
    Sliding-window perceptual deduplication.

    Like dedupe_frames_ffmpeg() but compares against the last N kept frames
    instead of just the last one. This catches A-B-A cutaways where frame A
    appears, then B, then A again.

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        threshold: Minimum pixel difference to keep (default 2.0)
        window: Number of recent kept frames to compare against (default 3)

    Returns:
        Deduplicated list of frames
    """
    if len(frames) <= 1:
        return frames

    result = [frames[0]]
    thumbnails = [_make_thumbnail(frames[0]["image"])]

    for frame in frames[1:]:
        thumb = _make_thumbnail(frame["image"])

        # Compare against last N kept frames
        is_duplicate = False
        for prev_thumb in thumbnails[-window:]:
            diff = np.mean(np.abs(prev_thumb.astype(float) - thumb.astype(float)))
            if diff <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(frame)
            thumbnails.append(thumb)

    return result


def _make_thumbnail(image: np.ndarray, size: int = 16) -> np.ndarray:
    """Create a small grayscale thumbnail for fast comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def auto_fps(duration_seconds: float) -> int:
    """
    Calculate frame budget based on video duration.

    Returns maximum number of frames to extract:
    - <=30s: 12 frames
    - <=60s: 40 frames
    - <=3min: 60 frames
    - <=10min: 80 frames
    - >10min: 100 frames
    """
    if duration_seconds <= 30:
        return 12
    elif duration_seconds <= 60:
        return 40
    elif duration_seconds <= 180:
        return 60
    elif duration_seconds <= 600:
        return 80
    else:
        return 100


def auto_fps_focus(duration_seconds: float) -> int:
    """
    Calculate dense frame budget for a focus range window.

    When user specifies --start/--end, we want much denser extraction
    since they're zooming into a specific section.

    Returns maximum number of frames:
    - <=5s: 30 frames (6 fps)
    - <=15s: 60 frames (4 fps)
    - <=30s: 60 frames (2 fps)
    - <=60s: 60 frames (1 fps)
    - >60s: 80 frames
    """
    if duration_seconds <= 5:
        return 30
    elif duration_seconds <= 15:
        return 60
    elif duration_seconds <= 30:
        return 60
    elif duration_seconds <= 60:
        return 60
    else:
        return 80
