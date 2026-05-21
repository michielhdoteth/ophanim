"""Smart frame sampling: scene detection, deduplication, keyframe selection."""
import cv2
import numpy as np
from typing import Optional
from visionclaw.core.video import extract_frames, probe


def detect_scenes(
    frames: list[dict],
    threshold: float = 30.0,
    min_scene_gap: int = 2,
) -> list[int]:
    """
    Detect scene changes using HSV histogram comparison.

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        threshold: Histogram difference threshold (lower = more sensitive)
        min_scene_gap: Minimum frames between scene changes

    Returns:
        List of frame indices where scene changes occur
    """
    if len(frames) < 2:
        return []

    scene_indices = []
    prev_hist = None

    for i, frame in enumerate(frames):
        if frame["image"] is None:
            continue

        # Convert to HSV for better illumination invariance
        hsv = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2HSV)
        # Use all 3 HSV channels so uniform black/white frames are distinguishable
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 30], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        if prev_hist is not None:
            # Chi-squared distance
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

            if diff > threshold:
                # Check gap from last scene change
                if not scene_indices or (i - scene_indices[-1]) >= min_scene_gap:
                    scene_indices.append(i)

        prev_hist = hist

    return scene_indices


def deduplicate(
    frames: list[dict],
    indices: Optional[list[int]] = None,
    similarity_threshold: float = 0.95,
) -> list[int]:
    """
    Remove near-identical frames using histogram correlation.

    Args:
        frames: List of frame dicts with 'image' key
        indices: Subset of frame indices to filter (None = all)
        similarity_threshold: 0-1, higher = more aggressive dedup

    Returns:
        Filtered list of frame indices
    """
    if indices is None:
        indices = list(range(len(frames)))

    if len(indices) <= 1:
        return indices

    result = [indices[0]]
    ref_hist = None

    for idx in indices[1:]:
        frame = frames[idx]
        if frame["image"] is None:
            continue

        hsv = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2HSV)
        # Use all 3 HSV channels so uniform black/white frames are distinguishable
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 30], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        if ref_hist is not None:
            # Correlation: 1 = identical, 0 = unrelated
            similarity = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)

            if similarity < similarity_threshold:
                result.append(idx)
                ref_hist = hist
        else:
            ref_hist = hist

    return result


def smart_sample(
    path: str,
    fps: float = 0.5,
    max_frames: int = 60,
    max_resolution: int = 768,
    scene_threshold: float = 30.0,
    dedup_threshold: float = 0.95,
) -> list[dict]:
    """
    Full smart sampling pipeline:
    1. Extract frames at given FPS (up to max_frames)
    2. Detect scene changes
    3. Deduplicate near-identical frames
    4. Ensure scene change frames are included

    Returns:
        List of selected frame dicts with {index, timestamp, timestamp_str, image}
    """
    # Step 1: Extract frames
    frames = extract_frames(path, fps=fps, max_frames=max_frames, max_resolution=max_resolution)

    if not frames:
        return []

    # Step 2: Detect scene changes
    scene_indices = detect_scenes(frames, threshold=scene_threshold)

    # Step 3: Start with uniform sample, then add scene changes
    uniform_indices = list(range(0, len(frames), max(1, len(frames) // min(max_frames, len(frames)))))
    uniform_indices = uniform_indices[:max_frames]

    # Merge uniform + scene indices, keep order
    all_indices = sorted(set(uniform_indices + scene_indices))

    # Step 4: Deduplicate (but keep scene change frames)
    # We dedup the uniform samples first, then merge scene changes
    deduped_uniform = deduplicate(frames, list(range(len(frames))), similarity_threshold=dedup_threshold)

    # Combine deduped uniform + scene changes, keep unique and in order
    final_indices = sorted(set(deduped_uniform + scene_indices))

    return [frames[i] for i in final_indices if i < len(frames)]


def find_keyframes_for_question(
    frames: list[dict],
    question_relevant_indices: Optional[list[int]] = None,
    max_keyframes: int = 10,
) -> list[dict]:
    """
    Select keyframes most relevant to a specific question.
    If no relevance hints, return evenly spaced keyframes.
    """
    if not frames:
        return []

    if question_relevant_indices:
        # Use specific indices if provided
        selected = [frames[i] for i in question_relevant_indices if i < len(frames)]
        return selected[:max_keyframes]

    # Evenly spaced
    step = max(1, len(frames) // min(max_keyframes, len(frames)))
    return [frames[i] for i in range(0, len(frames), step)][:max_keyframes]
