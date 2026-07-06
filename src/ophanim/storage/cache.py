"""Run cache and artifact management for Ophanim."""
import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


class RunCache:
    """
    Manages run directories for video processing artifacts.

    Directory structure:
    runs/YYYY-MM-DD_HHMMSS_<hash>/
        input_metadata.json
        config.json
        sampled_frames/
        thumbnails/
        masks/
        observations.json
        timeline.md
        summary.md
        logs.txt
    """

    def __init__(self, base_dir: str = "./runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, path: str, mode: str = "balanced", fps: float = 0.5,
                  resolution: int = 768) -> str:
        """
        Generate a cache key from video file properties + processing params.

        Uses: path + file_size + mtime + mode + fps + resolution
        """
        video_path = Path(path)
        if not video_path.exists():
            return ""

        stat = video_path.stat()
        raw = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime}:{mode}:{fps}:{resolution}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_run(self, key: str) -> Optional[Path]:
        """Find existing run directory by cache key. Returns None if not found."""
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.endswith(key):
                return run_dir
        return None

    def has_cached(self, key: str) -> bool:
        """Check if a cached run exists for this key."""
        return self.get_run(key) is not None

    def create_run(self, key: str, metadata: Optional[dict] = None) -> Path:
        """
        Create a new run directory.

        Returns:
            Path to the new run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{key}"
        run_dir = self.base_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "frames").mkdir(exist_ok=True)
        (run_dir / "thumbnails").mkdir(exist_ok=True)
        (run_dir / "masks").mkdir(exist_ok=True)

        # Save metadata
        if metadata:
            self._save_json(run_dir / "input_metadata.json", metadata)

        return run_dir

    def save_artifact(self, run_dir: Path, filename: str, data: dict):
        """Save a JSON artifact to the run directory."""
        self._save_json(run_dir / filename, data)

    def save_text(self, run_dir: Path, filename: str, text: str):
        """Save a text file to the run directory."""
        (run_dir / filename).write_text(text, encoding="utf-8")

    def save_frame(self, run_dir: Path, image, filename: str) -> Path:
        """Save an image frame to the run directory. Returns the path."""
        import cv2
        frame_path = run_dir / "frames" / filename
        cv2.imwrite(str(frame_path), image)
        return frame_path

    def list_runs(self) -> list[dict]:
        """List all cached runs with basic metadata."""
        runs = []
        for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            meta = {}
            meta_path = run_dir / "input_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())

            runs.append({
                "name": run_dir.name,
                "path": str(run_dir),
                "created": run_dir.stat().st_mtime,
                "metadata": meta,
            })
        return runs

    def _save_json(self, path: Path, data: dict):
        """Save dict as pretty JSON."""
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def create_cache_key(path: str, mode: str = "balanced", fps: float = 0.5,
                     resolution: int = 768) -> str:
    """Standalone helper to generate a cache key."""
    cache = RunCache()
    return cache.cache_key(path, mode, fps, resolution)
