"""Video downloading from URLs using yt-dlp."""
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Check if a string is a URL rather than a local file path."""
    return bool(re.match(r'https?://', path))


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    cmd = ["yt-dlp", "--dump-json", "--no-download", "--no-warnings", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")
        import json
        return json.loads(result.stdout)
    except FileNotFoundError:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")


def download_video(
    url: str,
    output_dir: Optional[str] = None,
    max_height: int = 720,
    audio_only: bool = False,
    write_subs: bool = True,
    cookies_file: Optional[str] = None,
) -> dict:
    """Download video from URL. Returns dict with path, title, duration, info."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ophanim_dl_")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if audio_only:
        cmd = [
            "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "64K",
            "-o", str(output_path / "%(title)s.%(ext)s"),
        ]
    else:
        cmd = [
            "yt-dlp",
            f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
            "--merge-output-format", "mp4",
            "-o", str(output_path / "%(title)s.%(ext)s"),
        ]

    if write_subs:
        cmd.extend(["--write-auto-sub", "--sub-lang", "en", "--convert-subs", "srt"])
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    cmd.append(url)

    logger.info(f"Downloading: {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp download failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timed out (10 min)")

    downloaded_files = list(output_path.glob("*"))
    if not downloaded_files:
        raise RuntimeError("Download completed but no files found")

    video_files = [f for f in downloaded_files if f.suffix in {'.mp4', '.mkv', '.webm', '.mov'}]
    audio_files = [f for f in downloaded_files if f.suffix in {'.mp3', '.wav', '.m4a', '.opus'}]
    main_file = video_files[0] if video_files else (audio_files[0] if audio_files else downloaded_files[0])

    # Find subtitle files
    sub_files = list(output_path.glob("*.srt")) + list(output_path.glob("*.vtt"))
    subs_file = str(sub_files[0]) if sub_files else None

    try:
        info = get_video_info(url)
    except Exception:
        info = {}

    return {
        "path": str(main_file),
        "title": info.get("title", main_file.stem),
        "duration": info.get("duration", 0),
        "info": info,
        "subs_file": subs_file,
    }


def fetch_captions(url: str, lang: str = "en") -> Optional[str]:
    """Fetch captions from URL without downloading the video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pattern = os.path.join(tmpdir, "subs")
        cmd = [
            "yt-dlp", "--skip-download", "--no-warnings",
            "--write-auto-sub", "--sub-lang", lang,
            "--convert-subs", "vtt",
            "-o", out_pattern,
            url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if vtt_files:
            return vtt_files[0].read_text(encoding="utf-8", errors="replace")
        return None
