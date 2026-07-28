"""Stable filesystem paths for Open Vision data (memory, downloads, runs).

Memory was previously written relative to the process CWD, so
``openvision memory list`` looked empty whenever the shell was in a
different directory from the observe run. Paths now resolve under
``OPENVISION_HOME`` (default ``~/.openvision``) unless overridden in config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_HOME_NAME = ".openvision"


def get_home(config: Optional[dict] = None) -> Path:
    """Return the Open Vision data home directory (created if missing).

    Priority:
    1. ``OPENVISION_HOME`` env var (or legacy ``OPHANIM_HOME`` for backward compat)
    2. ``paths.home`` in config
    3. ``~/.openvision``
    """
    env = os.environ.get("OPENVISION_HOME") or os.environ.get("OPHANIM_HOME")
    if env:
        home = Path(env).expanduser().resolve()
    else:
        cfg_home = None
        if config:
            cfg_home = (config.get("paths") or {}).get("home")
        if cfg_home:
            home = Path(cfg_home).expanduser().resolve()
        else:
            home = Path.home() / DEFAULT_HOME_NAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def _subpath(config: Optional[dict], key: str, default_name: str) -> Path:
    """Resolve a named data subdirectory under home (or absolute override)."""
    override = None
    if config:
        override = (config.get("paths") or {}).get(key)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = get_home(config) / path
    else:
        path = get_home(config) / default_name
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def memory_dir(config: Optional[dict] = None) -> Path:
    """Directory for ``--save-memory`` markdown notes (``memory/videos``)."""
    base = _subpath(config, "memory", "memory")
    videos = base / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    return videos


def downloads_dir(config: Optional[dict] = None) -> Path:
    """Directory for URL downloads via native yt-dlp."""
    return _subpath(config, "downloads", "downloads")


def runs_dir(config: Optional[dict] = None) -> Path:
    """Directory for observe run caches / frame artifacts."""
    return _subpath(config, "runs", "runs")


def migrate_from_legacy() -> Optional[Path]:
    """Migrate data from ~/.ophanim to ~/.openvision if it exists.

    Returns the old path if migration was performed, None otherwise.
    """
    old_home = Path.home() / ".ophanim"
    new_home = Path.home() / ".openvision"

    if old_home.exists() and not new_home.exists():
        import shutil
        shutil.move(str(old_home), str(new_home))
        return old_home
    return None
