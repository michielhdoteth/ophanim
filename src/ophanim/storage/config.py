"""Configuration loader for Ophanim."""
import os
import yaml
from pathlib import Path
from typing import Optional


def find_config() -> Path:
    """
    Find the config file to use.

    Priority:
    1. OPHANIM_CONFIG env var
    2. config/default.yaml relative to package
    3. config/default.yaml relative to cwd
    """
    env_config = os.environ.get("OPHANIM_CONFIG")
    if env_config:
        env_path = Path(env_config)
        if env_path.exists():
            return env_path

    # Package-relative
    pkg_config = Path(__file__).parent.parent / "config" / "default.yaml"
    if pkg_config.exists():
        return pkg_config

    # CWD-relative
    cwd_config = Path.cwd() / "config" / "default.yaml"
    if cwd_config.exists():
        return cwd_config

    raise FileNotFoundError(
        "Cannot find config file. Set OPHANIM_CONFIG env var "
        "or ensure config/default.yaml exists."
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Optional explicit path. If None, auto-discover.

    Returns:
        Dict with merged configuration values.
    """
    if config_path:
        path = Path(config_path)
    else:
        path = find_config()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def get_mode_config(config: dict, mode: str) -> dict:
    """Get configuration for a specific mode (fast/balanced/detailed)."""
    modes = config.get("modes", {})
    if mode in modes:
        return modes[mode]

    # Fall back to defaults
    defaults = config.get("defaults", {})
    return {
        "resolution": defaults.get("max_resolution", 768),
        "fps": defaults.get("fps", 0.5),
        "max_frames": defaults.get("max_frames", 60),
    }


def merge_config(base: dict, override: dict) -> dict:
    """Deep merge two config dicts. Override values win."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result
