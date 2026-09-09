"""Model registry and download management for Open Vision.

Centralizes model metadata (URLs, sizes, cache paths) and provides
download/install functionality for STT backends.
"""
import hashlib
import logging
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from storage.paths import get_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Metadata for a downloadable model."""
    name: str
    provider: str  # "parakeet" or "whisper"
    description: str
    url: str
    size_bytes: int
    compressed_size_bytes: int = 0  # 0 = same as size_bytes (no compression)
    files: list[str] = None  # expected files after extraction

    def __post_init__(self):
        if self.files is None:
            self.files = []


# Parakeet TDT 0.6B v3 INT8 -- the default STT model
PARAKEET_MODEL = ModelEntry(
    name="parakeet-tdt-0.6b-v3-int8",
    provider="parakeet",
    description="NVIDIA Parakeet TDT 0.6B v3 (INT8, 25 languages, ~640MB)",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
    size_bytes=670_000_000,
    compressed_size_bytes=350_000_000,
    files=["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
)

# Whisper models via faster-whisper (Hugging Face Hub)
WHISPER_MODELS = {
    "tiny": ModelEntry(
        name="tiny",
        provider="whisper",
        description="Whisper tiny (39M params, ~75MB)",
        url="https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin",
        size_bytes=75_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "base": ModelEntry(
        name="base",
        provider="whisper",
        description="Whisper base (74M params, ~140MB)",
        url="https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
        size_bytes=140_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "small": ModelEntry(
        name="small",
        provider="whisper",
        description="Whisper small (244M params, ~460MB)",
        url="https://huggingface.co/Systran/faster-whisper-small/resolve/main/model.bin",
        size_bytes=460_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "medium": ModelEntry(
        name="medium",
        provider="whisper",
        description="Whisper medium (769M params, ~1.5GB)",
        url="https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin",
        size_bytes=1_500_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
    "large-v3": ModelEntry(
        name="large-v3",
        provider="whisper",
        description="Whisper large-v3 (1550M params, ~3.1GB)",
        url="https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin",
        size_bytes=3_100_000_000,
        files=["model.bin", "config.json", "vocabulary.json"],
    ),
}

# All models indexed by (provider, name)
ALL_MODELS: dict[tuple[str, str], ModelEntry] = {
    ("parakeet", "parakeet-tdt-0.6b-v3-int8"): PARAKEET_MODEL,
}
ALL_MODELS.update({("whisper", k): v for k, v in WHISPER_MODELS.items()})


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def models_dir() -> Path:
    """Central model cache directory: ~/.openvision/models/"""
    d = get_home() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parakeet_model_dir() -> Path:
    """Where the parakeet model files live."""
    return models_dir() / "parakeet-tdt-0.6b-v3-int8"


def whisper_model_dir(model_size: str) -> Path:
    """Where the whisper model files live."""
    return models_dir() / f"whisper-{model_size}"


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

@dataclass
class ModelStatus:
    """Status of a model in the cache."""
    name: str
    provider: str
    description: str
    installed: bool
    size_bytes: int
    path: Optional[Path] = None


def check_model_status(provider: str, name: str) -> ModelStatus:
    """Check if a model is installed and its size on disk."""
    entry = ALL_MODELS.get((provider, name))
    if not entry:
        return ModelStatus(
            name=name, provider=provider, description="Unknown model",
            installed=False, size_bytes=0,
        )

    if provider == "parakeet":
        model_path = parakeet_model_dir()
        expected_files = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
    elif provider == "whisper":
        model_path = whisper_model_dir(name)
        expected_files = ["model.bin", "config.json", "vocabulary.json"]
    else:
        return ModelStatus(
            name=name, provider=provider, description=entry.description,
            installed=False, size_bytes=0,
        )

    installed = model_path.exists() and all((model_path / f).exists() for f in expected_files)

    if installed and model_path.is_dir():
        size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    else:
        size = entry.size_bytes  # estimated download size

    return ModelStatus(
        name=name,
        provider=provider,
        description=entry.description,
        installed=installed,
        size_bytes=size,
        path=model_path if installed else None,
    )


def list_all_models() -> list[ModelStatus]:
    """Return status of all known models."""
    statuses = []
    # Parakeet (default)
    statuses.append(check_model_status("parakeet", "parakeet-tdt-0.6b-v3-int8"))
    # Whisper variants
    for name in WHISPER_MODELS:
        statuses.append(check_model_status("whisper", name))
    return statuses


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int, callback=None):
    """urllib download progress hook."""
    downloaded = block_num * block_size
    if total_size > 0 and callback:
        pct = min(100.0, downloaded * 100.0 / total_size)
        callback(pct, downloaded, total_size)
    elif total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / 1_000_000
        mb_total = total_size / 1_000_000
        print(f"\r  Downloading: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)


def download_file(url: str, dest: Path, callback=None) -> Path:
    """Download a file with progress reporting.

    Args:
        url: URL to download
        dest: Destination file path
        callback: Optional (pct, downloaded, total) callback

    Returns:
        Path to downloaded file
    """
    logger.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def hook(block_num, block_size, total_size):
        _progress_hook(block_num, block_size, total_size, callback)

    urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    print()  # newline after progress
    return dest


def download_and_extract_parakeet(callback=None) -> Path:
    """Download and extract the Parakeet model.

    Downloads the official sherpa-onnx tarball and extracts the
    INT8 quantized model files to ~/.openvision/models/parakeet-tdt-0.6b-v3-int8/

    Returns:
        Path to the extracted model directory
    """
    target_dir = parakeet_model_dir()

    # Already installed
    if all((target_dir / f).exists() for f in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]):
        logger.info(f"Parakeet model already installed at {target_dir}")
        return target_dir

    # Download tarball
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
    tarball_path = models_dir() / "parakeet-download.tar.bz2"

    print(f"Downloading Parakeet TDT 0.6B v3 (~350MB compressed)...")
    download_file(url, tarball_path, callback=callback)

    # Extract
    print("Extracting model files...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(tarball_path), "r:bz2") as tar:
        # Find the model files in the tarball
        members = tar.getmembers()
        for member in members:
            # The tarball has a nested directory structure; we want specific files
            basename = member.name.split("/")[-1]
            if basename in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]:
                # Extract to our target dir with just the filename
                member.name = basename
                tar.extract(member, str(target_dir))

    # Verify
    for f in ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]:
        if not (target_dir / f).exists():
            raise RuntimeError(f"Extraction failed: {f} not found in {target_dir}")

    # Clean up tarball
    tarball_path.unlink(missing_ok=True)
    print(f"Parakeet model installed to {target_dir}")
    return target_dir


def download_whisper_model(model_size: str, callback=None) -> Path:
    """Download a Whisper model via faster-whisper.

    Uses faster-whisper's built-in Hugging Face Hub download.

    Returns:
        Path to the downloaded model directory
    """
    if model_size not in WHISPER_MODELS:
        raise ValueError(f"Unknown whisper model: {model_size}. Choose from: {', '.join(WHISPER_MODELS.keys())}")

    target_dir = whisper_model_dir(model_size)

    # Already installed
    if all((target_dir / f).exists() for f in ["model.bin", "config.json", "vocabulary.json"]):
        logger.info(f"Whisper {model_size} already installed at {target_dir}")
        return target_dir

    # Use faster-whisper's download mechanism
    try:
        from faster_whisper import download_model
        print(f"Downloading Whisper {model_size} via faster-whisper...")
        download_model(model_size, str(target_dir))
        print(f"Whisper {model_size} installed to {target_dir}")
        return target_dir
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )


def install_model(provider: str, name: str, callback=None) -> Path:
    """Install a model by provider and name.

    Args:
        provider: "parakeet" or "whisper"
        name: Model name (e.g., "parakeet-tdt-0.6b-v3-int8" or "base")
        callback: Optional progress callback

    Returns:
        Path to installed model directory
    """
    if provider == "parakeet":
        return download_and_extract_parakeet(callback=callback)
    elif provider == "whisper":
        return download_whisper_model(name, callback=callback)
    else:
        raise ValueError(f"Unknown provider: {provider}")
