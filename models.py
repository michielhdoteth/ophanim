"""Pydantic schemas for Ophanim tools."""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class Frame(BaseModel):
    """A single video frame reference."""
    index: int
    timestamp: float
    timestamp_str: str
    path: Optional[str] = None


class TimelineEntry(BaseModel):
    """An observation at a specific point in time."""
    time_seconds: float
    timestamp: str
    observation: str
    frame_path: Optional[str] = None


class ObserveResult(BaseModel):
    """Result of a video_observe operation."""
    summary: str
    timeline: list[TimelineEntry]
    entities: list[str]
    artifacts_dir: str
    confidence: Literal["low", "medium", "high"]


class AskResult(BaseModel):
    """Result of a video_ask operation."""
    answer: str
    evidence: list[dict] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class ProbeResult(BaseModel):
    """Metadata from video probing."""
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    frame_count: int
    estimated_processing_cost: Literal["low", "medium", "high"]


class SegmentObject(BaseModel):
    """A segmented object within a frame."""
    object_id: str
    timestamps: list[str]
    mask_paths: list[str]
    bbox: list[float]


class SegmentResult(BaseModel):
    """Result of a video_segment operation."""
    prompt: str
    frames_processed: int
    masks_dir: str
    objects: list[SegmentObject]


class TrackPosition(BaseModel):
    """Object position at a point in time."""
    timestamp: str
    time_seconds: float
    bbox: list[float]
    confidence: float


class TrackResult(BaseModel):
    """Result of a video_track operation."""
    track_id: str
    positions: list[TrackPosition]
    summary: str


class ImageResult(BaseModel):
    """Result of a single image observation."""
    description: str
    objects: list[str] = Field(default_factory=list)
    text_detected: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class StatusResult(BaseModel):
    """System status report."""
    gpu: str
    vram_total_gb: float
    vram_free_gb: float
    loaded_models: list[str]
    safe_mode: bool
    queue: int = 0


class OphanimConfig(BaseModel):
    """Full configuration model."""
    machine: dict = Field(default_factory=dict)
    defaults: dict = Field(default_factory=lambda: {
        "mode": "balanced",
        "max_resolution": 768,
        "fps": 0.5,
        "max_frames": 60,
    })
    models: dict = Field(default_factory=dict)
    gpu_policy: dict = Field(default_factory=dict)
    cache: dict = Field(default_factory=lambda: {"enabled": True, "directory": "./runs"})
    modes: dict = Field(default_factory=dict)
