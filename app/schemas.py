from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StreamStartRequest(BaseModel):
    source: str = Field(default="0", description="Camera index, RTSP URL, or HTTP stream URL")
    conf: float = Field(default=0.25, ge=0.01, le=1.0)
    iou: float = Field(default=0.45, ge=0.01, le=1.0)
    imgsz: int = Field(default=640, ge=160, le=2048)


class StreamStartResponse(BaseModel):
    stream_id: str
    websocket_url: str


class DetectionItem(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    is_violation: bool = False
    violation_type: Optional[str] = None


class DetectionStats(BaseModel):
    person_count: int = 0
    vest_count: int = 0
    no_vest_count: int = 0
    no_helmet_count: int = 0
    violation_confidences: List[float] = Field(default_factory=list)
    avg_violation_confidence: float = 0.0
    max_violation_confidence: float = 0.0
    min_violation_confidence: float = 0.0
    fps: float = 0.0
    inference_ms: float = 0.0


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result_video_url: Optional[str] = None
    summary_stats: Optional[Dict[str, Any]] = None
    frame_count: int = 0
    duration: float = 0.0
    fps: float = 0.0
