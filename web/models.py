from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class FlowVector(BaseModel):
    start: List[float] = Field(..., min_length=2, max_length=2)
    end: List[float] = Field(..., min_length=2, max_length=2)


class ZonePayload(BaseModel):
    zone_a_detection: List[List[float]]
    zone_b_wash: List[List[float]]
    flow_vector: FlowVector


class ConfigPayload(BaseModel):
    system: Optional[Dict[str, Any]] = None
    video: Optional[Dict[str, Any]] = None
    wheel: Optional[Dict[str, Any]] = None
    logic: Optional[Dict[str, Any]] = None
    zones: Optional[Dict[str, Any]] = None
    event_output_dir: Optional[str] = None
    event_capture_dir: Optional[str] = None
    event_capture_quality: Optional[int] = None


class ConfigSelectPayload(BaseModel):
    name: str


class ConfigSaveAsPayload(BaseModel):
    name: str
    data: Dict[str, Any]


class SnapshotKeepPayload(BaseModel):
    tag: Optional[str] = "manual"
    raw: bool = True
    annotated: bool = True
