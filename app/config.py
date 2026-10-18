from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ""


ROOT_DIR = _project_root()
APP_DIR = ROOT_DIR / "app"
STATIC_DIR = APP_DIR / "static"

DEFAULT_MODEL_PATH = ROOT_DIR / "weights" / "best.pt"
_raw_model_path = os.getenv("MODEL_PATH")
if _raw_model_path:
    _model_path = Path(_raw_model_path).expanduser()
    MODEL_PATH = (_model_path if _model_path.is_absolute() else ROOT_DIR / _model_path).resolve()
else:
    MODEL_PATH = DEFAULT_MODEL_PATH.resolve()

CUDA_AVAILABLE = _cuda_available()
GPU_NAME = _gpu_name()
REQUESTED_DEVICE = os.getenv("DEVICE", "cuda:0" if CUDA_AVAILABLE else "cpu")
DEVICE = REQUESTED_DEVICE if REQUESTED_DEVICE.startswith("cuda") and CUDA_AVAILABLE else "cpu"

DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(APP_DIR / "uploads"))).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(APP_DIR / "outputs"))).expanduser().resolve()
ANNOTATION_DIR = Path(os.getenv("ANNOTATION_DIR", str(APP_DIR / "annotations"))).expanduser().resolve()
ANNOTATION_CLASSES_PATH = ANNOTATION_DIR / "classes.json"
ANNOTATION_DATASETS_DIR = ANNOTATION_DIR / "datasets"
MODEL_UPLOAD_DIR = Path(os.getenv("MODEL_UPLOAD_DIR", str(APP_DIR / "model_uploads"))).expanduser().resolve()
MODEL_UPLOAD_MODELS_DIR = MODEL_UPLOAD_DIR / "models"
MODEL_UPLOAD_YAMLS_DIR = MODEL_UPLOAD_DIR / "yamls"
MODEL_REGISTRY_PATH = MODEL_UPLOAD_DIR / "registry.json"

DEFAULT_MODEL_DISPLAY_NAME = "安全帽安全服检测模型"
DEFAULT_MODEL_TYPE = "safety_ppe"
DEFAULT_MODEL_ID = "default_safety_model"
DEFAULT_MODEL_NAMES = {
    "0": "person",
    "1": "helmet",
    "2": "no_helmet",
    "3": "safety_vest",
    "4": "no_vest",
}

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
ALLOWED_MODEL_EXT = {"pt", "pth"}
ALLOWED_YAML_EXT = {"yaml", "yml"}
ALLOWED_IMAGE_MIME_PREFIXES = {"image/"}
ALLOWED_VIDEO_MIME_PREFIXES = {"video/", "application/octet-stream"}

STREAM_MAX_FPS = int(os.getenv("STREAM_MAX_FPS", "10"))
MAX_STREAMS = int(os.getenv("MAX_STREAMS", "2"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))

VIOLATION_MODE = os.getenv("VIOLATION_MODE", "auto")
VEST_OVERLAP_THRESHOLD = float(os.getenv("VEST_OVERLAP_THRESHOLD", "0.10"))
VEST_CONF_THRESHOLD = float(os.getenv("VEST_CONF_THRESHOLD", str(DEFAULT_CONF)))

CLASS_ALIASES: Dict[str, List[str]] = {
    "person": ["person", "worker", "people"],
    "vest": ["vest", "safety_vest", "safety vest", "reflective_vest"],
    "no_vest": ["no_vest", "no vest", "no_safety_vest", "without_vest"],
    "helmet": ["helmet", "hardhat", "safety_helmet"],
    "no_helmet": ["no_helmet", "no hardhat", "without_helmet"],
}


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_UPLOAD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_UPLOAD_YAMLS_DIR.mkdir(parents=True, exist_ok=True)
