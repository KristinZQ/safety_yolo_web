from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from app import config
from app.model_registry.label_mapping import default_label_mapping, normalize_mapping
from app.model_registry.yaml_utils import parse_yaml_names
from app.services.detector import YOLODetector

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024


class ModelManager:
    def __init__(self, device: str) -> None:
        self.device = device
        self.detector: Optional[YOLODetector] = None
        config.ensure_runtime_dirs()
        self._ensure_registry()
        self.load_active_model(raise_on_error=False)

    def get_registry(self) -> dict:
        self._ensure_registry()
        return self._read_registry()

    def get_active_model(self) -> Optional[dict]:
        registry = self.get_registry()
        return self._find_model(registry.get("active_model_id"), registry, required=False)

    def get_active_detector(self) -> YOLODetector:
        if self.detector is None:
            raise HTTPException(status_code=503, detail="No model is loaded. Please upload or activate a valid YOLO .pt model.")
        return self.detector

    def list_models(self) -> dict:
        registry = self.get_registry()
        return {
            "active_model_id": registry.get("active_model_id"),
            "active_model": self._public_model(self._find_model(registry.get("active_model_id"), registry, required=False)),
            "models": [self._public_model(model) for model in registry.get("models", [])],
        }

    async def upload_model(
        self,
        weight_file: UploadFile | None,
        yaml_file: UploadFile | None,
        display_name: str | None,
        model_type: str | None,
    ) -> dict:
        if weight_file is None and yaml_file is None:
            raise HTTPException(status_code=400, detail="Please upload a model weight or YAML file")

        model_id = uuid.uuid4().hex[:12]
        weight_path = None
        yaml_path = None
        original_weight_filename = None
        stored_weight_filename = None
        original_yaml_filename = None
        stored_yaml_filename = None

        if weight_file is not None:
            ext = _extension(weight_file.filename)
            if ext not in config.ALLOWED_MODEL_EXT:
                raise HTTPException(status_code=400, detail=f"Unsupported model extension: .{ext or 'unknown'}")
            original_weight_filename = Path(weight_file.filename or "model.pt").name
            stored_weight_filename = f"{model_id}_{_safe_filename(original_weight_filename)}"
            weight_path = config.MODEL_UPLOAD_MODELS_DIR / stored_weight_filename
            await _save_upload(weight_file, weight_path)

        if yaml_file is not None:
            ext = _extension(yaml_file.filename)
            if ext not in config.ALLOWED_YAML_EXT:
                raise HTTPException(status_code=400, detail=f"Unsupported YAML extension: .{ext or 'unknown'}")
            original_yaml_filename = Path(yaml_file.filename or "data.yaml").name
            stored_yaml_filename = f"{model_id}_{_safe_filename(original_yaml_filename)}"
            yaml_path = config.MODEL_UPLOAD_YAMLS_DIR / stored_yaml_filename
            await _save_upload(yaml_file, yaml_path)

        names = None
        if yaml_path is not None:
            names = parse_yaml_names(yaml_path)
        elif weight_path is not None:
            names = self._read_model_names(weight_path)

        is_detection_model = weight_path is not None
        record = {
            "model_id": model_id,
            "display_name": display_name or (Path(original_weight_filename).stem if original_weight_filename else "YAML 类别配置"),
            "model_type": model_type or ("custom" if is_detection_model else "annotation_yaml"),
            "weight_path": _relative_to_root(weight_path) if weight_path else None,
            "yaml_path": _relative_to_root(yaml_path) if yaml_path else None,
            "original_weight_filename": original_weight_filename,
            "stored_weight_filename": stored_weight_filename,
            "original_yaml_filename": original_yaml_filename,
            "stored_yaml_filename": stored_yaml_filename,
            "names": names or {},
            "label_mapping": default_label_mapping(names or {}),
            "created_at": _now(),
            "is_default": False,
            "is_detection_model": is_detection_model,
        }

        registry = self._read_registry()
        registry.setdefault("models", []).append(record)
        self._write_registry(registry)
        return {"model": self._public_model(record), "names": record["names"], "classes": _names_to_classes(record["names"])}

    def save_mapping(self, model_id: str, payload: dict) -> dict:
        registry = self._read_registry()
        model = self._find_model(model_id, registry)
        if payload.get("display_name"):
            model["display_name"] = str(payload["display_name"])
        if payload.get("model_type"):
            model["model_type"] = str(payload["model_type"])
        model["label_mapping"] = normalize_mapping(model.get("names", {}), payload.get("label_mapping", {}))
        self._write_registry(registry)
        if registry.get("active_model_id") == model_id and self.detector is not None:
            self.detector.set_model_info(model)
        return {"model": self._public_model(model)}

    def activate_model(self, model_id: str) -> dict:
        registry = self._read_registry()
        model = self._find_model(model_id, registry)
        if not model.get("weight_path"):
            raise HTTPException(status_code=400, detail="This entry has no model weight and cannot be activated")
        old_active = registry.get("active_model_id")
        old_detector = self.detector
        try:
            detector = self._load_detector_for_model(model)
        except Exception as exc:
            self.detector = old_detector
            registry["active_model_id"] = old_active
            self._write_registry(registry)
            raise HTTPException(status_code=400, detail=f"Model activation failed: {exc}") from exc
        self.detector = detector
        self._sync_model_names_from_detector(model, detector)
        registry["active_model_id"] = model_id
        self._write_registry(registry)
        detector.set_model_info(model)
        self.log_runtime_state(model)
        return {"active_model": self._public_model(model)}

    def delete_model(self, model_id: str) -> dict:
        registry = self._read_registry()
        model = self._find_model(model_id, registry)
        if model.get("is_default"):
            raise HTTPException(status_code=400, detail="Default model cannot be deleted")
        if registry.get("active_model_id") == model_id:
            raise HTTPException(status_code=400, detail="Active model cannot be deleted")
        registry["models"] = [item for item in registry.get("models", []) if item.get("model_id") != model_id]
        for key in ("weight_path", "yaml_path"):
            path = _root_path(model.get(key))
            if path and path.exists() and config.ROOT_DIR in path.resolve().parents:
                path.unlink(missing_ok=True)
        self._write_registry(registry)
        return {"ok": True}

    def load_active_model(self, raise_on_error: bool = False) -> None:
        registry = self._read_registry()
        model = self._find_model(registry.get("active_model_id"), registry, required=False)
        if not model:
            self.detector = None
            return
        try:
            self.detector = self._load_detector_for_model(model)
            self._sync_model_names_from_detector(model, self.detector)
            self._write_registry(registry)
            self.detector.set_model_info(model)
            self.log_runtime_state(model)
        except Exception as exc:
            self.detector = None
            logger.error("[ModelManager] Active model could not be loaded: %s", exc)
            if raise_on_error:
                raise

    def parse_yaml_upload(self, yaml_file: UploadFile) -> dict:
        raise RuntimeError("Use parse_yaml_upload_async")

    async def parse_yaml_upload_async(self, yaml_file: UploadFile) -> dict:
        ext = _extension(yaml_file.filename)
        if ext not in config.ALLOWED_YAML_EXT:
            raise HTTPException(status_code=400, detail=f"Unsupported YAML extension: .{ext or 'unknown'}")
        temp_name = f"parse_{uuid.uuid4().hex[:12]}_{_safe_filename(yaml_file.filename or 'data.yaml')}"
        yaml_path = config.MODEL_UPLOAD_YAMLS_DIR / temp_name
        await _save_upload(yaml_file, yaml_path)
        names = parse_yaml_names(yaml_path)
        yaml_path.unlink(missing_ok=True)
        ids = [int(key) for key in names.keys()]
        contiguous = ids == list(range(min(ids or [0]), max(ids or [0]) + 1)) if ids else True
        return {"names": names, "classes": _names_to_classes(names), "is_contiguous": contiguous}

    def log_runtime_state(self, model: dict) -> None:
        logger.info("[ModelManager] CUDA available: %s", config.CUDA_AVAILABLE)
        logger.info("[ModelManager] Using device: %s", self.device)
        logger.info("[ModelManager] GPU: %s", config.GPU_NAME or "not available")
        logger.info("[ModelManager] Active model: %s", model.get("display_name"))
        logger.info("[ModelManager] Weight path: %s", model.get("weight_path"))

    def _load_detector_for_model(self, model: dict) -> YOLODetector:
        weight_path = _root_path(model.get("weight_path"))
        if not weight_path or not weight_path.exists():
            raise FileNotFoundError("Model weight file does not exist. Please upload a valid model.")
        if weight_path.suffix.lower() == ".pth":
            raise RuntimeError(".pth is not a standard Ultralytics YOLO model file. Please upload an exported .pt model.")
        return YOLODetector(weight_path, self.device, model_info=model)

    def _sync_model_names_from_detector(self, model: dict, detector: YOLODetector) -> None:
        if model.get("yaml_path"):
            return
        actual_names = {str(class_id): name for class_id, name in detector.get_model_classes().items()}
        if actual_names and actual_names != model.get("names", {}):
            logger.info("[ModelManager] Syncing registry classes from model.names for %s", model.get("display_name"))
            model["names"] = actual_names
            model["label_mapping"] = normalize_mapping(actual_names, model.get("label_mapping", {}))

    def _read_model_names(self, weight_path: Path) -> dict:
        if weight_path.suffix.lower() == ".pth":
            raise HTTPException(status_code=400, detail=".pth cannot be read as an Ultralytics YOLO model. Please upload .pt or provide YAML names.")
        try:
            from ultralytics import YOLO

            model = YOLO(str(weight_path))
            names = getattr(model, "names", {}) or {}
            if isinstance(names, dict):
                return {str(int(key)): str(value) for key, value in names.items()}
            if isinstance(names, list):
                return {str(idx): str(value) for idx, value in enumerate(names)}
            return {}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read model classes: {exc}") from exc

    def _ensure_registry(self) -> None:
        config.ensure_runtime_dirs()
        if not config.MODEL_REGISTRY_PATH.exists():
            default_record = self._default_record()
            self._write_registry({"active_model_id": config.DEFAULT_MODEL_ID, "models": [default_record]})
            return
        registry = self._read_registry()
        if not self._find_model(config.DEFAULT_MODEL_ID, registry, required=False):
            registry.setdefault("models", []).insert(0, self._default_record())
            registry.setdefault("active_model_id", config.DEFAULT_MODEL_ID)
            self._write_registry(registry)

    def _default_record(self) -> dict:
        names = dict(config.DEFAULT_MODEL_NAMES)
        return {
            "model_id": config.DEFAULT_MODEL_ID,
            "display_name": config.DEFAULT_MODEL_DISPLAY_NAME,
            "model_type": config.DEFAULT_MODEL_TYPE,
            "weight_path": _relative_to_root(config.MODEL_PATH),
            "yaml_path": None,
            "original_weight_filename": config.MODEL_PATH.name,
            "stored_weight_filename": f"default_{config.MODEL_PATH.name}",
            "original_yaml_filename": None,
            "stored_yaml_filename": None,
            "names": names,
            "label_mapping": default_label_mapping(names),
            "created_at": _now(),
            "is_default": True,
            "is_detection_model": True,
        }

    def _read_registry(self) -> dict:
        try:
            return json.loads(config.MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="registry.json is invalid") from exc

    def _write_registry(self, registry: dict) -> None:
        config.MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.MODEL_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_model(self, model_id: str | None, registry: dict, required: bool = True) -> Optional[dict]:
        for model in registry.get("models", []):
            if model.get("model_id") == model_id:
                return model
        if required:
            raise HTTPException(status_code=404, detail="Model not found")
        return None

    def _public_model(self, model: dict | None) -> dict | None:
        if not model:
            return None
        return dict(model)


def _extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def _safe_filename(filename: str) -> str:
    name = Path(str(filename).replace("\\", "/")).name or "file"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


async def _save_upload(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    with target.open("wb") as buffer:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                buffer.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File is larger than {config.MAX_UPLOAD_MB} MB")
            buffer.write(chunk)
    await upload.close()


def _root_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (config.ROOT_DIR / path).resolve()


def _relative_to_root(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(config.ROOT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _names_to_classes(names: dict) -> list[dict]:
    return [{"id": int(class_id), "name": name} for class_id, name in sorted(names.items(), key=lambda item: int(item[0]))]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
