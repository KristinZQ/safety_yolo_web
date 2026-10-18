from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from fastapi import HTTPException

from app import config

DEFAULT_CLASSES = [
    {"id": 0, "name": "person", "color": "#00AEEF"},
    {"id": 1, "name": "helmet", "color": "#00CC66"},
    {"id": 2, "name": "no_helmet", "color": "#FF9900"},
    {"id": 3, "name": "safety_vest", "color": "#9933FF"},
    {"id": 4, "name": "no_vest", "color": "#FF3333"},
]


def ensure_class_config() -> None:
    config.ensure_runtime_dirs()
    if not config.ANNOTATION_CLASSES_PATH.exists():
        save_classes(DEFAULT_CLASSES)


def load_classes() -> List[dict]:
    ensure_class_config()
    try:
        data = json.loads(config.ANNOTATION_CLASSES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="classes.json is invalid") from exc
    classes = data.get("classes", data if isinstance(data, list) else [])
    return normalize_classes(classes)


def save_classes(classes: Iterable[dict]) -> List[dict]:
    config.ensure_runtime_dirs()
    normalized = normalize_classes(classes)
    payload = {"classes": normalized}
    config.ANNOTATION_CLASSES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def classes_from_model_names(model_classes: Dict[int, str]) -> List[dict]:
    palette = ["#00AEEF", "#00CC66", "#FF9900", "#9933FF", "#FF3333", "#4D96FF", "#F46036", "#2EC4B6"]
    result = []
    for idx, class_id in enumerate(sorted(model_classes.keys(), key=int)):
        result.append({"id": int(class_id), "name": str(model_classes[class_id]), "color": palette[idx % len(palette)]})
    return result


def normalize_classes(classes: Iterable[dict]) -> List[dict]:
    normalized = []
    seen_ids = set()
    for item in classes:
        try:
            class_id = int(item.get("id"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Each class must have an integer id") from exc
        name = str(item.get("name", "")).strip()
        color = str(item.get("color", "")).strip()
        if class_id < 0:
            raise HTTPException(status_code=400, detail="Class id must be >= 0")
        if class_id in seen_ids:
            raise HTTPException(status_code=400, detail=f"Duplicate class id: {class_id}")
        if not name:
            raise HTTPException(status_code=400, detail="Class name cannot be empty")
        if not _valid_hex_color(color):
            raise HTTPException(status_code=400, detail=f"Invalid color for class {name}")
        seen_ids.add(class_id)
        normalized.append({"id": class_id, "name": name, "color": color.upper()})
    normalized.sort(key=lambda item: item["id"])
    return normalized


def _valid_hex_color(color: str) -> bool:
    if len(color) != 7 or not color.startswith("#"):
        return False
    try:
        int(color[1:], 16)
        return True
    except ValueError:
        return False
