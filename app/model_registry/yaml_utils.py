from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import HTTPException


def parse_yaml_names(path: Path) -> Dict[str, str]:
    try:
        import yaml
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PyYAML is required to parse YAML files") from exc

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = yaml.safe_load(path.read_text(encoding="gbk"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"YAML parse failed: {exc}") from exc

    if not isinstance(data, dict) or "names" not in data:
        raise HTTPException(status_code=400, detail="YAML must contain a names field")
    return normalize_names(data["names"])


def normalize_names(names: object) -> Dict[str, str]:
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                class_id = int(key)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="YAML names keys must be class ids") from exc
            normalized[str(class_id)] = str(value)
        return dict(sorted(normalized.items(), key=lambda item: int(item[0])))
    if isinstance(names, list):
        return {str(idx): str(name) for idx, name in enumerate(names)}
    raise HTTPException(status_code=400, detail="YAML names must be a dict or list")


def names_to_classes(names: Dict[str, str]) -> list[dict]:
    return [{"id": int(class_id), "name": name} for class_id, name in sorted(names.items(), key=lambda item: int(item[0]))]
