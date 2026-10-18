from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def validate_box(box: dict, image_width: int, image_height: int) -> Optional[dict]:
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    left = max(0.0, min(x1, x2))
    right = min(float(image_width), max(x1, x2))
    top = max(0.0, min(y1, y2))
    bottom = min(float(image_height), max(y1, y2))

    if right - left < 1 or bottom - top < 1:
        return None

    clean = dict(box)
    clean["class_id"] = int(clean.get("class_id", 0))
    clean["class_name"] = str(clean.get("class_name", clean["class_id"]))
    clean["color"] = str(clean.get("color", "#00AEEF"))
    clean["x1"] = round(left, 2)
    clean["y1"] = round(top, 2)
    clean["x2"] = round(right, 2)
    clean["y2"] = round(bottom, 2)
    return clean


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> Optional[tuple[float, float, float, float]]:
    if image_width <= 0 or image_height <= 0:
        return None

    left = max(0.0, min(float(x1), float(x2)))
    right = min(float(image_width), max(float(x1), float(x2)))
    top = max(0.0, min(float(y1), float(y2)))
    bottom = min(float(image_height), max(float(y1), float(y2)))

    width = right - left
    height = bottom - top
    if width < 1 or height < 1:
        return None

    x_center = ((left + right) / 2.0) / float(image_width)
    y_center = ((top + bottom) / 2.0) / float(image_height)
    norm_width = width / float(image_width)
    norm_height = height / float(image_height)
    return (
        round(_clamp01(x_center), 6),
        round(_clamp01(y_center), 6),
        round(_clamp01(norm_width), 6),
        round(_clamp01(norm_height), 6),
    )


def save_yolo_txt(label_path: Path, boxes: Iterable[dict], image_width: int, image_height: int) -> List[str]:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for box in boxes:
        clean = validate_box(box, image_width, image_height)
        if not clean:
            continue
        yolo = xyxy_to_yolo(clean["x1"], clean["y1"], clean["x2"], clean["y2"], image_width, image_height)
        if not yolo:
            continue
        lines.append(f'{int(clean["class_id"])} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}')
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def load_yolo_txt(label_path: Path, image_width: int, image_height: int, classes: Sequence[dict]) -> List[dict]:
    if not label_path.exists():
        return []
    class_map = {int(item["id"]): item for item in classes}
    boxes = []
    for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        box_width = width * image_width
        box_height = height * image_height
        cx = x_center * image_width
        cy = y_center * image_height
        class_info = class_map.get(class_id, {"name": str(class_id), "color": "#00AEEF"})
        boxes.append(
            {
                "id": f"loaded_{idx}",
                "class_id": class_id,
                "class_name": class_info["name"],
                "color": class_info["color"],
                "x1": round(cx - box_width / 2.0, 2),
                "y1": round(cy - box_height / 2.0, 2),
                "x2": round(cx + box_width / 2.0, 2),
                "y2": round(cy + box_height / 2.0, 2),
            }
        )
    return boxes


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
