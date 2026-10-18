from __future__ import annotations

import base64
import hashlib
from typing import Dict, List, Tuple

import cv2
import numpy as np

Detection = Dict[str, object]


def color_for_class(class_name: str, is_violation: bool = False) -> Tuple[int, int, int]:
    if is_violation:
        return (40, 40, 230)
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    return (int(digest[0]) % 170 + 50, int(digest[1]) % 170 + 50, int(digest[2]) % 170 + 50)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 4)
    cv2.rectangle(frame, (x, top), (x + width + 6, top + height + baseline + 4), color, -1)
    cv2.putText(frame, text, (x + 3, top + height + 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: List[Detection], stats: Dict[str, object]) -> np.ndarray:
    output = frame.copy()
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        class_name = str(det.get("class_name", "unknown"))
        confidence = float(det.get("confidence", 0.0))
        is_violation = bool(det.get("is_violation", False))
        color = color_for_class(class_name, is_violation)
        label = f"{class_name} {confidence:.2f}"
        if is_violation:
            label += " VIOLATION"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        draw_label(output, label, x1, y1, color)

    draw_stats_panel(output, stats)
    return output


def draw_stats_panel(frame: np.ndarray, stats: Dict[str, object]) -> None:
    if "total_detections" in stats:
        lines = [
            f"Total: {stats.get('total_detections', 0)}",
            f"Violations: {stats.get('violation_total', 0)}",
            f"Avg conf: {float(stats.get('avg_confidence', 0.0)):.2f}",
            f"FPS: {float(stats.get('fps', 0.0)):.1f}",
            f"Inference ms: {float(stats.get('inference_ms', 0.0)):.1f}",
        ]
    else:
        lines = [
            f"Persons: {stats.get('person_count', 0)}",
            f"Vest: {stats.get('vest_count', 0)}",
            f"No Vest: {stats.get('no_vest_count', 0)}",
            f"No Helmet: {stats.get('no_helmet_count', 0)}",
            f"FPS: {float(stats.get('fps', 0.0)):.1f}",
            f"Inference ms: {float(stats.get('inference_ms', 0.0)):.1f}",
        ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 22
    panel_width = 245
    panel_height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (20, 25, 35), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for idx, line in enumerate(lines):
        y = 33 + idx * line_height
        cv2.putText(frame, line, (22, y), font, scale, (245, 248, 255), thickness, cv2.LINE_AA)


def encode_jpeg_base64(frame: np.ndarray, quality: int = 80) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
