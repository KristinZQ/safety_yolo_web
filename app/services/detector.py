from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from app import config
from app.utils.draw_utils import draw_detections
from app.utils.violation_utils import apply_violation_rules, canonical_class_name, normalize_class_name

logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(self, model_path: Path, device: str, model_info: dict | None = None) -> None:
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.model_info = model_info or {}
        self._lock = threading.Lock()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weight file not found: {self.model_path}. "
                "Place your YOLOv11 weight at weights/best.pt or set MODEL_PATH in .env."
            )

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        logger.info("Loading YOLO model once from %s on %s", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))
        self.names = self._read_model_names()
        if self.model_info.get("yaml_path") and self.model_info.get("names"):
            self.names = {int(k): str(v) for k, v in self.model_info["names"].items()}
        logger.info("Model loaded with classes: %s", self.names)

    def set_model_info(self, model_info: dict) -> None:
        self.model_info = model_info or {}

    def _read_model_names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}

    def get_model_classes(self) -> Dict[int, str]:
        return dict(self.names)

    @staticmethod
    def normalize_class_name(name: object) -> str:
        return normalize_class_name(name)

    def build_detection_list(self, result: object) -> List[dict]:
        detections: List[dict] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for idx, box in enumerate(boxes):
            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            except Exception:
                continue

            class_name = self.names.get(class_id, str(class_id))
            mapping = self._mapping_for_class(class_name)
            business_label = mapping.get("business_label") or mapping.get("display_label") or class_name
            detections.append(
                {
                    "id": idx + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "display_label": mapping.get("display_label") or business_label,
                    "business_label": business_label,
                    "report_label": mapping.get("report_label") or business_label,
                    "confidence": round(confidence, 4),
                    "bbox": [round(float(value), 2) for value in xyxy],
                    "is_violation": bool(mapping.get("is_violation", False)),
                    "violation_type": mapping.get("violation_type"),
                }
            )
        return detections

    def _predict(self, image: np.ndarray, conf: float, iou: float, imgsz: int) -> dict:
        started = time.perf_counter()
        with self._lock:
            results = self.model.predict(
                source=image,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                device=self.device,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - started) * 1000.0
        result = results[0] if results else None
        detections = self.build_detection_list(result) if result is not None else []
        if self.model_info.get("model_type", config.DEFAULT_MODEL_TYPE) == "safety_ppe":
            detections, stats = apply_violation_rules(
                detections=detections,
                model_classes=self.names,
                aliases=config.CLASS_ALIASES,
                mode=config.VIOLATION_MODE,
                inference_ms=inference_ms,
            )
            detections = self._reapply_label_mapping(detections)
            stats.update(self._mapped_violation_stats(detections))
        else:
            detections = self._reapply_label_mapping(detections)
            stats = self._generic_stats(detections, inference_ms)
        stats["model_type"] = self.model_info.get("model_type", config.DEFAULT_MODEL_TYPE)
        stats["model_display_name"] = self.model_info.get("display_name", config.DEFAULT_MODEL_DISPLAY_NAME)
        stats["report_counts"] = self._report_counts(detections)
        stats["report_cards"] = self._report_cards(detections, stats)
        return {"detections": detections, "stats": stats}

    def predict_image(self, image: np.ndarray, conf: float, iou: float, imgsz: int) -> dict:
        return self._predict(image, conf, iou, imgsz)

    def predict_frame(self, frame: np.ndarray, conf: float, iou: float, imgsz: int) -> dict:
        return self._predict(frame, conf, iou, imgsz)

    def draw_result(self, frame: np.ndarray, detections: List[dict], stats: dict) -> np.ndarray:
        return draw_detections(frame, detections, stats)

    def _mapping_for_class(self, class_name: str) -> dict:
        return dict((self.model_info.get("label_mapping") or {}).get(class_name, {}))

    def _reapply_label_mapping(self, detections: List[dict]) -> List[dict]:
        result = []
        for det in detections:
            item = dict(det)
            mapping = self._mapping_for_class(str(item.get("class_name", "")))
            if mapping:
                item["display_label"] = mapping.get("display_label") or mapping.get("business_label") or item.get("class_name")
                item["business_label"] = mapping.get("business_label") or item.get("display_label")
                item["report_label"] = mapping.get("report_label") or item.get("business_label") or item.get("class_name")
                if mapping.get("is_violation"):
                    item["is_violation"] = True
                    item["violation_type"] = mapping.get("violation_type") or item.get("violation_type") or "custom_violation"
            else:
                item.setdefault("display_label", item.get("class_name"))
                item.setdefault("business_label", item.get("class_name"))
                item.setdefault("report_label", item.get("business_label") or item.get("class_name"))
            result.append(item)
        return result

    def _generic_stats(self, detections: List[dict], inference_ms: float) -> dict:
        class_counts: dict[str, int] = {}
        violation_counts: dict[str, int] = {}
        confidences = []
        for det in detections:
            label = str(det.get("report_label") or det.get("business_label") or det.get("class_name"))
            class_counts[label] = class_counts.get(label, 0) + 1
            confidences.append(float(det.get("confidence", 0.0)))
            if det.get("is_violation"):
                violation_type = str(det.get("violation_type") or label)
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        return {
            "total_detections": len(detections),
            "class_counts": class_counts,
            "violation_counts": violation_counts,
            "violation_total": sum(violation_counts.values()),
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "fps": round(1000.0 / inference_ms, 4) if inference_ms > 0 else 0.0,
            "inference_ms": round(float(inference_ms), 4),
        }

    def _mapped_violation_stats(self, detections: List[dict]) -> dict:
        violation_counts: dict[str, int] = {}
        for det in detections:
            if det.get("is_violation"):
                violation_type = str(det.get("violation_type") or det.get("business_label") or det.get("class_name"))
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        return {"violation_counts": violation_counts, "violation_total": sum(violation_counts.values())}

    def _report_counts(self, detections: List[dict]) -> List[dict]:
        counts: dict[str, dict] = {}
        for det in detections:
            class_name = str(det.get("class_name", "unknown"))
            label = str(det.get("report_label") or det.get("business_label") or class_name)
            if label not in counts:
                counts[label] = {"label": label, "value": 0, "class_name": class_name}
            counts[label]["value"] += 1
        return list(counts.values())

    def _report_cards(self, detections: List[dict], stats: dict) -> List[dict]:
        if self.model_info.get("model_type", config.DEFAULT_MODEL_TYPE) == "safety_ppe":
            cards = [
                {"label": self._report_label_for_category("person", "总人数"), "value": stats.get("person_count", 0)},
                {"label": self._report_label_for_category("vest", "已穿安全服"), "value": stats.get("vest_worn_count", 0)},
                {"label": self._report_label_for_category("no_vest", "未穿安全服"), "value": stats.get("no_vest_count", 0), "tone": "warning"},
                {"label": self._report_label_for_category("no_helmet", "未戴安全帽"), "value": stats.get("no_helmet_count", 0), "tone": "warning"},
                {"label": "平均违规置信度", "value": stats.get("avg_violation_confidence", 0)},
                {"label": "FPS", "value": stats.get("fps", 0)},
                {"label": "耗时 ms", "value": stats.get("inference_ms", 0)},
            ]
            used_categories = {"person", "vest", "helmet", "no_vest", "no_helmet"}
            extra_cards = []
            for item in self._report_counts(detections):
                category = canonical_class_name(item.get("class_name"), config.CLASS_ALIASES)
                if category not in used_categories:
                    extra_cards.append({"label": item["label"], "value": item["value"]})
            return cards + extra_cards

        cards = [
            {"label": "总目标数", "value": stats.get("total_detections", 0)},
            {"label": "违规目标数", "value": stats.get("violation_total", 0), "tone": "warning"},
            {"label": "平均置信度", "value": stats.get("avg_confidence", 0)},
        ]
        cards.extend({"label": item["label"], "value": item["value"]} for item in self._report_counts(detections))
        cards.extend(
            [
                {"label": "FPS", "value": stats.get("fps", 0)},
                {"label": "耗时 ms", "value": stats.get("inference_ms", 0)},
            ]
        )
        return cards

    def _report_label_for_category(self, category: str, fallback: str) -> str:
        for class_name in self.names.values():
            if canonical_class_name(class_name, config.CLASS_ALIASES) != category:
                continue
            mapping = self._mapping_for_class(str(class_name))
            report_label = str(mapping.get("report_label") or "").strip()
            if report_label and report_label != str(class_name):
                return report_label
        return fallback
