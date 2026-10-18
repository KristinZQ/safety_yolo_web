from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app import config

Detection = Dict[str, object]


def normalize_class_name(name: object) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text


def build_alias_lookup(aliases: Dict[str, Sequence[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, names in aliases.items():
        lookup[normalize_class_name(canonical)] = canonical
        for name in names:
            lookup[normalize_class_name(name)] = canonical
    return lookup


def canonical_class_name(name: object, aliases: Dict[str, Sequence[str]]) -> str:
    lookup = build_alias_lookup(aliases)
    normalized = normalize_class_name(name)
    return lookup.get(normalized, normalized)


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def bbox_intersection(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def center_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in inner]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def class_set(model_classes: Dict[int, str], aliases: Dict[str, Sequence[str]]) -> set[str]:
    return {canonical_class_name(name, aliases) for name in model_classes.values()}


def _detections_by_category(detections: Iterable[Detection], category: str) -> List[Detection]:
    return [det for det in detections if det.get("category") == category]


def _round(value: float) -> float:
    return round(float(value), 4)


def apply_violation_rules(
    detections: List[Detection],
    model_classes: Dict[int, str],
    aliases: Dict[str, Sequence[str]],
    mode: str = "auto",
    inference_ms: float = 0.0,
) -> Tuple[List[Detection], Dict[str, object]]:
    for det in detections:
        det["category"] = canonical_class_name(det.get("class_name"), aliases)
        det["is_violation"] = False
        det["violation_type"] = None

    persons = _detections_by_category(detections, "person")
    vests = _detections_by_category(detections, "vest")
    no_vests = _detections_by_category(detections, "no_vest")
    no_helmets = _detections_by_category(detections, "no_helmet")
    available_classes = class_set(model_classes, aliases)

    direct_available = "no_vest" in available_classes
    use_direct = mode == "direct_no_vest" or (mode == "auto" and direct_available)
    use_person_without_vest = mode == "person_without_vest" or (mode == "auto" and not direct_available)

    violation_confidences: List[float] = []
    no_vest_count = 0
    inferred_no_vest_count = 0

    if use_direct:
        for det in no_vests:
            det["is_violation"] = True
            det["violation_type"] = "no_vest"
            violation_confidences.append(float(det.get("confidence", 0.0)))
        no_vest_count = len(no_vests)

    if use_person_without_vest and persons:
        for person in persons:
            person_bbox = person.get("bbox", [0, 0, 0, 0])
            person_area = max(bbox_area(person_bbox), 1.0)
            max_vest_conf_inside = 0.0
            has_valid_vest = False
            for vest in vests:
                vest_conf = float(vest.get("confidence", 0.0))
                if vest_conf < config.VEST_CONF_THRESHOLD:
                    continue
                vest_bbox = vest.get("bbox", [0, 0, 0, 0])
                overlap_ratio = bbox_intersection(person_bbox, vest_bbox) / person_area
                if center_inside(vest_bbox, person_bbox) or overlap_ratio >= config.VEST_OVERLAP_THRESHOLD:
                    has_valid_vest = True
                    max_vest_conf_inside = max(max_vest_conf_inside, vest_conf)
            if not has_valid_vest:
                person["is_violation"] = True
                person["violation_type"] = "person_without_vest"
                # Absence of a vest has no vest box confidence, so the person confidence
                # is used as the violation confidence for this inferred violation.
                violation_confidences.append(float(person.get("confidence", 0.0)))
                inferred_no_vest_count += 1

    if use_person_without_vest and not use_direct:
        no_vest_count = inferred_no_vest_count

    for det in no_helmets:
        det["is_violation"] = True
        det["violation_type"] = "no_helmet"

    clean_detections = []
    for det in detections:
        item = dict(det)
        item.pop("category", None)
        clean_detections.append(item)

    no_helmet_count = len(no_helmets)
    avg_conf = sum(violation_confidences) / len(violation_confidences) if violation_confidences else 0.0
    avg_all_conf = (
        sum(float(det.get("confidence", 0.0)) for det in detections) / len(detections)
        if detections
        else 0.0
    )
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    stats = {
        "person_count": len(persons),
        "vest_count": len(vests),
        "vest_worn_count": max(0, len(persons) - no_vest_count) if persons else len(vests),
        "no_vest_count": no_vest_count,
        "inferred_no_vest_count": inferred_no_vest_count,
        "no_helmet_count": no_helmet_count,
        "violation_confidences": [_round(v) for v in violation_confidences],
        "avg_violation_confidence": _round(avg_conf),
        "max_violation_confidence": _round(max(violation_confidences) if violation_confidences else 0.0),
        "min_violation_confidence": _round(min(violation_confidences) if violation_confidences else 0.0),
        "avg_confidence": _round(avg_all_conf),
        "fps": _round(fps),
        "inference_ms": _round(inference_ms),
        "violation_mode": "direct_no_vest" if use_direct else ("person_without_vest" if use_person_without_vest else "none"),
    }
    return clean_detections, stats
