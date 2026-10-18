from __future__ import annotations

from typing import Dict


PALETTE = ["#00AEEF", "#00CC66", "#FF9900", "#9933FF", "#FF3333", "#4D96FF", "#F46036", "#2EC4B6"]


def default_label_mapping(names: Dict[str, str]) -> dict:
    mapping = {}
    for idx, (class_id, label) in enumerate(sorted(names.items(), key=lambda item: int(item[0]))):
        label_text = str(label)
        violation_type = _default_violation_type(label_text)
        mapping[label_text] = {
            "class_id": int(class_id),
            "model_label": label_text,
            "display_label": label_text,
            "business_label": label_text,
            "report_label": label_text,
            "color": PALETTE[idx % len(PALETTE)],
            "is_violation": violation_type is not None,
            "violation_type": violation_type,
        }
    return mapping


def normalize_mapping(names: Dict[str, str], mapping: dict | None) -> dict:
    defaults = default_label_mapping(names)
    mapping = mapping or {}
    result = {}
    for class_id, label in sorted(names.items(), key=lambda item: int(item[0])):
        label_text = str(label)
        current = dict(defaults[label_text])
        provided = mapping.get(label_text, {})
        current.update(
            {
                "class_id": int(class_id),
                "model_label": label_text,
                "display_label": str(provided.get("display_label") or provided.get("business_label") or current["display_label"]),
                "business_label": str(provided.get("business_label") or current["business_label"]),
                "report_label": str(provided.get("report_label") or provided.get("business_label") or current["report_label"]),
                "color": str(provided.get("color") or current["color"]),
                "is_violation": bool(provided.get("is_violation", current["is_violation"])),
                "violation_type": provided.get("violation_type", current["violation_type"]) or None,
            }
        )
        result[label_text] = current
    return result


def _default_violation_type(label: str) -> str | None:
    normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"no_vest", "no_safety_vest", "without_vest"}:
        return "no_vest"
    if normalized in {"no_helmet", "no_hardhat", "without_helmet"}:
        return "no_helmet"
    return None
