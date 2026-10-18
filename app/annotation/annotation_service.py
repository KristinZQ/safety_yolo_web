from __future__ import annotations

import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

from app import config
from app.annotation.label_utils import save_yolo_txt, validate_box

CHUNK_SIZE = 1024 * 1024


def dataset_dir(dataset_id: str) -> Path:
    root = config.ANNOTATION_DATASETS_DIR.resolve()
    target = (root / dataset_id).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid dataset id")
    return target


def create_dataset_dirs(dataset_id: str) -> tuple[Path, Path, Path]:
    base = dataset_dir(dataset_id)
    images_dir = base / "images"
    labels_dir = base / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return base, images_dir, labels_dir


async def create_dataset_from_uploads(files: List[UploadFile], relative_paths: Optional[List[str]] = None) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No image files uploaded")
    dataset_id = uuid.uuid4().hex
    base, images_dir, _ = create_dataset_dirs(dataset_id)
    images = []
    relative_paths = relative_paths or []
    used_label_filenames: set[str] = set()

    for index, upload in enumerate(files):
        _validate_image_upload(upload)
        ext = Path(upload.filename or "").suffix.lower().lstrip(".")
        image_id = uuid.uuid4().hex
        safe_name = f"{image_id}.{ext}"
        target = images_dir / safe_name
        await _save_upload(upload, target)
        width, height = _read_image_size(target)
        relative_path = relative_paths[index] if index < len(relative_paths) else (upload.filename or safe_name)
        original_name = _safe_original_filename(relative_path or upload.filename or safe_name)
        original_stem = _safe_stem(original_name)
        label_filename = _unique_label_filename(original_stem, used_label_filenames)
        images.append(
            {
                "image_id": image_id,
                "stored_filename": safe_name,
                "filename": safe_name,
                "original_filename": original_name,
                "original_stem": original_stem,
                "label_filename": label_filename,
                "relative_path": relative_path,
                "url": f"/annotation-files/{dataset_id}/images/{safe_name}",
                "width": width,
                "height": height,
                "labeled": False,
                "label_url": f"/api/annotation/datasets/{dataset_id}/images/{image_id}/label",
            }
        )

    meta = {
        "dataset_id": dataset_id,
        "created_at": time.time(),
        "updated_at": time.time(),
        "total": len(images),
        "labeled_count": 0,
        "images": images,
    }
    write_json(base / "meta.json", meta)
    write_json(base / "annotations.json", {})
    return {"dataset_id": dataset_id, "images": images, "total": len(images), "labeled_count": 0}


def get_dataset(dataset_id: str) -> dict:
    base = dataset_dir(dataset_id)
    meta = read_json(base / "meta.json")
    meta["labeled_count"] = sum(1 for item in meta.get("images", []) if item.get("labeled"))
    meta["total"] = len(meta.get("images", []))
    return meta


def get_image_annotations(dataset_id: str, image_id: str) -> dict:
    meta = get_dataset(dataset_id)
    image = find_image(meta, image_id)
    annotations = read_json(dataset_dir(dataset_id) / "annotations.json", default={})
    record = annotations.get(image_id, {"image_width": image["width"], "image_height": image["height"], "boxes": []})
    return {"dataset_id": dataset_id, "image": image, **record}


def save_image_annotations(dataset_id: str, image_id: str, image_width: int, image_height: int, boxes: List[dict]) -> dict:
    base = dataset_dir(dataset_id)
    meta = get_dataset(dataset_id)
    image = find_image(meta, image_id)
    # Use dimensions read from the uploaded image metadata, not canvas display
    # dimensions supplied by the browser.
    width = int(image["width"])
    height = int(image["height"])
    clean_boxes = []
    for box in boxes:
        clean = validate_box(box, width, height)
        if clean:
            clean_boxes.append(clean)

    annotations_path = base / "annotations.json"
    annotations = read_json(annotations_path, default={})
    annotations[image_id] = {"image_width": width, "image_height": height, "boxes": clean_boxes, "updated_at": time.time()}
    write_json(annotations_path, annotations)

    label_path = base / "labels" / label_filename_for_image(image)
    save_yolo_txt(label_path, clean_boxes, width, height)

    for item in meta.get("images", []):
        if item["image_id"] == image_id:
            item["labeled"] = True
            item["label_url"] = f"/api/annotation/datasets/{dataset_id}/images/{image_id}/label"
            break
    meta["updated_at"] = time.time()
    meta["labeled_count"] = sum(1 for item in meta.get("images", []) if item.get("labeled"))
    write_json(base / "meta.json", meta)

    return {
        "ok": True,
        "label_url": f"/api/annotation/datasets/{dataset_id}/images/{image_id}/label",
        "label_filename": label_path.name,
        "next_image_id": next_unlabeled_image_id(meta, image_id),
        "boxes": clean_boxes,
    }


def clear_image_annotations(dataset_id: str, image_id: str) -> dict:
    base = dataset_dir(dataset_id)
    meta = get_dataset(dataset_id)
    image = find_image(meta, image_id)
    annotations_path = base / "annotations.json"
    annotations = read_json(annotations_path, default={})
    annotations.pop(image_id, None)
    write_json(annotations_path, annotations)
    label_path = base / "labels" / label_filename_for_image(image)
    label_path.unlink(missing_ok=True)

    for item in meta.get("images", []):
        if item["image_id"] == image_id:
            item["labeled"] = False
            break
    meta["updated_at"] = time.time()
    meta["labeled_count"] = sum(1 for item in meta.get("images", []) if item.get("labeled"))
    write_json(base / "meta.json", meta)
    return {"ok": True, "image_id": image_id, "labeled": False}


def label_file_path(dataset_id: str, image_id: str) -> Path:
    meta = get_dataset(dataset_id)
    image = find_image(meta, image_id)
    return dataset_dir(dataset_id) / "labels" / label_filename_for_image(image)


def labels_zip_path(dataset_id: str) -> Path:
    base = dataset_dir(dataset_id)
    labels_dir = base / "labels"
    zip_path = base / "labels.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for label in sorted(labels_dir.glob("*.txt")):
            archive.write(label, arcname=label.name)
    return zip_path


def label_filename_for_image(image: dict) -> str:
    if image.get("label_filename"):
        return str(image["label_filename"])
    if image.get("original_stem"):
        return f"{_safe_stem(str(image['original_stem']))}.txt"
    if image.get("original_filename"):
        return f"{_safe_stem(str(image['original_filename']))}.txt"
    return f"{Path(str(image.get('filename', image.get('stored_filename', 'label')))).stem}.txt"


def find_image(meta: dict, image_id: str) -> dict:
    for item in meta.get("images", []):
        if item.get("image_id") == image_id:
            return item
    raise HTTPException(status_code=404, detail="Image not found")


def next_unlabeled_image_id(meta: dict, current_image_id: str) -> Optional[str]:
    images = meta.get("images", [])
    if not images:
        return None
    current_index = next((idx for idx, item in enumerate(images) if item.get("image_id") == current_image_id), -1)
    ordered = images[current_index + 1 :] + images[: current_index + 1]
    for item in ordered:
        if not item.get("labeled"):
            return item.get("image_id")
    return None


def read_json(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        if default is not None:
            return default
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON file: {path.name}") from exc


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_original_filename(value: str) -> str:
    name = Path(str(value).replace("\\", "/")).name
    return name or f"image_{uuid.uuid4().hex}.jpg"


def _safe_stem(value: str) -> str:
    stem = Path(str(value).replace("\\", "/")).stem.strip()
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stem)
    return clean.strip("._") or "image"


def _unique_label_filename(original_stem: str, used_label_filenames: set[str]) -> str:
    base = _safe_stem(original_stem)
    candidate = f"{base}.txt"
    suffix = 1
    while candidate.lower() in used_label_filenames:
        candidate = f"{base}_{suffix}.txt"
        suffix += 1
    used_label_filenames.add(candidate.lower())
    return candidate


def _validate_image_upload(upload: UploadFile) -> None:
    ext = Path(upload.filename or "").suffix.lower().lstrip(".")
    if ext not in config.ALLOWED_IMAGE_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension: .{ext or 'unknown'}")
    content_type = (upload.content_type or "").lower()
    if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported image MIME type: {content_type}")


async def _save_upload(upload: UploadFile, target: Path) -> None:
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


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid image: {path.name}") from exc
