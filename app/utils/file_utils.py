from __future__ import annotations

import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Optional, Set

from fastapi import HTTPException, UploadFile

from app import config

CHUNK_SIZE = 1024 * 1024


def extension_from_name(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")


def make_uuid_filename(original_name: Optional[str], allowed_ext: Set[str]) -> str:
    ext = extension_from_name(original_name)
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: .{ext or 'unknown'}")
    return f"{uuid.uuid4().hex}.{ext}"


def validate_upload(
    upload: UploadFile,
    allowed_ext: Set[str],
    allowed_mime_prefixes: Iterable[str],
    kind: str,
) -> None:
    ext = extension_from_name(upload.filename)
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported {kind} extension: .{ext or 'unknown'}")

    content_type = (upload.content_type or "").lower()
    guessed_type = (mimetypes.guess_type(upload.filename or "")[0] or "").lower()
    prefixes = tuple(prefix.lower() for prefix in allowed_mime_prefixes)
    if content_type and content_type.startswith(prefixes):
        return
    if guessed_type and guessed_type.startswith(prefixes):
        return
    raise HTTPException(status_code=400, detail=f"Unsupported {kind} MIME type: {content_type or 'unknown'}")


def assert_inside_directory(path: Path, directory: Path) -> None:
    resolved_path = path.resolve()
    resolved_dir = directory.resolve()
    if resolved_dir not in resolved_path.parents and resolved_path != resolved_dir:
        raise HTTPException(status_code=400, detail="Invalid file path")


async def save_upload_file(upload: UploadFile, directory: Path, allowed_ext: Set[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = make_uuid_filename(upload.filename, allowed_ext)
    target = directory / filename
    assert_inside_directory(target, directory)

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
    return target


def output_path(prefix: str, suffix: str) -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return config.OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex}{safe_suffix}"


def url_for_upload(path: Path) -> str:
    return f"/uploads/{path.name}"


def url_for_output(path: Path) -> str:
    return f"/outputs/{path.name}"


def clear_directory_contents(directory: Path, keep_names: Optional[Set[str]] = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    keep = keep_names or {".gitkeep"}
    removed: list[str] = []
    failed: list[dict[str, str]] = []

    for path in directory.iterdir():
        if path.name in keep:
            continue
        try:
            assert_inside_directory(path, directory)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)
        except OSError as exc:
            failed.append({"name": path.name, "error": str(exc)})

    return {"removed": removed, "failed": failed, "removed_count": len(removed), "failed_count": len(failed)}


def clear_detection_cache() -> dict:
    return {
        "uploads": clear_directory_contents(config.UPLOAD_DIR),
        "outputs": clear_directory_contents(config.OUTPUT_DIR),
    }
