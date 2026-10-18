from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.annotation import annotation_service
from app.annotation.class_config import classes_from_model_names, load_classes, save_classes

router = APIRouter()


class ClassItem(BaseModel):
    id: int
    name: str
    color: str


class ClassesPayload(BaseModel):
    classes: List[ClassItem] = Field(default_factory=list)


class AnnotationBox(BaseModel):
    id: str
    class_id: int
    class_name: str
    color: str
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationPayload(BaseModel):
    image_width: int
    image_height: int
    boxes: List[AnnotationBox] = Field(default_factory=list)


@router.get("/annotation")
async def annotation_page() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "annotation.html")


@router.get("/api/annotation/classes")
async def get_classes() -> dict:
    return {"classes": load_classes()}


@router.post("/api/annotation/classes")
async def post_classes(payload: ClassesPayload) -> dict:
    return {"classes": save_classes([item.dict() for item in payload.classes])}


@router.get("/api/annotation/classes/model")
async def get_classes_from_model(request: Request) -> dict:
    detector = request.app.state.model_manager.get_active_detector()
    return {"classes": classes_from_model_names(detector.get_model_classes())}


@router.post("/api/annotation/upload")
async def upload_annotation_images(
    files: List[UploadFile] = File(...),
    relative_paths: List[str] = Form(default=[]),
) -> dict:
    return await annotation_service.create_dataset_from_uploads(files, relative_paths)


@router.get("/api/annotation/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict:
    return annotation_service.get_dataset(dataset_id)


@router.get("/api/annotation/datasets/{dataset_id}/images/{image_id}/annotations")
async def get_image_annotations(dataset_id: str, image_id: str) -> dict:
    return annotation_service.get_image_annotations(dataset_id, image_id)


@router.post("/api/annotation/datasets/{dataset_id}/images/{image_id}/annotations")
async def save_image_annotations(dataset_id: str, image_id: str, payload: AnnotationPayload) -> dict:
    return annotation_service.save_image_annotations(
        dataset_id,
        image_id,
        payload.image_width,
        payload.image_height,
        [item.dict() for item in payload.boxes],
    )


@router.get("/api/annotation/datasets/{dataset_id}/images/{image_id}/label")
async def download_label(dataset_id: str, image_id: str) -> FileResponse:
    label_path = annotation_service.label_file_path(dataset_id, image_id)
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="Label file has not been generated yet")
    return FileResponse(label_path, media_type="text/plain", filename=label_path.name)


@router.get("/api/annotation/datasets/{dataset_id}/labels.zip")
async def download_labels_zip(dataset_id: str) -> FileResponse:
    zip_path = annotation_service.labels_zip_path(dataset_id)
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


@router.delete("/api/annotation/datasets/{dataset_id}/images/{image_id}/annotations")
async def clear_image_annotations(dataset_id: str, image_id: str) -> dict:
    return annotation_service.clear_image_annotations(dataset_id, image_id)
