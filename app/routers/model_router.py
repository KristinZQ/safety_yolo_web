from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/models", tags=["models"])


class LabelMappingPayload(BaseModel):
    display_name: Optional[str] = None
    model_type: Optional[str] = None
    label_mapping: dict = Field(default_factory=dict)


def manager(request: Request):
    return request.app.state.model_manager


@router.get("")
async def list_models(request: Request) -> dict:
    return manager(request).list_models()


@router.post("/upload")
async def upload_model(
    request: Request,
    weight_file: UploadFile | None = File(default=None),
    yaml_file: UploadFile | None = File(default=None),
    display_name: str | None = Form(default=None),
    model_type: str | None = Form(default=None),
) -> dict:
    return await manager(request).upload_model(weight_file, yaml_file, display_name, model_type)


@router.post("/{model_id}/mapping")
async def save_model_mapping(request: Request, model_id: str, payload: LabelMappingPayload) -> dict:
    return manager(request).save_mapping(model_id, payload.dict())


@router.post("/{model_id}/activate")
async def activate_model(request: Request, model_id: str) -> dict:
    return manager(request).activate_model(model_id)


@router.delete("/{model_id}")
async def delete_model(request: Request, model_id: str) -> dict:
    return manager(request).delete_model(model_id)


@router.post("/parse-yaml")
async def parse_yaml(request: Request, yaml_file: UploadFile = File(...)) -> dict:
    return await manager(request).parse_yaml_upload_async(yaml_file)
