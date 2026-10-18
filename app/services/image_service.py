from __future__ import annotations

import cv2
from fastapi import HTTPException, UploadFile

from app import config
from app.services.detector import YOLODetector
from app.utils.file_utils import save_upload_file, url_for_output, url_for_upload, validate_upload, output_path


async def detect_image(
    upload: UploadFile,
    detector: YOLODetector,
    conf: float,
    iou: float,
    imgsz: int,
) -> dict:
    validate_upload(upload, config.ALLOWED_IMAGE_EXT, config.ALLOWED_IMAGE_MIME_PREFIXES, "image")
    source_path = await save_upload_file(upload, config.UPLOAD_DIR, config.ALLOWED_IMAGE_EXT)

    image = cv2.imread(str(source_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Image cannot be decoded by OpenCV")

    result = detector.predict_image(image, conf, iou, imgsz)
    annotated = detector.draw_result(image, result["detections"], result["stats"])
    target_path = output_path("image_result", ".jpg")
    if not cv2.imwrite(str(target_path), annotated):
        raise HTTPException(status_code=500, detail="Failed to save detected image")

    return {
        "filename": source_path.name,
        "original_image_url": url_for_upload(source_path),
        "result_image_url": url_for_output(target_path),
        "detections": result["detections"],
        "stats": result["stats"],
        "model_classes": detector.get_model_classes(),
    }


async def detect_images(
    uploads: list[UploadFile],
    detector: YOLODetector,
    conf: float,
    iou: float,
    imgsz: int,
) -> dict:
    results = []
    for upload in uploads:
        results.append(await detect_image(upload, detector, conf, iou, imgsz))
    return {"count": len(results), "results": results, "model_classes": detector.get_model_classes()}
