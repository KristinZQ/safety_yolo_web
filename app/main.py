from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.annotation.class_config import ensure_class_config
from app.model_registry.model_manager import ModelManager
from app.routers.annotation_router import router as annotation_router
from app.routers.model_router import router as model_router
from app.schemas import StreamStartRequest
from app.services.detector import YOLODetector
from app.services.image_service import detect_image, detect_images
from app.services.stream_service import StreamManager
from app.services.video_service import VideoTaskManager
from app.utils.file_utils import clear_detection_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    config.ensure_runtime_dirs()
    ensure_class_config()
    application.state.model_manager = ModelManager(config.DEVICE)
    application.state.video_manager = VideoTaskManager()
    application.state.stream_manager = StreamManager()
    yield
    application.state.stream_manager.stop_all()


app = FastAPI(
    title="Safety Vest and Helmet YOLO Detection",
    description="Local FastAPI web app for YOLOv11 safety vest and helmet detection.",
    version="1.0.0",
    lifespan=lifespan,
)

config.ensure_runtime_dirs()
ensure_class_config()
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
app.mount("/annotation-files", StaticFiles(directory=config.ANNOTATION_DATASETS_DIR), name="annotation_files")
app.include_router(annotation_router)
app.include_router(model_router)


def get_detector(request: Request) -> YOLODetector:
    return request.app.state.model_manager.get_active_detector()


def get_video_manager(request: Request) -> VideoTaskManager:
    return request.app.state.video_manager


def get_stream_manager(request: Request) -> StreamManager:
    return request.app.state.stream_manager


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled request error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "message": str(exc)})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
async def health(request: Request) -> dict:
    model_manager = request.app.state.model_manager
    active_model = model_manager.get_active_model()
    return {
        "server": "ok",
        "status": "ok",
        "model_loaded": model_manager.detector is not None,
        "active_model": active_model,
        "model_classes": model_manager.detector.get_model_classes() if model_manager.detector else {},
        "default_conf": config.DEFAULT_CONF,
        "default_iou": config.DEFAULT_IOU,
        "default_imgsz": config.DEFAULT_IMGSZ,
        "stream_max_fps": config.STREAM_MAX_FPS,
        "violation_mode": config.VIOLATION_MODE,
    }


@app.post("/api/detect/image")
async def detect_single_image(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Form(config.DEFAULT_CONF),
    iou: float = Form(config.DEFAULT_IOU),
    imgsz: int = Form(config.DEFAULT_IMGSZ),
) -> dict:
    return await detect_image(file, get_detector(request), conf, iou, imgsz)


@app.post("/api/cache/detection")
async def clear_detection_upload_cache(request: Request) -> dict:
    cache_result = clear_detection_cache()
    cleared_tasks = get_video_manager(request).clear_finished_tasks()
    return {"ok": True, "cleared_tasks": cleared_tasks, **cache_result}


@app.post("/api/detect/images")
async def detect_multiple_images(
    request: Request,
    files: List[UploadFile] = File(...),
    conf: float = Form(config.DEFAULT_CONF),
    iou: float = Form(config.DEFAULT_IOU),
    imgsz: int = Form(config.DEFAULT_IMGSZ),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No image files uploaded")
    return await detect_images(files, get_detector(request), conf, iou, imgsz)


@app.post("/api/detect/video")
async def detect_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = Form(config.DEFAULT_CONF),
    iou: float = Form(config.DEFAULT_IOU),
    imgsz: int = Form(config.DEFAULT_IMGSZ),
) -> dict:
    video_manager = get_video_manager(request)
    source_path = await video_manager.save_video_upload(file)
    task_id = video_manager.create_task(source_path, conf, iou, imgsz)
    background_tasks.add_task(
        video_manager.process_task,
        task_id,
        source_path,
        get_detector(request),
        conf,
        iou,
        imgsz,
    )
    return {
        "task_id": task_id,
        "status_url": f"/api/tasks/{task_id}",
        "source_video_url": f"/uploads/{source_path.name}",
    }


@app.get("/api/tasks/{task_id}")
async def get_task_status(request: Request, task_id: str) -> dict:
    return get_video_manager(request).get_task(task_id)


@app.post("/api/stream/start")
async def start_stream(request: Request, payload: StreamStartRequest) -> dict:
    stream_id = get_stream_manager(request).start_stream(payload.source, payload.conf, payload.iou, payload.imgsz)
    return {"stream_id": stream_id, "websocket_url": f"/ws/stream/{stream_id}"}


@app.websocket("/ws/stream/{stream_id}")
async def stream_websocket(websocket: WebSocket, stream_id: str) -> None:
    detector = websocket.app.state.model_manager.get_active_detector()
    stream_manager = websocket.app.state.stream_manager
    await stream_manager.websocket_loop(websocket, stream_id, detector)


@app.post("/api/stream/stop/{stream_id}")
async def stop_stream(request: Request, stream_id: str) -> dict:
    return get_stream_manager(request).stop_stream(stream_id)
