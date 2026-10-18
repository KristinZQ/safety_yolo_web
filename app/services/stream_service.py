from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict
from urllib.parse import urlsplit, urlunsplit

import cv2
from fastapi import HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect

from app import config
from app.services.detector import YOLODetector
from app.utils.draw_utils import encode_jpeg_base64

logger = logging.getLogger(__name__)


@dataclass
class StreamSession:
    stream_id: str
    source: str
    conf: float
    iou: float
    imgsz: int
    created_at: float = field(default_factory=time.time)
    stop_event: threading.Event = field(default_factory=threading.Event)
    capture: cv2.VideoCapture | None = None


class StreamManager:
    def __init__(self) -> None:
        self._streams: Dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def start_stream(self, source: str, conf: float, iou: float, imgsz: int) -> str:
        source = (source or "0").strip()
        with self._lock:
            self._drop_stopped_locked()
            if len(self._streams) >= config.MAX_STREAMS:
                raise HTTPException(status_code=429, detail=f"Too many streams. Max is {config.MAX_STREAMS}.")
            stream_id = uuid.uuid4().hex
            self._streams[stream_id] = StreamSession(stream_id, source, conf, iou, imgsz)
        logger.info("Created stream %s for %s", stream_id, self._mask_source(source))
        return stream_id

    def stop_stream(self, stream_id: str) -> dict:
        session = self._streams.get(stream_id)
        if not session:
            raise HTTPException(status_code=404, detail="Stream not found")
        self._stop_session(stream_id, session)
        return {"stream_id": stream_id, "status": "stopped"}

    def stop_all(self) -> None:
        for stream_id, session in list(self._streams.items()):
            self._stop_session(stream_id, session)

    async def websocket_loop(self, websocket: WebSocket, stream_id: str, detector: YOLODetector) -> None:
        await websocket.accept()
        session = self._streams.get(stream_id)
        if not session:
            await websocket.send_json({"type": "error", "message": "Stream not found or already stopped"})
            await websocket.close()
            return

        try:
            capture = self._ensure_capture(session)
            if not capture.isOpened():
                await websocket.send_json({"type": "error", "message": "Cannot open camera or stream source"})
                return

            min_interval = 1.0 / max(config.STREAM_MAX_FPS, 1)
            while not session.stop_event.is_set():
                frame_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok or frame is None:
                    await websocket.send_json({"type": "error", "message": "Failed to read frame from stream"})
                    break

                prediction = detector.predict_frame(frame, session.conf, session.iou, session.imgsz)
                annotated = detector.draw_result(frame, prediction["detections"], prediction["stats"])
                image = encode_jpeg_base64(annotated)
                await websocket.send_json(
                    {
                        "type": "frame",
                        "image": image,
                        "stats": prediction["stats"],
                        "detections": prediction["detections"],
                    }
                )

                elapsed = time.perf_counter() - frame_started
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for stream %s", stream_id)
        except Exception as exc:
            logger.exception("Stream %s failed", stream_id)
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            self._stop_session(stream_id, session)

    def _ensure_capture(self, session: StreamSession) -> cv2.VideoCapture:
        if session.capture is not None:
            return session.capture
        source_obj: int | str = int(session.source) if session.source.isdigit() else session.source
        session.capture = cv2.VideoCapture(source_obj)
        return session.capture

    def _stop_session(self, stream_id: str, session: StreamSession) -> None:
        session.stop_event.set()
        if session.capture is not None:
            session.capture.release()
            session.capture = None
        with self._lock:
            self._streams.pop(stream_id, None)
        logger.info("Stopped stream %s", stream_id)

    def _drop_stopped_locked(self) -> None:
        stopped = [stream_id for stream_id, session in self._streams.items() if session.stop_event.is_set()]
        for stream_id in stopped:
            session = self._streams.pop(stream_id, None)
            if session and session.capture is not None:
                session.capture.release()

    @staticmethod
    def _mask_source(source: str) -> str:
        try:
            parsed = urlsplit(source)
            if parsed.password:
                username = parsed.username or ""
                netloc = f"{username}:***@{parsed.hostname or ''}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        except Exception:
            pass
        return source
