from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
from fastapi import HTTPException, UploadFile

from app import config
from app.services.detector import YOLODetector
from app.utils.file_utils import output_path, save_upload_file, url_for_output, url_for_upload, validate_upload

logger = logging.getLogger(__name__)


class VideoTaskManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, dict] = {}
        self._lock = threading.Lock()

    async def save_video_upload(self, upload: UploadFile) -> Path:
        validate_upload(upload, config.ALLOWED_VIDEO_EXT, config.ALLOWED_VIDEO_MIME_PREFIXES, "video")
        return await save_upload_file(upload, config.UPLOAD_DIR, config.ALLOWED_VIDEO_EXT)

    def create_task(self, source_path: Path, conf: float, iou: float, imgsz: int) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0.0,
                "message": "Waiting for background worker",
                "source_video_url": url_for_upload(source_path),
                "result_video_url": None,
                "summary_stats": None,
                "frame_count": 0,
                "duration": 0.0,
                "fps": 0.0,
                "params": {"conf": conf, "iou": iou, "imgsz": imgsz},
            }
        return task_id

    def get_task(self, task_id: str) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return dict(task)

    def clear_finished_tasks(self) -> int:
        with self._lock:
            before = len(self._tasks)
            self._tasks = {
                task_id: task
                for task_id, task in self._tasks.items()
                if task.get("status") in {"pending", "running"}
            }
            return before - len(self._tasks)

    def _update_task(self, task_id: str, **values: object) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(values)

    def process_task(
        self,
        task_id: str,
        source_path: Path,
        detector: YOLODetector,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> None:
        try:
            self._process_video(task_id, source_path, detector, conf, iou, imgsz)
        except Exception as exc:
            logger.exception("Video task %s failed", task_id)
            self._update_task(task_id, status="failed", message=str(exc), progress=100.0)

    def _process_video(
        self,
        task_id: str,
        source_path: Path,
        detector: YOLODetector,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> None:
        self._update_task(task_id, status="running", message="Opening video")
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError("OpenCV cannot open uploaded video")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Video has invalid width or height")

        raw_output = output_path("video_result_raw", ".mp4")
        writer = cv2.VideoWriter(
            str(raw_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps if source_fps > 0 else 25.0,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV cannot create output video writer")

        started = time.perf_counter()
        processed = 0
        no_vest_counts = []
        frame_details = []

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = detector.predict_frame(frame, conf, iou, imgsz)
                annotated = detector.draw_result(frame, prediction["detections"], prediction["stats"])
                writer.write(annotated)
                processed += 1
                no_vest = int(prediction["stats"].get("no_vest_count", 0))
                no_vest_counts.append(no_vest)
                frame_details.append({"frame": processed, "no_vest_count": no_vest})

                if processed % 5 == 0 or processed == 1:
                    progress = (processed / frame_count * 100.0) if frame_count else 0.0
                    self._update_task(
                        task_id,
                        progress=round(min(progress, 99.0), 2),
                        message=f"Processed {processed}/{frame_count or '?'} frames",
                        frame_count=processed,
                    )
        finally:
            capture.release()
            writer.release()

        elapsed = max(time.perf_counter() - started, 0.001)
        result_output = self._transcode_h264(raw_output)
        details_path = output_path("video_frame_violations", ".json")
        details_path.write_text(json.dumps(frame_details, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            "total_frames": frame_count,
            "processed_frames": processed,
            "source_fps": round(float(source_fps), 3),
            "average_processing_fps": round(processed / elapsed, 3),
            "max_no_vest_count": max(no_vest_counts) if no_vest_counts else 0,
            "avg_no_vest_count": round(sum(no_vest_counts) / len(no_vest_counts), 3) if no_vest_counts else 0.0,
            "violation_frame_count": sum(1 for value in no_vest_counts if value > 0),
            "frame_violation_json_url": url_for_output(details_path),
            "codec_note": "H.264 via ffmpeg" if result_output != raw_output else "mp4v fallback",
        }
        duration = processed / source_fps if source_fps > 0 else 0.0
        self._update_task(
            task_id,
            status="done",
            progress=100.0,
            message="Video processing complete",
            result_video_url=url_for_output(result_output),
            summary_stats=summary,
            frame_count=processed,
            duration=round(duration, 3),
            fps=round(float(source_fps), 3),
        )

    def _transcode_h264(self, raw_output: Path) -> Path:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return raw_output
        h264_output = raw_output.with_name(raw_output.stem.replace("_raw", "") + "_h264.mp4")
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(raw_output),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(h264_output),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            logger.warning("ffmpeg transcode failed, using mp4v output: %s", completed.stderr[-500:])
            return raw_output
        raw_output.unlink(missing_ok=True)
        return h264_output
