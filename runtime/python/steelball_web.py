#!/usr/bin/env python3
"""Steel-ball detection web service for an RDK board.

The service keeps one camera and one model instance open.  A background worker
updates the latest annotated frame and detection data, which are then shared by
the MJPEG stream, the web UI, and the capture endpoint.
"""

import argparse
import atexit
import base64
import json
import logging
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
SHARED_PROJECTS_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
DEFAULT_MODEL_PATH = "/userdata/rdkstudio/projects/ultralytics_yolo26/model/steelball-yolo26n-det_bayese_640x640_nv12.bin"

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SHARED_PROJECTS_ROOT not in sys.path:
    # Keep the shared utility import path consistent with runtime/python/main.py.
    sys.path.append(SHARED_PROJECTS_ROOT)

from yolo26_det import YOLO26Config, YOLO26Detect
from yolo26_seg import YOLO26Seg, YOLO26SegConfig


LOG = logging.getLogger("steelball_web")

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>钢球识别</title>
  <style>
    :root { color-scheme: dark; font-family: Arial, "Microsoft YaHei", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; background: #111827; color: #e5e7eb; }
    header { height: 64px; display: flex; align-items: center; justify-content: space-between; padding: 0 28px; background: #182235; border-bottom: 1px solid #334155; }
    h1 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: 0; }
    main { width: min(1400px, calc(100% - 40px)); margin: 28px auto; }
    .tabs { display: flex; gap: 8px; margin-bottom: 20px; }
    button { min-height: 38px; border: 1px solid #475569; border-radius: 5px; padding: 0 15px; background: #1e293b; color: #e5e7eb; font: inherit; cursor: pointer; }
    button:hover { background: #334155; }
    button.active, #capture { background: #047857; border-color: #10b981; color: #ecfdf5; }
    button:disabled { opacity: .55; cursor: wait; }
    .mode { display: none; }
    .mode.active { display: block; }
    .live-layout { display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 22px; align-items: start; }
    .capture-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 22px; }
    .video-panel { background: #0b1220; border: 1px solid #334155; border-radius: 6px; overflow: hidden; }
    .panel-title { display: flex; align-items: center; justify-content: space-between; height: 46px; padding: 0 14px; background: #182235; border-bottom: 1px solid #334155; font-size: 14px; }
    .stream-wrap { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #020617; }
    .stream { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #020617; }
    .stream-wrap .stream { height: 100%; aspect-ratio: auto; }
    .overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    .overlay rect { fill: none; stroke: #22c55e; stroke-width: 2; vector-effect: non-scaling-stroke; }
    .overlay circle { fill: #00bfff; }
    .overlay text { fill: #22c55e; font: 14px Arial, sans-serif; paint-order: stroke; stroke: #020617; stroke-width: 3px; stroke-linejoin: round; }
    .placeholder { display: flex; align-items: center; justify-content: center; width: 100%; aspect-ratio: 16 / 9; color: #94a3b8; background: #020617; }
    .data-panel { border: 1px solid #334155; border-radius: 6px; overflow: hidden; background: #182235; }
    .stat { padding: 18px; border-bottom: 1px solid #334155; }
    .stat-label { color: #94a3b8; font-size: 13px; }
    .count { margin-top: 7px; color: #34d399; font-size: 34px; font-weight: 700; }
    .coordinates { max-height: 440px; overflow: auto; }
    .coordinate { display: flex; justify-content: space-between; gap: 10px; padding: 12px 14px; border-bottom: 1px solid #293548; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
    .coordinate:last-child { border-bottom: 0; }
    .empty { padding: 18px 14px; color: #94a3b8; font-size: 14px; }
    .capture-actions { display: flex; align-items: center; gap: 14px; margin: 0 0 18px; }
    #capture-status { color: #94a3b8; font-size: 14px; }
    #capture-title-status { color: #fbbf24; }
    #capture-title-status.success { color: #34d399; font-weight: 600; }
    #capture-title-status.error { color: #fb7185; }
    .capture-results { margin-top: 22px; }
    #capture-results { display: none; }
    .capture-results .data-panel { width: min(680px, 100%); }
    @media (max-width: 820px) { header { padding: 0 16px; } main { width: min(100% - 24px, 1400px); margin-top: 16px; } .live-layout, .capture-layout { grid-template-columns: 1fr; } .coordinates { max-height: 240px; } }
  </style>
</head>
<body>
  <header><h1>钢球视觉识别</h1><span id="connection">正在连接</span></header>
  <main>
    <nav class="tabs" aria-label="功能选择">
      <button class="tab active" data-mode="live">实时识别</button>
      <button class="tab" data-mode="capture">拍摄识别</button>
    </nav>

    <section id="live" class="mode active">
      <div class="live-layout">
        <div class="video-panel">
          <div class="panel-title"><span>实时视频流</span><span id="live-fps"></span></div>
          <div class="stream-wrap">
            <img class="stream" src="/video_feed" alt="实时钢球检测视频流">
            <svg class="overlay" id="live-overlay" aria-hidden="true" preserveAspectRatio="xMidYMid meet"></svg>
          </div>
        </div>
        <div class="data-panel">
          <div class="stat"><div class="stat-label">当前钢球数量</div><div class="count" id="live-count">0</div></div>
          <div class="panel-title"><span>钢球中心点坐标（像素）</span></div>
          <div class="coordinates" id="live-coordinates"><div class="empty">等待检测结果</div></div>
        </div>
      </div>
    </section>

    <section id="capture-mode" class="mode">
      <div class="capture-actions"><button id="capture" type="button">拍摄</button><span id="capture-status">拍摄后将进行一次识别</span></div>
      <div class="capture-layout">
        <div class="video-panel">
          <div class="panel-title"><span>实时视频流</span><span id="capture-fps"></span></div>
          <img class="stream" src="/video_feed" alt="实时钢球检测视频流">
        </div>
        <div class="video-panel">
          <div class="panel-title"><span>拍摄照片与识别框</span><span id="capture-title-status"></span></div>
          <img class="stream" id="capture-image" alt="拍摄完成后显示识别结果">
          <div class="placeholder" id="capture-placeholder">尚未拍摄</div>
        </div>
      </div>
      <div class="capture-results" id="capture-results">
        <div class="data-panel">
          <div class="stat"><div class="stat-label">本次拍摄钢球数量</div><div class="count" id="capture-count">0</div></div>
          <div class="panel-title"><span>钢球中心点坐标（像素）</span></div>
          <div class="coordinates" id="capture-coordinates"><div class="empty">尚未拍摄</div></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const tabs = document.querySelectorAll('.tab');
    let activeMode = 'live';
    let capturePollTimer = null;

    async function setMode(mode) {
      const response = await fetch('/api/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      if (!response.ok) throw new Error('模式切换失败');
      activeMode = mode;
      tabs.forEach((item) => item.classList.toggle('active', item.dataset.mode === mode));
      document.getElementById('live').classList.toggle('active', mode === 'live');
      document.getElementById('capture-mode').classList.toggle('active', mode === 'capture');
      if (mode === 'live') refreshLiveResults();
    }

    tabs.forEach((tab) => tab.addEventListener('click', async () => {
      if (tab.dataset.mode === activeMode) return;
      try {
        await setMode(tab.dataset.mode);
      } catch (error) {
        document.getElementById('connection').textContent = error.message;
      }
    }));

    function coordinateRows(items, emptyText) {
      if (!items.length) return `<div class="empty">${emptyText}</div>`;
      return items.map((item, index) => `<div class="coordinate"><span>钢球 ${index + 1}</span><span>(${item.center_x}, ${item.center_y})</span></div>`).join('');
    }

    function drawLiveOverlay(detections, frameWidth, frameHeight) {
      const overlay = document.getElementById('live-overlay');
      if (!frameWidth || !frameHeight) {
        overlay.innerHTML = '';
        return;
      }
      overlay.setAttribute('viewBox', `0 0 ${frameWidth} ${frameHeight}`);
      overlay.innerHTML = detections.map((item) => {
        const [x1, y1, x2, y2] = item.box;
        const labelY = Math.max(y1 - 8, 18);
        return `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" />`
          + `<circle cx="${item.center_x}" cy="${item.center_y}" r="4" />`
          + `<text x="${x1}" y="${labelY}">ball ${item.id} ${item.score.toFixed(2)}</text>`;
      }).join('');
    }

    async function refreshLiveResults() {
      if (activeMode !== 'live') return;
      try {
        const response = await fetch('/api/results', { cache: 'no-store' });
        if (!response.ok) throw new Error('request failed');
        const data = await response.json();
        document.getElementById('live-count').textContent = data.count;
        document.getElementById('live-coordinates').innerHTML = coordinateRows(data.detections, '当前画面未识别到钢球');
        drawLiveOverlay(data.detections, data.frame_width, data.frame_height);
        document.getElementById('live-fps').textContent = data.fps ? `${data.fps.toFixed(1)} FPS` : '';
        document.getElementById('connection').textContent = '设备已连接';
      } catch (_) {
        document.getElementById('connection').textContent = '等待设备响应';
      }
    }

    async function refreshCaptureFps() {
      if (activeMode !== 'capture') return;
      try {
        const response = await fetch('/api/results', { cache: 'no-store' });
        if (!response.ok) throw new Error('request failed');
        const data = await response.json();
        document.getElementById('capture-fps').textContent = data.fps ? `${data.fps.toFixed(1)} FPS` : '';
        document.getElementById('connection').textContent = '设备已连接';
      } catch (_) {
        document.getElementById('connection').textContent = '等待设备响应';
      }
    }

    function setCaptureTitle(text, state) {
      const title = document.getElementById('capture-title-status');
      title.textContent = text;
      title.className = state || '';
    }

    function showCaptureImage(imageData) {
      const image = document.getElementById('capture-image');
      image.src = imageData;
      image.style.display = 'block';
      document.getElementById('capture-placeholder').style.display = 'none';
    }

    function showCaptureResult(data) {
      showCaptureImage(data.image);
      document.getElementById('capture-count').textContent = data.count;
      document.getElementById('capture-coordinates').innerHTML = coordinateRows(data.detections, '本次拍摄未识别到钢球');
      document.getElementById('capture-results').style.display = 'block';
    }

    async function pollCapture(captureId) {
      try {
        const response = await fetch('/api/capture/status', { cache: 'no-store' });
        const data = await response.json();
        if (!response.ok || data.capture_id !== captureId) return;
        if (data.state === 'recognizing') {
          capturePollTimer = window.setTimeout(() => pollCapture(captureId), 250);
          return;
        }
        document.getElementById('capture').disabled = false;
        if (data.state === 'success') {
          showCaptureResult(data);
          setCaptureTitle('识别成功', 'success');
          document.getElementById('capture-status').textContent = `识别完成：${new Date().toLocaleTimeString()}`;
          return;
        }
        setCaptureTitle('识别失败', 'error');
        document.getElementById('capture-status').textContent = `识别失败：${data.error || '未知错误'}`;
      } catch (_) {
        capturePollTimer = window.setTimeout(() => pollCapture(captureId), 500);
      }
    }

    document.getElementById('capture').addEventListener('click', async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      document.getElementById('capture-status').textContent = '正在获取图像...';
      try {
        const response = await fetch('/api/capture', { method: 'POST', cache: 'no-store' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'capture failed');
        showCaptureImage(data.image);
        document.getElementById('capture-results').style.display = 'none';
        setCaptureTitle('已获取图像，识别中', '');
        document.getElementById('capture-status').textContent = '正在进行识别...';
        if (capturePollTimer) window.clearTimeout(capturePollTimer);
        pollCapture(data.capture_id);
      } catch (error) {
        document.getElementById('capture-status').textContent = `拍摄失败：${error.message}`;
        setCaptureTitle('拍摄失败', 'error');
      } finally {
        if (document.getElementById('capture-title-status').textContent !== '已获取图像，识别中') button.disabled = false;
      }
    });

    refreshLiveResults();
    window.setInterval(refreshLiveResults, 100);
    window.setInterval(refreshCaptureFps, 350);
  </script>
</body>
</html>"""


class SteelBallService:
    """Own the camera, model, and the latest result shared by web requests."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.camera = self._open_camera(args.camera, args.width, args.height, args.camera_fps,
                                        args.camera_format, args.mjpeg_passthrough)
        self.model = self._load_model(args)
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.inference_lock = threading.Lock()
        self.running = True
        self.mode = "live"
        self.mjpeg_passthrough = args.mjpeg_passthrough and args.camera_format == "MJPG"
        self.mjpeg_passthrough_confirmed = False
        self.latest_jpeg: Optional[bytes] = None
        self.latest_raw_jpeg: Optional[bytes] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_detections: List[Dict[str, object]] = []
        self.latest_frame_width = 0
        self.latest_frame_height = 0
        self.latest_fps = 0.0
        self.timing_samples = 0
        self.timing_totals: Dict[str, float] = {}
        self.capture_id = 0
        self.capture_state = "idle"
        self.capture_jpeg: Optional[bytes] = None
        self.capture_detections: List[Dict[str, object]] = []
        self.capture_error = ""
        self.worker = threading.Thread(target=self._run, name="steelball-inference", daemon=True)
        self.worker.start()

    @staticmethod
    def _open_camera(camera: str, width: int, height: int, fps: int, pixel_format: str,
                     mjpeg_passthrough: bool) -> cv2.VideoCapture:
        source: object = int(camera) if camera.isdigit() else camera
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开摄像头：{camera}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if mjpeg_passthrough and pixel_format == "MJPG":
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        active_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        active_format = "".join(chr((active_fourcc >> (8 * index)) & 0xFF) for index in range(4))
        LOG.info("摄像头协商结果：%dx%d, %.1f FPS, 像素格式=%s（请求=%s），MJPG直通=%s",
                 int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 capture.get(cv2.CAP_PROP_FPS), active_format, pixel_format,
                 "已请求" if mjpeg_passthrough and pixel_format == "MJPG" else "未启用")
        return capture

    @staticmethod
    def _load_model(args: argparse.Namespace):
        if args.model_type == "seg":
            config = YOLO26SegConfig(
                model_path=args.model_path,
                classes_num=1,
                score_thres=args.score_thres,
                nms_thres=args.nms_thres,
                strides=np.array([8, 16, 32], dtype=np.int32),
            )
            model = YOLO26Seg(config)
        else:
            config = YOLO26Config(
                model_path=args.model_path,
                classes_num=1,
                score_thres=args.score_thres,
                nms_thres=args.nms_thres,
                strides=[8, 16, 32],
            )
            model = YOLO26Detect(config)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        return model

    def _infer(self, frame: np.ndarray, annotate: bool = True) -> Tuple[np.ndarray, List[Dict[str, object]], float, float]:
        inference_started = time.monotonic()
        with self.inference_lock:
            if self.args.model_type == "seg":
                boxes, scores, _, _ = self.model.predict(frame, return_masks=False)
            else:
                boxes, scores, _ = self.model.predict(frame)
        inference_ms = 1000 * (time.monotonic() - inference_started)

        render_started = time.monotonic()
        detections: List[Dict[str, object]] = []
        for index, (box, score) in enumerate(zip(boxes, scores), start=1):
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            center_x = int(round((x1 + x2) / 2))
            center_y = int(round((y1 + y2) / 2))
            detections.append({
                "id": index,
                "center_x": center_x,
                "center_y": center_y,
                "score": round(float(score), 3),
                "box": [x1, y1, x2, y2],
            })
        annotated = self._draw(frame, detections) if annotate else frame
        render_ms = 1000 * (time.monotonic() - render_started)
        return annotated, detections, inference_ms, render_ms

    def _record_timing(self, timings: Dict[str, float]) -> None:
        """Log rolling live-pipeline averages without per-frame I/O."""
        self.timing_samples += 1
        for name, value in timings.items():
            self.timing_totals[name] = self.timing_totals.get(name, 0.0) + value
        if self.timing_samples < self.args.timing_interval:
            return
        average = {name: value / self.timing_samples for name, value in self.timing_totals.items()}
        LOG.info(
            "最近 %d 帧平均耗时：采集 %.1f ms，MJPG解码 %.1f ms，模型流水线 %.1f ms，"
            "绘制 %.1f ms，JPEG编码 %.1f ms，总计 %.1f ms（%.1f FPS）",
            self.timing_samples,
            average["camera_read_ms"],
            average["decode_ms"],
            average["inference_ms"],
            average["render_ms"],
            average["encode_ms"],
            average["total_ms"],
            1000 / max(average["total_ms"], 1e-6),
        )
        self.timing_samples = 0
        self.timing_totals.clear()

    @staticmethod
    def _draw(frame: np.ndarray, detections: List[Dict[str, object]]) -> np.ndarray:
        result = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            cx, cy = detection["center_x"], detection["center_y"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cv2.circle(result, (cx, cy), 4, (0, 191, 255), -1)
            cv2.putText(result, f"ball {detection['id']} {detection['score']:.2f}", (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2, cv2.LINE_AA)
            cv2.putText(result, f"({cx}, {cy})", (x1, min(y2 + 20, result.shape[0] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 191, 255), 1, cv2.LINE_AA)
        cv2.putText(result, f"Steel balls: {len(detections)}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2, cv2.LINE_AA)
        return result

    @staticmethod
    def _encode(image: np.ndarray, quality: int = 85) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("无法编码视频帧")
        return encoded.tobytes()

    def _preview_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize only the browser preview; inference and capture keep full resolution."""
        if frame.shape[1] <= self.args.stream_width:
            return frame
        stream_height = round(frame.shape[0] * self.args.stream_width / frame.shape[1])
        return cv2.resize(frame, (self.args.stream_width, stream_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _as_mjpeg(frame: np.ndarray) -> Optional[bytes]:
        """Return a raw JPEG supplied by V4L2 when RGB conversion is disabled."""
        raw = frame.reshape(-1)
        if raw.size < 4 or raw[0] != 0xFF or raw[1] != 0xD8:
            return None
        return raw.tobytes()

    @staticmethod
    def _decode_mjpeg(jpeg: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("无法解码摄像头 MJPG 帧")
        return image

    def _run(self) -> None:
        previous_time = time.monotonic()
        while self.running:
            frame_started = time.monotonic()
            ok, camera_frame = self.camera.read()
            camera_read_ms = 1000 * (time.monotonic() - frame_started)
            if not ok:
                LOG.warning("摄像头读取失败，准备重试")
                time.sleep(0.1)
                continue
            now = time.monotonic()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now
            detections: List[Dict[str, object]] = []
            fps = 1.0 / elapsed
            timings = {
                "camera_read_ms": camera_read_ms,
                "decode_ms": 0.0,
                "inference_ms": 0.0,
                "render_ms": 0.0,
                "encode_ms": 0.0,
            }
            raw_jpeg: Optional[bytes] = None
            raw_frame: Optional[np.ndarray] = None
            if self.mjpeg_passthrough:
                raw_jpeg = self._as_mjpeg(camera_frame)
                if raw_jpeg is None:
                    self.mjpeg_passthrough = False
                    self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    LOG.warning("OpenCV 未返回原始 MJPG 帧，已回退到 BGR 解码和 JPEG 编码")
                    continue
                if not self.mjpeg_passthrough_confirmed:
                    LOG.info("已启用 MJPG 原始帧直通，拍摄页面不再进行逐帧图像编解码")
                    self.mjpeg_passthrough_confirmed = True

            with self.lock:
                mode = self.mode

            try:
                if mode == "live":
                    decode_started = time.monotonic()
                    frame = self._decode_mjpeg(raw_jpeg) if raw_jpeg is not None else camera_frame
                    timings["decode_ms"] = 1000 * (time.monotonic() - decode_started)
                    _, detections, timings["inference_ms"], timings["render_ms"] = self._infer(frame, annotate=False)
                    if raw_jpeg is not None:
                        jpeg = raw_jpeg
                    else:
                        encode_started = time.monotonic()
                        jpeg = self._encode(self._preview_frame(frame), self.args.stream_jpeg_quality)
                        timings["encode_ms"] = 1000 * (time.monotonic() - encode_started)
                    raw_frame = frame
                elif raw_jpeg is not None:
                    jpeg = raw_jpeg
                else:
                    raw_frame = camera_frame
                    jpeg = self._encode(self._preview_frame(camera_frame), self.args.stream_jpeg_quality)
            except Exception:
                LOG.exception("视频帧处理失败")
                time.sleep(0.1)
                continue
            timings["total_ms"] = 1000 * (time.monotonic() - frame_started)
            if mode == "live":
                self._record_timing(timings)
            with self.frame_ready:
                self.latest_jpeg = jpeg
                self.latest_raw_jpeg = raw_jpeg
                self.latest_raw_frame = raw_frame
                self.latest_fps = fps
                if mode == "live":
                    self.latest_detections = detections
                    self.latest_frame_height, self.latest_frame_width = frame.shape[:2]
                self.frame_ready.notify_all()

    def get_results(self) -> Dict[str, object]:
        with self.lock:
            detections = [dict(item) for item in self.latest_detections]
            return {"count": len(detections), "detections": detections, "fps": self.latest_fps,
                    "frame_width": self.latest_frame_width, "frame_height": self.latest_frame_height}

    def set_mode(self, mode: str) -> None:
        if mode not in ("live", "capture"):
            raise ValueError("不支持的页面模式")
        with self.lock:
            self.mode = mode

    def _recognize_capture(self, frame: np.ndarray | bytes, capture_id: int) -> None:
        try:
            if isinstance(frame, bytes):
                frame = self._decode_mjpeg(frame)
            annotated, detections, _, _ = self._infer(frame)
            jpeg = self._encode(annotated)
            with self.lock:
                if self.capture_id == capture_id:
                    self.capture_jpeg = jpeg
                    self.capture_detections = detections
                    self.capture_state = "success"
        except Exception as error:
            LOG.exception("拍摄图像识别失败")
            with self.lock:
                if self.capture_id == capture_id:
                    self.capture_state = "error"
                    self.capture_error = str(error)

    def capture_status(self) -> Dict[str, object]:
        with self.lock:
            result: Dict[str, object] = {
                "capture_id": self.capture_id,
                "state": self.capture_state,
                "error": self.capture_error,
            }
            if self.capture_state == "success" and self.capture_jpeg is not None:
                detections = [dict(item) for item in self.capture_detections]
                result.update({
                    "image": "data:image/jpeg;base64," + base64.b64encode(self.capture_jpeg).decode("ascii"),
                    "count": len(detections),
                    "detections": detections,
                })
            return result

    def frame_stream(self):
        last_frame: Optional[bytes] = None
        while self.running:
            with self.frame_ready:
                self.frame_ready.wait_for(lambda: self.latest_jpeg is not None and self.latest_jpeg != last_frame, timeout=2.0)
                frame = self.latest_jpeg
            if frame is None:
                continue
            last_frame = frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    def capture(self) -> Dict[str, object]:
        """Show a raw snapshot immediately, then recognize it in the background."""
        with self.lock:
            if self.mode != "capture":
                raise RuntimeError("请先切换到拍摄识别页面")
            if self.capture_state == "recognizing":
                raise RuntimeError("上一张照片仍在识别中")
            if self.latest_raw_frame is None and self.latest_raw_jpeg is None:
                raise RuntimeError("摄像头尚未产生可用画面")
            if self.latest_raw_jpeg is not None:
                frame: np.ndarray | bytes = self.latest_raw_jpeg
                image = base64.b64encode(frame).decode("ascii")
            else:
                frame = self.latest_raw_frame.copy()
                image = base64.b64encode(self._encode(frame)).decode("ascii")
            self.capture_id += 1
            capture_id = self.capture_id
            self.capture_state = "recognizing"
            self.capture_jpeg = None
            self.capture_detections = []
            self.capture_error = ""
        threading.Thread(target=self._recognize_capture, args=(frame, capture_id),
                         name="steelball-capture", daemon=True).start()
        return {
            "image": "data:image/jpeg;base64," + image,
            "capture_id": capture_id,
            "state": "recognizing",
        }

    def close(self) -> None:
        self.running = False
        with self.frame_ready:
            self.frame_ready.notify_all()
        if self.camera.isOpened():
            self.camera.release()


def create_server(service: SteelBallService, host: str, port: int) -> ThreadingHTTPServer:
    """Create a dependency-free HTTP server for the browser interface."""

    class SteelBallRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            LOG.info("%s - %s", self.client_address[0], format % args)

        def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: Dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", body)

        def _read_json(self) -> Dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            return payload

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", PAGE.encode("utf-8"))
                return
            if path == "/api/results":
                self._send_json(HTTPStatus.OK, service.get_results())
                return
            if path == "/api/capture/status":
                self._send_json(HTTPStatus.OK, service.capture_status())
                return
            if path != "/video_feed":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for frame in service.frame_stream():
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/mode":
                try:
                    mode = self._read_json().get("mode")
                    if not isinstance(mode, str):
                        raise ValueError("缺少模式参数")
                    service.set_mode(mode)
                    self._send_json(HTTPStatus.OK, {"mode": mode})
                except (ValueError, json.JSONDecodeError) as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if path != "/api/capture":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                self._send_json(HTTPStatus.OK, service.capture())
            except RuntimeError as error:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})

    class SteelBallHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    return SteelBallHTTPServer((host, port), SteelBallRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDK 钢球实时识别网页服务")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="钢球模型 .bin 文件路径")
    parser.add_argument("--model-type", choices=("seg", "detect"), default="detect", help="模型任务类型")
    parser.add_argument("--camera", default="0", help="摄像头索引或设备路径，例如 0 或 /dev/video0")
    parser.add_argument("--width", type=int, default=1280, help="摄像头请求宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头请求高度")
    parser.add_argument("--camera-fps", type=int, default=30, help="摄像头请求帧率")
    parser.add_argument("--camera-format", choices=("MJPG", "YUYV"), default="MJPG",
                        help="摄像头像素格式，默认 MJPG 以降低 USB 传输带宽")
    parser.add_argument("--mjpeg-passthrough", action=argparse.BooleanOptionalAction, default=True,
                        help="拍摄模式直接转发摄像头 MJPG，避免 OpenCV 解码后重新编码")
    parser.add_argument("--stream-width", type=int, default=640,
                        help="拍摄页面网页预览流宽度；不影响原图拍摄和模型输入")
    parser.add_argument("--stream-jpeg-quality", type=int, default=75, choices=range(1, 101),
                        help="拍摄页面网页预览流 JPEG 质量")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--score-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--nms-thres", type=float, default=0.65, help="NMS 阈值")
    parser.add_argument("--priority", type=int, default=0, help="BPU 调度优先级")
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0], help="使用的 BPU Core 编号")
    parser.add_argument("--timing-interval", type=int, default=60,
                        help="每隔多少实时识别帧输出一次服务端平均耗时")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
                        help="日志级别；使用 DEBUG 可查看各推理阶段耗时")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(name)s] %(levelname)s: %(message)s")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"未找到模型文件：{args.model_path}")
    service = SteelBallService(args)
    atexit.register(service.close)
    server = create_server(service, args.host, args.port)
    LOG.info("网页服务已启动，请在浏览器访问 http://<开发板IP>:%s", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("网页服务已停止")
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
