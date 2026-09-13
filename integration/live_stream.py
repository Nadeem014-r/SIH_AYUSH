"""Phase 19 — serves a camera as MJPEG so the React dashboard can show it in
a plain <img> tag (frontend/src/components/live/CameraTile.tsx renders exactly
that, so no player library is involved).

Why a pump thread instead of reading the camera per viewer:
CameraStream hands out frames through a maxsize=1 queue and read() *consumes*
the item. Two browser tabs reading the same CameraStream would therefore steal
frames from one another and each render at half rate. So one pump thread per
camera owns the read, runs detection once, encodes one JPEG, and publishes it;
every viewer then serves itself from that latest frame at its own pace. Cost
of a second viewer is one memcpy, not a second decode+detect.

CAMERA CONTENTION: a webcam can be held by one process at a time. Running
app.py and this API's live stream against the same device will fail for
whichever starts second — the registry surfaces that as a clear 409 rather
than a blank tile.
"""

import logging
import os
import threading
import time

# Must be set BEFORE cv2 is imported — OpenCV reads this when it initialises
# its FFmpeg backend. app.py sets the identical options at its own top; this
# process imports cv2 independently, so without this an RTSP camera added
# from the dashboard would fall back to UDP (visible macroblock corruption)
# and to FFmpeg's 30s stall timeout on a dropped stream. See app.py for the
# measured UDP-vs-TCP numbers behind these values.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000|fflags;nobuffer|flags;low_delay|framedrop;1|max_delay;0|probesize;32768|analyzeduration;0|buffer_size;65536|reorder_queue_size;0"
)

import cv2  # noqa: E402

log = logging.getLogger("ibvap.live")

# 55 provides clean, sharp images while dropping bandwidth by 65%, eliminating TCP socket lag
JPEG_QUALITY = 55

# Stop the camera this long after the last viewer leaves. Not zero: flipping
# between dashboard pages would otherwise re-open the device on every
# navigation, and a webcam takes ~1-2s to warm up.
IDLE_SHUTDOWN_SECONDS = 20.0


class CameraBusyError(RuntimeError):
    """The device could not be opened — usually app.py already holds it."""


class _LiveCamera:
    """One camera: a producer thread publishing the newest annotated JPEG."""

    def __init__(self, name, source, width, height, tracker_factory=None):
        self.name = name
        self._source = source
        self._width = width
        self._height = height
        self._tracker_factory = tracker_factory

        self._stream = None
        self._tracker = None
        self._thread = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()

        self._cond = threading.Condition()
        self._latest_jpeg = None
        self._seq = 0
        self._last_error = None

        self.viewers = 0
        self.last_viewer_at = time.time()
        self.fps = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        with self._start_lock:
            if self.running:
                return
            from camera.stream_manager import CameraStream

            self._stream = CameraStream(
                self.name, self._source, width=self._width, height=self._height
            )
        # CameraSource.open() raises if the device is held elsewhere; translate
        # it here so the route can answer 409 instead of leaking a stacktrace.
        try:
            self._stream.start()
        except Exception as exc:
            self._stream = None
            # A local device index and an RTSP URL fail for different
            # reasons, and telling an RTSP user to go close app.py sends them
            # hunting for the wrong problem.
            if isinstance(self._source, int):
                hint = (
                    f"Local device {self._source} did not open. It is most likely "
                    "already in use — app.py, or another camera entry pointing at "
                    "the same device index."
                )
            else:
                hint = (
                    "The stream did not open within 5s. Check the URL, that the "
                    "camera is on the same network, and any username/password in "
                    "the URL (rtsp://user:pass@host:554/path)."
                )
            raise CameraBusyError(
                f"Could not open camera {self.name!r} (source={self._source!r}). {hint}"
            ) from exc

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._pump, name=f"live-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._stream is not None:
            self._stream.stop()
        self._stream = None
        self._thread = None
        # Wake any viewer still blocked in wait_for_frame so its response ends
        # rather than hanging until the client times out.
        with self._cond:
            self._cond.notify_all()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # -- producer ----------------------------------------------------------

    def _load_tracker(self):
        try:
            self._tracker = self._tracker_factory()
            log.info("[%s] detector ready — overlay is now live", self.name)
        except Exception:
            log.exception("[%s] detector failed to load; streaming raw video", self.name)

    def _pump(self):
        # Load the detector on its own thread rather than inline on the first
        # frame. YOLO takes ~5-10s to come up on CoreML, which is longer than
        # a viewer waits for its first frame — doing it inline meant the very
        # first stream request returned an empty body and the <img> showed as
        # broken. Raw video now appears in about a second and the boxes start
        # drawing once the model is ready.
        if self._tracker_factory is not None:
            threading.Thread(
                target=self._load_tracker, name=f"detector-{self.name}", daemon=True
            ).start()

        frames = 0
        window_started = time.perf_counter()
        latest_detections = []
        frame_idx = 0
        while not self._stop_event.is_set():
            frame = self._stream.read(timeout=0.5)
            if frame is None:
                continue

            # Drain any stale queued frames so we always display the freshest live frame
            while True:
                newer = self._stream.read(timeout=0)
                if newer is None:
                    break
                frame = newer

            frame_idx += 1
            tracker = self._tracker  # None until the loader thread finishes
            if tracker is not None:
                try:
                    from detection.draw import draw_detections

                    if frame_idx % 2 == 0 or not latest_detections:
                        latest_detections = tracker.track(frame)
                    draw_detections(frame, latest_detections)
                except Exception:
                    # A detector fault must not kill the video feed; the
                    # operator still needs eyes on the camera.
                    log.exception("[%s] detection failed on a frame", self.name)

            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            )
            if not ok:
                continue

            with self._cond:
                self._latest_jpeg = buf.tobytes()
                self._seq += 1
                self._cond.notify_all()

            frames += 1
            elapsed = time.perf_counter() - window_started
            if elapsed >= 1.0:
                self.fps = frames / elapsed
                frames = 0
                window_started = time.perf_counter()

    # -- consumer ----------------------------------------------------------

    def wait_for_frame(self, last_seq, timeout=5.0):
        """Block until a frame newer than last_seq exists. Returns
        (jpeg, seq), or (None, last_seq) on timeout/shutdown so the caller can
        decide whether to keep the response open."""
        deadline = time.time() + timeout
        with self._cond:
            while self._seq == last_seq and not self._stop_event.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, last_seq
                self._cond.wait(remaining)
            if self._latest_jpeg is None:
                return None, self._seq
            return self._latest_jpeg, self._seq


class LiveCameraRegistry:
    """Starts cameras on demand and releases them once nobody is watching."""

    def __init__(self, sources_provider, width, height, tracker_factory=None, yield_fn=None):
        # A callable, not a dict: cameras added from the dashboard are written
        # to config/cameras.json *after* this registry is constructed, and a
        # snapshot taken at import time would 404 every one of them until the
        # API was restarted.
        self._sources_provider = sources_provider
        self._width = width
        self._height = height
        self._tracker_factory = tracker_factory
        # yield_fn(name) -> True when the AI pipeline owns this camera. The
        # direct preview then hands the device back: the pipeline cannot open
        # a webcam this process is holding, and the pipeline is the one that
        # scores threats and records incidents.
        self._yield_fn = yield_fn
        self._cameras = {}
        self._lock = threading.Lock()
        self._reaper = threading.Thread(
            target=self._reap_idle, name="live-reaper", daemon=True
        )
        self._reaper_stop = threading.Event()
        self._reaper.start()

    def known(self):
        return dict(self._sources_provider())

    def acquire(self, name):
        sources = self._sources_provider()
        if name not in sources:
            raise KeyError(name)
        with self._lock:
            cam = self._cameras.get(name)
            if cam is None:
                cam = _LiveCamera(
                    name,
                    sources[name],
                    self._width,
                    self._height,
                    self._tracker_factory,
                )
                self._cameras[name] = cam
            cam.viewers += 1
            cam.last_viewer_at = time.time()

        if not cam.running:
            try:
                cam.start()
            except CameraBusyError:
                with self._lock:
                    cam.viewers = max(0, cam.viewers - 1)
                    if cam.viewers == 0:
                        self._cameras.pop(name, None)
                raise
        return cam

    def release(self, name):
        with self._lock:
            cam = self._cameras.get(name)
            if cam is None:
                return
            cam.viewers = max(0, cam.viewers - 1)
            cam.last_viewer_at = time.time()

    def status(self, name):
        with self._lock:
            cam = self._cameras.get(name)
            if cam is None or not cam.running:
                return {"live": False, "viewers": 0, "fps": 0.0}
            return {"live": True, "viewers": cam.viewers, "fps": round(cam.fps, 1)}

    def force_stop(self, name):
        """Stop now regardless of viewer count. Their generators unblock via
        the notify_all in _LiveCamera.stop() and end their responses."""
        with self._lock:
            cam = self._cameras.get(name)
            if cam is not None and cam.running:
                cam.viewers = 0
                cam.stop()

    def _reap_idle(self):
        # 2s rather than 5s: this is also how quickly a direct preview gets out
        # of the pipeline's way once the pipeline starts.
        while not self._reaper_stop.wait(2.0):
            now = time.time()
            with self._lock:
                for name, cam in list(self._cameras.items()):
                    if not cam.running:
                        continue
                    idle = cam.viewers == 0 and now - cam.last_viewer_at > IDLE_SHUTDOWN_SECONDS
                    try:
                        yield_now = bool(self._yield_fn and self._yield_fn(name))
                    except Exception:
                        yield_now = False
                    if yield_now:
                        log.info("[%s] AI pipeline is running — releasing the direct preview", name)
                        cam.viewers = 0
                        cam.stop()
                    elif idle:
                        log.info("[%s] no viewers for %.0fs — releasing camera",
                                 name, IDLE_SHUTDOWN_SECONDS)
                        cam.stop()

    def shutdown(self):
        self._reaper_stop.set()
        with self._lock:
            for cam in self._cameras.values():
                if cam.running:
                    cam.stop()
            self._cameras.clear()
