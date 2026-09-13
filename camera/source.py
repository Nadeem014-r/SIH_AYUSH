import logging
import os
import re
import socket
import urllib.parse

# Must be set BEFORE cv2 is imported so OpenCV's FFmpeg backend uses TCP/UDP and fast timeouts.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp;udp|timeout;3000000|stimeout;3000000|max_delay;500000",
)

import cv2

log = logging.getLogger("ibvap.camera")

# `rtsp://user:password@host/stream` is the normal way ONVIF/RTSP credentials
# are supplied (see CAMERA_SOURCES in config/settings.py), so anything that
# logs a source has to strip them first — the reconnect path logs repeatedly.
_URL_CREDENTIALS = re.compile(r"//[^/@\s]*:[^/@\s]*@")


def redact_source(source) -> str:
    """A log-safe rendering of a camera source, with any embedded
    `user:password@` credentials masked out."""
    return _URL_CREDENTIALS.sub("//***:***@", str(source))


def resolve_camera_source(src_str: str) -> str:
    """If user enters the dummy cellular IP shown on Android RTSP apps (e.g. 192.0.0.2),
    automatically resolve it to the reachable gateway or LAN device IP."""
    if not isinstance(src_str, str):
        return src_str
    try:
        parsed = urllib.parse.urlparse(src_str)
        host = parsed.hostname
        port = parsed.port or (554 if "rtsp" in parsed.scheme.lower() else 80)
        clean_path = parsed.path in ("", "/")
        if host and (host.startswith("192.0.0.") or host == "0.0.0.0"):
            import subprocess

            gw = None
            try:
                gw_out = subprocess.check_output("route print 0.0.0.0", text=True)
                m = re.search(r"0\.0\.0\.0\s+0\.0\.0\.0\s+([\d.]+)", gw_out)
                gw = m.group(1) if m else None
                if gw:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.6)
                        if s.connect_ex((gw, port)) == 0:
                            log.info("Auto-mapped mobile cellular IP %s -> %s on port %d", host, gw, port)
                            res = src_str.replace(host, gw)
                            return res.rstrip("/") if clean_path else res
            except Exception:
                pass

            try:
                arp_out = subprocess.check_output("arp -a", text=True)
                for ip in re.findall(r"(\d+\.\d+\.\d+\.\d+)", arp_out):
                    if not ip.endswith(".255") and not ip.startswith(("224.", "239.", "127.")):
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.3)
                            if s.connect_ex((ip, port)) == 0:
                                log.info("Auto-mapped mobile cellular IP %s -> %s on port %d", host, ip, port)
                                res = src_str.replace(host, ip)
                                return res.rstrip("/") if clean_path else res
            except Exception:
                pass

            if gw:
                res = src_str.replace(host, gw)
                return res.rstrip("/") if clean_path else res
        if clean_path:
            return src_str.rstrip("/")
    except Exception:
        pass
    return src_str


class CameraSource:
    """Wraps a single camera feed: a USB index (0, 1, ...), an RTSP/ONVIF URL,
    or a local video file path (useful for testing against recorded footage,
    e.g. vehicles, when no live feed is available).

    Every method is safe to call in any state: `open()` releases a previous
    capture before replacing it (so reconnecting can't leak handles), and
    `read()`/`native_fps()`/`rewind()` return a benign value rather than
    raising when there is no open capture. That lets the producer thread in
    `camera/stream_manager.py` drive open/read/release in a recovery loop
    without having to reason about half-open states.
    """

    def __init__(self, source: int | str, width: int = 640, height: int = 480):
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        self.source = source
        self.width = width
        self.height = height
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        # Reconnecting reuses this method, so drop any previous capture first
        # rather than orphaning an unreleased VideoCapture on every attempt.
        self.release()
        src = self.source
        if isinstance(src, str) and src.strip().isdigit():
            src = int(src.strip())
            self.source = src

        if isinstance(src, int):
            cap = cv2.VideoCapture(src)
        else:
            src_str = resolve_camera_source(str(src).strip())
            self.source = src_str
            if src_str.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
                try:
                    parsed = urllib.parse.urlparse(src_str)
                    host = parsed.hostname
                    port = parsed.port or (554 if "rtsp" in parsed.scheme.lower() else 80)
                    if host:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.6)
                            if s.connect_ex((host, port)) != 0:
                                raise RuntimeError(f"Network destination unreachable: {host}:{port}")
                except Exception as e:
                    if isinstance(e, RuntimeError):
                        raise

                candidates = [src_str]
                if src_str.endswith("/"):
                    candidates.insert(0, src_str.rstrip("/"))
                    candidates.append(src_str.rstrip("/") + "/live")
                elif not parsed.path or parsed.path == "/":
                    candidates.append(src_str + "/live")

                cap = None
                for candidate in candidates:
                    test_cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
                    if test_cap.isOpened():
                        cap = test_cap
                        break
                    test_cap.release()

                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(src_str)
            else:
                cap = cv2.VideoCapture(src_str)

        if not cap.isOpened():
            # A VideoCapture that failed to open still holds an object (and,
            # for network sources, possibly a socket) — release it explicitly.
            cap.release()
            raise RuntimeError(f"Could not open camera source: {redact_source(self.source)}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30.0)

        self.cap = cap
        log.info(
            "Camera opened: %s (requested %dx%d @ 30 FPS)", redact_source(self.source), self.width, self.height
        )

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def native_fps(self) -> float:
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def read(self):
        """Returns the next frame, or None if the capture is closed, gave no
        frame, or handed back an empty/invalid one. Never raises: a dropped
        RTSP stream can surface as an OpenCV exception rather than `ok=False`,
        and the caller's recovery path is the same either way."""
        cap = self.cap
        if cap is None:
            return None
        try:
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                return None

            # Resize on the measured size, not on is_file. cap.set(WIDTH/HEIGHT)
            # is honored only by local capture devices — it is a silent no-op
            # for video files AND for network streams, because an RTSP sender
            # picks its own resolution. Gating on is_file therefore misses
            # RTSP entirely: a phone pushing 1080p drove every downstream
            # stage at 6.75x the configured pixel budget (the temporal median
            # filter alone stacks 5 frames, ~31MB at 1080p). Comparing the
            # actual size covers all three source types and costs one tuple
            # compare when the camera already gave us what we asked for.
            h, w = frame.shape[:2]
            if (w, h) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            return frame
        except Exception as e:
            # Logged at debug only: the caller (CameraStream) reports the
            # failure once, at warning level, with backoff — logging it here
            # too would produce a message per dropped frame.
            log.debug("Read failed on camera %s: %s: %s", redact_source(self.source), type(e).__name__, e)
            return None

    def rewind(self) -> bool:
        """Seeks a video-file source back to its first frame (used to loop
        recorded footage). False if there is no capture or the backend
        refused the seek, in which case the caller should treat it as a
        broken source rather than retrying forever."""
        cap = self.cap
        if cap is None:
            return False
        try:
            return bool(cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
        except Exception as e:
            log.debug("Rewind failed on camera %s: %s: %s", redact_source(self.source), type(e).__name__, e)
            return False

    def release(self) -> None:
        # Clear the attribute first: another thread reading `self.cap`
        # concurrently gets None rather than a handle that is being released.
        cap, self.cap = self.cap, None
        if cap is None:
            return
        try:
            cap.release()
        except Exception as e:
            log.warning(
                "Error releasing camera %s: %s: %s", redact_source(self.source), type(e).__name__, e
            )
        else:
            log.info("Camera released: %s", redact_source(self.source))
