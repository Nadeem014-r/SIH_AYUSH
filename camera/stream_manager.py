import logging
import queue
import threading
import time

from camera.health import CameraHealth
from camera.source import CameraSource

log = logging.getLogger("ibvap.stream_manager")


class CameraStream:
    """Runs one camera in its own producer thread, feeding a bounded queue.

    Only the newest frame is kept: if the consumer falls behind, the oldest
    queued frame is dropped rather than letting a backlog build up.

    A failing camera does not end the thread. When reads stop working the
    capture is released and reopened with exponential backoff, and the
    stream's `health` moves ONLINE -> RECONNECTING -> ONLINE, or ->
    OFFLINE once `max_reconnect_attempts` is exhausted, so the rest of the
    application can tell a live feed from a dead one. The queue is drained on
    entering RECONNECTING so a frame captured before the outage is never
    handed out as if it were a live view of the scene.

    Backoff waits on the stop event rather than sleeping, so `stop()` during a
    reconnect returns promptly instead of waiting out the delay.
    """

    def __init__(
        self,
        name: str,
        source: int | str,
        width: int = 640,
        height: int = 480,
        *,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        max_reconnect_attempts: int = 0,  # 0 = keep retrying for as long as the app runs
        max_read_failures: int = 50,
        reconnect_log_every: int = 10,
        stop_timeout: float = 2.0,
        camera_factory=CameraSource,
    ):
        self.name = name
        self.reconnect_initial_delay = reconnect_initial_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.max_read_failures = max_read_failures
        self.reconnect_log_every = reconnect_log_every
        self.stop_timeout = stop_timeout
        self.reconnect_count = 0
        self._camera = camera_factory(source, width=width, height=height)
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._health = CameraHealth.OFFLINE

    @property
    def health(self) -> str:
        with self._state_lock:
            return self._health

    def start(self) -> None:
        """Opens the camera and starts its producer thread.

        A camera that cannot be opened right now does *not* raise: it starts
        in RECONNECTING and its thread retries in the background, so one dead
        camera can never stop the others from coming up.
        """
        self._stop_event.clear()
        try:
            self._camera.open()
        except Exception as exc:
            self._set_health(
                CameraHealth.RECONNECTING, f"initial open failed: {type(exc).__name__}"
            )
            log.error(
                "[%s] initial camera open failed (%s: %s) - retrying in the background",
                self.name, type(exc).__name__, exc,
            )
        else:
            self._set_health(CameraHealth.ONLINE, "opened")
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self.name}", daemon=True
        )
        self._thread.start()

    # A live RTSP stream commonly fails its first few reads while the H.264
    # decoder is still resolving SPS/PPS and waiting for a clean keyframe —
    # that is transient noise, not the stream ending. Only give up after
    # this many *consecutive* failures (~2.5s at the retry sleep below),
    # which still detects a genuinely dead/disconnected camera quickly.
    MAX_CONSECUTIVE_FAILURES = 50
    RETRY_SLEEP_SECONDS = 0.05

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self._camera.is_open():
                    if not self._reconnect():
                        return  # stopped, or the reconnect budget ran out
                    continue
                self._capture_until_failure()
        except Exception:
            # The producer must never die quietly leaving the camera looking
            # healthy - anything unexpected here is reported and marked OFFLINE.
            self._set_health(CameraHealth.OFFLINE, "producer thread aborted")
            log.exception("[%s] camera producer thread aborted", self.name)
        finally:
            self._drain_queue()

    def _capture_until_failure(self) -> None:
        """Publishes frames until the source breaks or a stop is requested.

        On a broken source it puts the stream into RECONNECTING and returns,
        leaving the reconnect itself to `_run`'s loop.
        """
        frame_interval = 1.0 / self._camera.native_fps() if self._camera.is_file else 0.0
        next_frame_at = time.perf_counter()
        failures = 0

        while not self._stop_event.is_set():
            frame = self._camera.read()
            if frame is None:
                failures += 1
                # First miss on a file is just the end of the footage - loop it.
                if self._camera.is_file and failures == 1 and self._camera.rewind():
                    log.info("[%s] video file ended, looping back to start", self.name)
                    continue
                # A live camera drops the odd frame; only a run of them means
                # the connection itself is gone. Bounded, so a source that
                # returns nothing can't spin here indefinitely either.
                if failures >= self.max_read_failures:
                    self._enter_reconnecting(f"{failures} consecutive read failures")
                    return
                time.sleep(self.RETRY_SLEEP_SECONDS)
                continue
            failures = 0

            if self._camera.is_file:
                # Recorded files have no natural playback pace like a live
                # camera does, so throttle to the file's own FPS. Waiting on
                # the stop event keeps shutdown responsive during the throttle.
                now = time.perf_counter()
                sleep_for = next_frame_at - now
                if sleep_for > 0 and self._stop_event.wait(sleep_for):
                    return
                next_frame_at = max(now, next_frame_at) + frame_interval

            self._publish(frame)

    def _publish(self, frame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # The consumer refilled the slot in between; dropping this frame is
            # exactly the newest-frame-only policy, and a non-blocking put means
            # the producer can never hang here during shutdown.
            pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _enter_reconnecting(self, reason: str) -> None:
        self._set_health(CameraHealth.RECONNECTING, reason)
        log.warning(
            "[%s] capture lost (%s) - releasing capture and reconnecting", self.name, reason
        )
        self._camera.release()
        # Anything still queued predates the outage; serving it would present a
        # stale scene as a healthy live feed.
        self._drain_queue()

    def _reconnect(self) -> bool:
        """Reopens the camera with bounded exponential backoff.

        True once reconnected; False if the stream was stopped or the attempt
        budget ran out (health is then left OFFLINE).
        """
        if self.health != CameraHealth.RECONNECTING:
            self._set_health(CameraHealth.RECONNECTING, "capture not open")
        delay = self.reconnect_initial_delay
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            if self.max_reconnect_attempts and attempt > self.max_reconnect_attempts:
                self._set_health(
                    CameraHealth.OFFLINE, f"gave up after {self.max_reconnect_attempts} attempts"
                )
                log.error(
                    "[%s] camera OFFLINE - gave up after %d reconnect attempts",
                    self.name, self.max_reconnect_attempts,
                )
                return False

            # Interruptible backoff: a stop during the wait returns immediately
            # rather than sleeping out the full delay.
            if self._stop_event.wait(delay):
                return False

            try:
                self._camera.open()
            except Exception as exc:
                self._log_reconnect_failure(attempt, exc, delay)
                delay = min(delay * 2.0, self.reconnect_max_delay)
                continue

            if self._stop_event.is_set():
                # Stopped while the (blocking) open was in flight - don't leave
                # a freshly opened capture behind with nobody to release it.
                self._camera.release()
                return False

            self.reconnect_count += 1
            self._drain_queue()
            self._set_health(CameraHealth.ONLINE, f"reconnected after {attempt} attempt(s)")
            log.info("[%s] camera reconnected after %d attempt(s)", self.name, attempt)
            return True

        return False

    def _log_reconnect_failure(self, attempt: int, exc: Exception, delay: float) -> None:
        # A source that is down for hours would otherwise print the same line
        # forever: report the first attempt and then every Nth, debug in between.
        message = "[%s] reconnect attempt %d failed (%s: %s) - retrying in %.1fs"
        args = (self.name, attempt, type(exc).__name__, exc, delay)
        if attempt == 1 or attempt % self.reconnect_log_every == 0:
            log.warning(message, *args)
        else:
            log.debug(message, *args)

    def _set_health(self, new_health: str, reason: str = "") -> str:
        with self._state_lock:
            previous, self._health = self._health, new_health
        # Logged outside the lock, and only on an actual transition, so a
        # camera sitting in one state doesn't repeat itself.
        if previous != new_health:
            log.info(
                "[%s] camera health %s -> %s%s",
                self.name, previous.upper(), new_health.upper(),
                f" ({reason})" if reason else "",
            )
        return previous

    def read(self, timeout: float = 0.0):
        try:
            if timeout <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.stop_timeout)
            if thread.is_alive():
                # Blocked inside an OpenCV open/read we cannot interrupt. Do
                # NOT release the capture here: the producer may still be
                # reading the same cv2.VideoCapture, and releasing it out from
                # under that read is a genuine race — observed on a glitchy
                # RTSP source as a native mutex abort, which kills the whole
                # process rather than raising. The thread is a daemon, so the
                # OS reclaims the handle at exit.
                log.warning(
                    "[%s] producer thread still running after %.1fs - likely blocked in "
                    "an OpenCV call; skipping camera release to avoid a "
                    "use-after-release race. It is a daemon thread and will not "
                    "outlive the process.",
                    self.name, self.stop_timeout,
                )
                self._drain_queue()
                self._set_health(CameraHealth.OFFLINE, "stopped (capture left open)")
                return
        self._camera.release()
        self._drain_queue()
        self._set_health(CameraHealth.OFFLINE, "stopped")


class StreamManager:
    def __init__(
        self,
        sources: dict[str, int | str],
        width: int = 640,
        height: int = 480,
        **stream_kwargs,
    ):
        self.streams = {
            name: CameraStream(name, src, width=width, height=height, **stream_kwargs)
            for name, src in sources.items()
        }

    def start_all(self) -> None:
        def _start_one(name: str, stream: CameraStream) -> None:
            try:
                stream.start()
            except Exception:
                # start() already tolerates an unopenable camera; this guards
                # against anything else so one camera can't stop the others.
                log.exception("[%s] failed to start camera stream - continuing without it", name)

        threads = [
            threading.Thread(target=_start_one, args=(name, stream), daemon=True)
            for name, stream in self.streams.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log.info(
            "Started %d camera stream(s): %s - health: %s",
            len(self.streams), list(self.streams), self.health(),
        )

    def read_all(self) -> dict[str, object]:
        return {name: stream.read(timeout=0.0) for name, stream in self.streams.items()}

    def health(self) -> dict[str, str]:
        """Per-camera ONLINE / RECONNECTING / OFFLINE state (see CameraHealth)."""
        return {name: stream.health for name, stream in self.streams.items()}

    def stop_all(self) -> None:
        for name, stream in self.streams.items():
            try:
                stream.stop()
            except Exception:
                log.exception("[%s] error while stopping camera stream", name)
        log.info("Stopped all camera streams")
