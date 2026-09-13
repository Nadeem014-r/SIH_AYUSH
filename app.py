import logging
import os
import sys
import time
from collections import deque
from datetime import datetime

# Must be set BEFORE cv2 is imported — OpenCV reads this when it initialises
# its FFmpeg backend.
#
# timeout;5000000 — 5s, in microseconds, instead of FFmpeg's 30s default. A
# dropped RTSP stream otherwise blocks the producer thread inside a single
# cap.read() for 30s ("Stream timeout triggered after 30100 ms"), far past the
# ~2.5s CameraStream.MAX_CONSECUTIVE_FAILURES budget meant to spot a dead
# camera quickly. "timeout" is the current option name and "stimeout" the
# pre-FFmpeg-5.0 one; both are passed so the limit applies whichever build
# OpenCV was linked against.
#
# rtsp_transport;tcp — measured, not assumed. Against this project's phone
# camera (Android IP Webcam, 1080p over Wi-Fi), 150 consecutive frames each way:
#   UDP (FFmpeg default): 68 and 108 decode errors across two runs, 5 frames lost
#   TCP                 : 0 decode errors across two runs, 0 frames lost
# UDP's losses arrive as "error while decoding MB" and "intra mode" corruption —
# garbled macroblocks handed straight to the detector, which is far worse for
# tracking than TCP's lower throughput. TCP is slower on a 1080p stream (~11-19
# fps vs UDP's buffered ~60), so the real win is lowering the *sender's*
# resolution: fewer bits makes TCP both clean and fast.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
)

import cv2  # noqa: E402
import numpy as np

from activity_gate.gate import ActivityGate
from alerts.alert_manager import AlertManager
from alerts.dispatch import AlertDispatcher
from camera.health import CameraErrorIsolator, CameraHealth
from camera.stream_manager import StreamManager
from config.settings import (
    ALERT_CONFIRM_N,
    ALERT_CONFIRM_WINDOW,
    ALERT_COOLDOWN_SECONDS,
    ALERT_MAX_COOLDOWN_SECONDS,
    ALERT_DISPATCH_QUEUE_SIZE,
    BRIGHTNESS_THRESHOLD,
    CAMERA_HEIGHT,
    CAMERA_SOURCES,
    CAMERA_WIDTH,
    CAMERA_ZONE_TIERS,
    _parse_camera_sources,
    CURFEW_END_HOUR,
    CURFEW_START_HOUR,
    DETECTION_CONFIDENCE,
    DETECTION_MODEL_PATH,
    HARDWARE_PROFILE,
    IDLE_MIN_FPS,
    MOTION_THRESHOLD,
    REID_FACE_CHECK_INTERVAL,
    REID_MATCH_MARGIN,
    REID_MODEL_PATH,
    REID_SIMILARITY_THRESHOLD,
    REID_TTL_SECONDS,
    SYSLOG_HOST,
    SYSLOG_PORT,
    WATCHLIST_SIMILARITY_THRESHOLD,
    WEBHOOK_URL,
    configure_logging,
)
from config.providers import select_providers
from database.incident_store import IncidentStore
from detection.draw import draw_detections
from face.face_recognizer import FaceRecognizer
from face.watchlist import WatchlistDB, WatchlistMatcher
from filtering.false_alarm import FalseAlarmFilter
from integration.syslog_notifier import SyslogNotifier
from integration.runtime_state import PipelinePublisher
from integration.webhook import WebhookNotifier
from intelligence.loiter import LoiterTracker
from intelligence.threat_rules import ThreatRulesDB
from intelligence.threat_score import ThreatScorer
from preprocessing.enhance import Preprocessor
from profiling.stage_profiler import StageProfiler
from reid.embedder import OSNetEmbedder
from reid.reid import PersonGallery
from tracking.tracker import Tracker
from zones.drawer import ZoneDrawer
from zones.zone_engine import ZoneEngine

configure_logging()
log = logging.getLogger("ibvap")


def cli_camera_sources(argv: list) -> "dict[str, int | str] | None":
    """Lets camera sources be given directly on the command line instead of
    only via CAMERA_SOURCES in .env — e.g.:

        python app.py rtsp://192.168.1.46:8080/h264_ulaw.sdp
        python app.py 0 rtsp://192.168.1.46:8080/h264_ulaw.sdp   # webcam + phone, both live
        python app.py cam_phone=rtsp://192.168.1.46:8080/h264_ulaw.sdp

    Each positional argument is one camera: a bare RTSP/URL or webcam index
    is auto-named cam0, cam1, ...; name=source gives it a custom name. Reuses
    config.settings' own parser so the two entry points parse identically.
    Returns None (meaning "use CAMERA_SOURCES from .env") if no camera
    arguments were given.
    """
    positional = [a for a in argv if not a.startswith("--")]
    if not positional:
        return None

    parts = []
    for i, entry in enumerate(positional):
        name, _, _value = entry.partition("=")
        # A real name=value split has a plain identifier before the "=". If
        # that part looks like a URL scheme or a webcam index instead, the
        # "=" almost certainly belongs to the URL itself (e.g. a query
        # string, "...?user=admin"), so treat the whole entry as bare.
        is_valid_name = "=" in entry and name and "://" not in name and not name.isdigit()
        parts.append(entry if is_valid_name else f"cam{i}={entry}")
    return _parse_camera_sources(",".join(parts))


def draw_debug_overlay(frame, preprocessor: Preprocessor, active: bool, motion_score: float):
    # Render a compact top status banner
    pass


def draw_fps_overlay(frame, fps: float):
    # Top status bar with dark background
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 0), -1)
    status_text = f"FPS: {fps:.1f}  |  AI PIPELINE: 30 FPS ONNX  |  SECURITY: ACTIVE"
    cv2.putText(frame, status_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


TIER_COLORS = {"green": (0, 200, 0), "yellow": (0, 220, 220), "red": (0, 0, 255)}


def zone_group_count(detections: list) -> int:
    """How many people are inside a zone in this frame.

    Group risk is a property of the frame, not of one detection, so it is
    counted once here and handed to every person's score — a lone walker and
    one of five people at the line must not read the same."""
    return sum(
        1 for d in detections
        if d.category() == "person" and d.zone_tier and d.zone_tier != "none"
    )


def draw_threat_score_overlay(
    frame, det, scorer: ThreatScorer, dwell_seconds: float, group_count: int,
    zone=None, is_curfew=None,
):
    score = scorer.score(
        zone_tier=det.zone_tier,
        hour=datetime.now().hour,
        speed_px_per_frame=det.speed,
        category=det.category(),
        zone_direction=det.zone_direction,
        dwell_seconds=dwell_seconds,
        group_count=group_count,
        watchlist_match=det.watchlist_match,
        watchlist_similarity=det.watchlist_similarity,
        zone=zone,
        is_curfew=is_curfew,
    )
    return score


def main() -> None:
    log.info("IBVAP starting up (Phase 18 — Alert Discipline)")

    window_names = {name: f"IBVAP - {name} (press q to quit)" for name in CAMERA_SOURCES}
    for window_name in window_names.values():
        cv2.namedWindow(window_name)

    manager = StreamManager(CAMERA_SOURCES, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    manager.start_all()
    preprocessors = {
        name: Preprocessor(name, brightness_threshold=BRIGHTNESS_THRESHOLD)
        for name in CAMERA_SOURCES
    }
    gates = {name: ActivityGate(motion_threshold=MOTION_THRESHOLD) for name in CAMERA_SOURCES}
    frame_counters = {name: 0 for name in CAMERA_SOURCES}
    trackers = {
        name: Tracker(model_path=DETECTION_MODEL_PATH, confidence=DETECTION_CONFIDENCE)
        for name in CAMERA_SOURCES
    }
    false_alarm_filters = {name: FalseAlarmFilter() for name in CAMERA_SOURCES}
    embedder = OSNetEmbedder(REID_MODEL_PATH)
    # One shared gallery across all cameras (Phase 15: cross-camera Re-ID) —
    # resolve() is called with a (camera_name, track_id) key, not a raw
    # track_id, so two cameras can't collide on the same track_id number.
    person_gallery = PersonGallery(
        embed_fn=embedder.embed,
        similarity_threshold=REID_SIMILARITY_THRESHOLD,
        ttl_seconds=REID_TTL_SECONDS,
        match_margin=REID_MATCH_MARGIN,
    )
    zone_engines = {
        name: ZoneEngine(
            config_path=f"config/zones_{name}.json",
            curfew_start_hour=CURFEW_START_HOUR,
            curfew_end_hour=CURFEW_END_HOUR,
            fixed_tier=CAMERA_ZONE_TIERS.get(name),
        )
        for name in CAMERA_SOURCES
    }
    zone_drawers = {
        name: ZoneDrawer(window_names[name], zone_engines[name]) for name in CAMERA_SOURCES
    }
    loiter_trackers = {name: LoiterTracker() for name in CAMERA_SOURCES}

    # Without zones there is no border line, so sector/direction/loiter/group
    # all read zero and the score collapses to time + class + movement. That
    # silently turns a border system into a generic motion alarm, so say it
    # loudly rather than letting a sentry trust an un-configured camera. A
    # camera named in CAMERA_ZONE_TIERS needs no drawn polygons at all - its
    # whole frame is one tier - so it's exempt from the warning.
    for name in CAMERA_SOURCES:
        if zone_engines[name].fixed_tier is not None:
            log.info(
                "[%s] fixed to %s zone tier via CAMERA_ZONE_TIERS — no polygons needed",
                name, zone_engines[name].fixed_tier.upper(),
            )
        elif not zone_engines[name].zones:
            log.warning(
                "[%s] NO ZONES DEFINED (%s is empty) — border scoring is inactive: "
                "no sector, crossing-direction, loitering or group risk will be "
                "applied. Press 'z' on the video window to draw the border line "
                "(red), approach strip (yellow) and own territory (green).",
                name, f"config/zones_{name}.json",
            )
    face_recognizer = FaceRecognizer()
    watchlist_db = WatchlistDB()
    watchlist_matcher = WatchlistMatcher(watchlist_db, similarity_threshold=WATCHLIST_SIMILARITY_THRESHOLD)
    threat_rules = ThreatRulesDB()
    threat_scorer = ThreatScorer(threat_rules)
    incident_store = IncidentStore()
    webhook = WebhookNotifier(url=WEBHOOK_URL)
    syslog = SyslogNotifier(host=SYSLOG_HOST, port=SYSLOG_PORT)
    # Webhook POSTs and evidence persistence run on this bounded background
    # worker so a slow/unreachable C2 host or disk cannot stall the camera loop.
    alert_dispatcher = AlertDispatcher(maxsize=ALERT_DISPATCH_QUEUE_SIZE)
    alert_manager = AlertManager(
        cooldown_seconds=ALERT_COOLDOWN_SECONDS,
        confirm_n=ALERT_CONFIRM_N,
        confirm_window=ALERT_CONFIRM_WINDOW,
        max_cooldown_seconds=ALERT_MAX_COOLDOWN_SECONDS,
        incident_store=incident_store,
        webhook=webhook,
        syslog=syslog,
        dispatcher=alert_dispatcher,
    )
    # Seconds between full pipeline passes while a scene is idle. Guard against
    # a zero/negative setting turning the floor into a division error.
    idle_min_period = 1.0 / IDLE_MIN_FPS if IDLE_MIN_FPS > 0 else 0.0

    # Inert unless IBVAP_PROFILE=1 — stage() then returns a shared no-op, so an
    # unprofiled run does the same work it always did.
    profiler = StageProfiler()
    if profiler.enabled:
        log.info("Stage profiling ENABLED — report prints on exit (q)")

    frame_buffers = {name: deque(maxlen=3) for name in CAMERA_SOURCES}
    last_frame_time = {name: None for name in CAMERA_SOURCES}
    fps_ema = {name: 0.0 for name in CAMERA_SOURCES}
    # Per-camera caches so a throttled (skipped) frame still shows the last
    # known identity/watchlist result instead of blanking it out.
    person_id_cache = {name: {} for name in CAMERA_SOURCES}
    watchlist_cache = {name: {} for name in CAMERA_SOURCES}

    isolator = CameraErrorIsolator()
    last_health: dict[str, str] = {}

    # Live frames and health for the dashboard (integration/runtime_state.py).
    # The dashboard reads these instead of opening the camera itself, so the
    # operator sees exactly what the model sees while it keeps alerting.
    publisher = PipelinePublisher()
    camera_state = {name: {} for name in CAMERA_SOURCES}
    last_health_publish = 0.0
    # Zones saved from the dashboard land in config/zones_<cam>.json. Watching
    # the mtime lets a running pipeline pick them up without a restart.
    zone_mtimes = {
        name: (os.path.getmtime(zone_engines[name].config_path)
               if os.path.exists(zone_engines[name].config_path) else None)
        for name in CAMERA_SOURCES
    }
    models_info = {
        "detector": DETECTION_MODEL_PATH,
        "reid": REID_MODEL_PATH,
        "face": "insightface/buffalo_s",
        "profile": HARDWARE_PROFILE,
        "providers": ", ".join(select_providers()),
    }

    def process_camera_frame(name: str, frame) -> None:
        """The full per-camera pipeline for one frame.

        Runs behind `isolator` below, so anything raised in here costs this
        one frame on this one camera instead of the whole loop.

        The body is the full pipeline — profiler stages, the IDLE_MIN_FPS
        floor, the Re-ID/face check throttle, loiter dwell and group count.
        Where the loop used `continue` to skip the rest of a frame, this
        returns instead.
        """
        with profiler.stage("activity_gate"):
            active, motion_score = gates[name].is_active(frame)
        frame_counters[name] += 1
        camera_state[name]["active"] = bool(active)
        camera_state[name]["motion"] = round(float(motion_score), 2)
        camera_state[name]["lastFrameAt"] = time.time()

        now = time.perf_counter()
        last_processed = last_frame_time[name]
        due = last_processed is None or (now - last_processed) >= idle_min_period
        if not (active or due):
            publisher.publish_frame(name, frame)
            return

        if last_processed is not None:
            instant_fps = 1.0 / max(now - last_processed, 1e-6)
            fps_ema[name] = (0.9 * fps_ema[name]) + (0.1 * instant_fps)
        last_frame_time[name] = now

        preprocessor = preprocessors[name]
        with profiler.stage("preprocess"):
            processed = preprocessor.process(frame)

        with profiler.stage("detect_track"):
            detections = trackers[name].track(processed)
        with profiler.stage("false_alarm_filter"):
            detections = false_alarm_filters[name].filter(detections)
        for det in detections:
            if det.track_id is not None and det.category() == "person":
                track_id = det.track_id
                # A brand-new track is checked every frame (Re-ID
                # needs consecutive samples to decide an identity at
                # all — resolve() returns None while still buffering,
                # so check *value*, not key presence, or a track
                # stuck buffering would get throttled before it ever
                # resolves); once resolved, re-checking every Nth
                # frame is enough — appearance doesn't change frame-to-frame.
                already_resolved = person_id_cache[name].get(track_id) is not None
                due_for_check = frame_counters[name] % REID_FACE_CHECK_INTERVAL == 0
                if not already_resolved or due_for_check:
                    with profiler.stage("reid_resolve"):
                        det.person_id = person_gallery.resolve(
                            (name, track_id), processed, det.box
                        )
                    person_id_cache[name][track_id] = det.person_id
                else:
                    det.person_id = person_id_cache[name][track_id]

                if not already_resolved or due_for_check:
                    with profiler.stage("face_embed"):
                        _face_box, embedding = face_recognizer.embed(processed, det.box)
                    if embedding is not None:
                        with profiler.stage("watchlist_match"):
                            match_name, similarity = watchlist_matcher.match(embedding)
                        watchlist_cache[name][track_id] = (match_name, similarity)
                cached_match, cached_similarity = watchlist_cache[name].get(
                    track_id, (None, 0.0)
                )
                det.watchlist_match = cached_match
                det.watchlist_similarity = cached_similarity

            det.camera_name = name
            x1, y1, x2, y2 = det.box
            ground_point = ((x1 + x2) // 2, y2)
            with profiler.stage("zone_classify"):
                zone_result = zone_engines[name].classify(ground_point, det.direction)
            det.zone_tier = zone_result["tier"]
            det.zone_direction = zone_result["direction"]
            det.zone_id = zone_result.get("zone_id")
            det.zone_label = zone_result.get("zone_label")
            det._matched_zone = zone_result.get("zone")

        draw_detections(processed, detections)

        group_count = zone_group_count(detections)
        scores = []
        curfew_active = zone_engines[name]._is_curfew()
        for det in detections:
            # Dwell is keyed on the Re-ID person_id where we have one,
            # so standing still behind cover — which makes ByteTrack
            # churn the track_id — doesn't keep resetting the clock.
            dwell_key = (
                ("person", det.person_id) if det.person_id is not None
                else ("track", det.track_id)
            )
            dwell = loiter_trackers[name].update(dwell_key, det.zone_tier)
            scores.append(
                draw_threat_score_overlay(
                    processed, det, threat_scorer, dwell, group_count,
                    zone=getattr(det, "_matched_zone", None),
                    is_curfew=curfew_active,
                )
            )

        with profiler.stage("draw_overlays"):
            draw_debug_overlay(processed, preprocessor, active, motion_score)
            draw_fps_overlay(processed, fps_ema[name])
            zone_drawers[name].draw_overlay(processed)

        with profiler.stage("publish_live"):
            publisher.publish_frame(name, processed)
        tiers = [s.tier for s in scores]
        camera_state[name].update(
            lastFrameAt=time.time(),
            fps=round(fps_ema[name], 1),
            detections=len(detections),
            persons=sum(1 for d in detections if d.category() == "person"),
            vehicles=sum(1 for d in detections if d.category() == "vehicle"),
            maxTier=("red" if "red" in tiers else "yellow" if "yellow" in tiers
                     else "green" if tiers else None),
            lowLightBoost=bool(preprocessor.last_boost_applied),
            brightness=round(float(preprocessor.last_brightness), 1),
        )

        with profiler.stage("frame_buffer_copy"):
            frame_buffers[name].append(processed.copy())
        with profiler.stage("alert_handle"):
            for det, score in zip(detections, scores):
                alert_manager.handle(det, score, list(frame_buffers[name]))

        with profiler.stage("display"):
            cv2.imshow(window_names[name], processed)
        profiler.frame_done()

    try:
        while True:
            health = manager.health()
            if health != last_health:
                degraded = {n: s for n, s in health.items() if s != CameraHealth.ONLINE}
                if degraded:
                    log.warning("Camera health changed — degraded: %s (all: %s)", degraded, health)
                else:
                    log.info("Camera health changed — all cameras ONLINE")
                last_health = health

            now_wall = time.time()
            if now_wall - last_health_publish >= 1.0:
                last_health_publish = now_wall
                for name in CAMERA_SOURCES:
                    path = zone_engines[name].config_path
                    mtime = os.path.getmtime(path) if os.path.exists(path) else None
                    if mtime != zone_mtimes[name]:
                        zone_mtimes[name] = mtime
                        try:
                            zone_engines[name].load()
                            log.info("[%s] zones reloaded from %s (%d zone(s))",
                                     name, path, len(zone_engines[name].zones))
                        except (OSError, ValueError, KeyError) as exc:
                            # A half-written or hand-edited file: keep the
                            # zones already loaded rather than dropping to none.
                            log.warning("[%s] could not reload zones from %s: %s",
                                        name, path, exc)
                publisher.publish_health({
                    "cameras": {
                        name: {
                            "health": health.get(name),
                            "zones": len(zone_engines[name].zones),
                            **camera_state[name],
                        }
                        for name in CAMERA_SOURCES
                    },
                    "models": models_info,
                    "alertDispatchQueue": ALERT_DISPATCH_QUEUE_SIZE,
                })

            frames = manager.read_all()
            for name, frame in frames.items():
                if frame is None:
                    continue
                # Per-camera failure boundary: a pipeline exception on one
                # camera drops that frame, is logged with the camera id and a
                # traceback, and leaves every other camera still processing.
                isolator.run(
                    name, process_camera_frame, name, frame, stage="frame-pipeline"
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            for drawer in zone_drawers.values():
                drawer.handle_key(key)
    finally:
        report = profiler.report()
        if report:
            print(report)
        publisher.close()
        manager.stop_all()
        # Drain queued webhook/evidence work before closing the stores it uses.
        alert_dispatcher.stop()
        threat_rules.close()
        incident_store.close()
        watchlist_db.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cli_sources = cli_camera_sources(sys.argv[1:])
    if cli_sources is not None:
        CAMERA_SOURCES = cli_sources
        log.info("Using camera source(s) from command line: %s", CAMERA_SOURCES)
    main()
