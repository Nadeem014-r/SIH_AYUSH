import logging

import numpy as np

try:
    # The package itself may not be installed at all (not just missing its
    # model cache) — e.g. a dev/CI machine without the ML stack, or a partial
    # deployment. That used to be an uncaught ModuleNotFoundError out of this
    # import, which crashed any caller transitively importing this module
    # (including face/watchlist.py and its FastAPI routes) before __init__'s
    # own try/except below ever got a chance to degrade gracefully. Kept as a
    # module-level name (None, not omitted) so it stays patchable the same
    # way tests already patch "face.face_recognizer.FaceAnalysis".
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from config.providers import select_providers

log = logging.getLogger("ibvap.face")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceRecognizer:
    """Wraps InsightFace's buffalo_s pack (bundles its own face detector plus
    a 512-d ArcFace-style recognition embedding) to extract a face embedding
    from a person's bounding box crop. Should only be called on detections
    that already passed the false-alarm filter, per the roadmap's "keeps
    cost low" guidance — face embedding is one of the pricier steps.
    """

    # Faces sit in the top portion of a person's box. Handing the *whole*
    # (tall, narrow) person box to the detector forces it to shrink the
    # entire crop down to fit det_size, which shrinks the face far more
    # than necessary and is the main cause of inconsistent detection.
    # Cropping to just the head/shoulder region keeps the face much larger
    # relative to the detector's analysis window.
    HEAD_HEIGHT_FRACTION = 0.5

    def __init__(self, det_size: tuple = (416, 416), det_thresh: float = 0.4):
        log.info("Loading InsightFace buffalo_s for face recognition")
        # Only the detector and the recognition head are used here: embed()
        # reads face.bbox (detection) and face.normed_embedding (recognition)
        # and nothing else. Left unrestricted, FaceAnalysis also loads and RUNS
        # 2D landmarks, 3D landmarks and gender/age on every detected face,
        # every call — computed and thrown away. Same pack, same two models,
        # same output; measured effect is ~97MB less resident memory, and it
        # only saves inference time on frames where a face is actually found
        # (the discarded modules run per detected face, not per call).
        #
        # buffalo_s auto-downloads on first use if ~/.insightface has no
        # cache yet — fine on a machine with internet, but a hard crash (not
        # a warning) that takes down the whole pipeline on a fresh, offline
        # machine, which is exactly the deployment this project targets.
        # Face recognition/watchlist matching is one feature, not a startup
        # requirement, so a failure here degrades that feature only — same
        # pattern as alerts/alert_manager.py's audio fallback.
        self.available = True
        if FaceAnalysis is None:
            # The insightface package itself isn't installed — a different
            # failure than a missing/uncached model, but the same graceful
            # degradation applies: this feature is disabled, everything else
            # (detection, tracking, zones, scoring, alerting, incident DB)
            # runs unaffected.
            self.available = False
            self._app = None
            log.warning(
                "Face recognition unavailable (insightface is not installed) — "
                "running without face recognition or watchlist matching; "
                "everything else is unaffected. Install insightface (see "
                "requirements.txt) to enable this."
            )
            return
        try:
            self._app = FaceAnalysis(
                name="buffalo_s",
                providers=select_providers(),
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        except Exception as e:
            self.available = False
            self._app = None
            log.warning(
                "Face recognition unavailable (%s) — no cached model at "
                "~/.insightface and no network to fetch it. Running without "
                "face recognition or watchlist matching; everything else is "
                "unaffected. Copy ~/.insightface/models/buffalo_s from a "
                "machine that has it to fix this offline.",
                e,
            )

    def embed(self, frame, person_box: tuple):
        """Returns (face_box_in_frame_coords, embedding) for the largest
        detected face within person_box's head/shoulder region, or
        (None, None) if no face found (or face recognition is unavailable)."""
        if not self.available:
            return None, None
        x1, y1, x2, y2 = person_box
        x1, y1 = max(x1, 0), max(y1, 0)
        height = y2 - y1
        if height <= 0 or x2 <= x1:
            return None, None

        head_bottom = y1 + max(int(height * self.HEAD_HEIGHT_FRACTION), 1)
        crop = frame[y1:head_bottom, x1:x2]
        if crop.size == 0:
            return None, None

        faces = self._app.get(crop)
        if not faces:
            return None, None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        face_box = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
        return face_box, face.normed_embedding
