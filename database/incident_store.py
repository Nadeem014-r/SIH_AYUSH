import json
import logging
import os
import sqlite3
import time

import cv2
from cryptography.fernet import Fernet

log = logging.getLogger("ibvap.incidents")

STATUS_OPEN = "open"
STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_RESOLVED = "resolved"

# Controlled vocabulary for how an operator closed out an incident — doubles
# as a self-generating false-alarm dataset (most incidents in practice are
# cattle/vegetation, not genuine intrusions) once a real operator uses this.
RESOLUTION_REASONS = [
    "cattle",
    "vegetation",
    "authorized_personnel",
    "genuine_intrusion",
    "patrol_dispatched",
]


_BREAKDOWN_FIELDS = (
    "sector_risk",
    "time_risk",
    "kinematics_risk",
    "class_confidence",
    "direction_risk",
    "loiter_risk",
    "group_risk",
)


def _score_breakdown(score) -> dict:
    """The components of a ThreatScore, as stored alongside its total.

    Tolerant of partial score objects — a component that is absent or not a
    number is skipped rather than failing the insert, because losing an
    incident over a missing diagnostic field would be the wrong trade.
    """
    out = {}
    for field in _BREAKDOWN_FIELDS:
        value = getattr(score, field, None)
        try:
            out[field] = round(float(value), 2)
        except (TypeError, ValueError):
            continue
    for field in (
        "override_reason", "tier_ceiling", "ceiling_reason",
        "rule_name", "zone_id", "zone_label", "rule_evidence",
    ):
        value = getattr(score, field, None)
        if isinstance(value, str) and value:
            out[field] = value
    if getattr(score, "immediate", False):
        out["immediate"] = True
    return out


class IncidentStore:
    """Persists alert-worthy events to a local, queryable `incidents.db` and
    encrypts the accompanying evidence images at rest (Fernet, per the
    roadmap's zero-cost stack) — no cloud, no network dependency, matching
    the air-gapped deployment requirement. The encryption key is generated
    once and persisted locally; losing it makes existing evidence
    unreadable, same tradeoff as any local-key encryption scheme.
    """

    def __init__(
        self,
        db_path: str = "database/incidents.db",
        evidence_dir: str = "snapshots",
        key_path: str = "database/evidence.key",
    ):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(evidence_dir, exist_ok=True)
        self.evidence_dir = evidence_dir
        self._fernet = Fernet(self._load_or_create_key(key_path))
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()
        self._migrate_schema()

    def _load_or_create_key(self, key_path: str) -> bytes:
        os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
        log.info("Generated new evidence encryption key at %s", key_path)
        return key

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                person_id INTEGER,
                category TEXT,
                zone_tier TEXT,
                score REAL,
                tier TEXT,
                timestamp REAL,
                snapshot_path TEXT,
                crop_path TEXT,
                burst_paths TEXT
            )
            """
        )
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Adds operator-workflow columns to a pre-existing incidents.db
        without touching its existing rows — ALTER TABLE ADD COLUMN only,
        never a table rebuild, so real incident history survives the
        upgrade. Safe to call on every startup: only missing columns are added.
        """
        cur = self._conn.execute("PRAGMA table_info(incidents)")
        existing = {row[1] for row in cur.fetchall()}
        new_columns = {
            "status": f"TEXT NOT NULL DEFAULT '{STATUS_OPEN}'",
            "acknowledged_by": "TEXT",
            "acknowledged_at": "REAL",
            "resolved_by": "TEXT",
            "resolved_at": "REAL",
            "resolution_reason": "TEXT",
            # Which camera raised it. Without this an alert cannot be traced to
            # a location, and the dashboard can only say "unknown".
            "camera_name": "TEXT",
            # The threat-score components as JSON. Only total and tier were
            # stored before, so an operator could see *that* something scored
            # 84 but not whether that came from the zone, the hour or movement.
            "breakdown": "TEXT",
        }
        for column, definition in new_columns.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE incidents ADD COLUMN {column} {definition}")
        self._conn.commit()

    def acknowledge(self, incident_id: int, operator: str) -> None:
        now = time.time()
        self._conn.execute(
            "UPDATE incidents SET status = ?, acknowledged_by = ?, acknowledged_at = ? WHERE id = ?",
            (STATUS_ACKNOWLEDGED, operator, now, incident_id),
        )
        self._conn.commit()
        log.info("Incident #%d acknowledged by %s", incident_id, operator)

    def resolve(self, incident_id: int, operator: str, reason: str) -> None:
        if reason not in RESOLUTION_REASONS:
            raise ValueError(f"Unknown resolution reason: {reason!r}, expected one of {RESOLUTION_REASONS}")
        now = time.time()
        self._conn.execute(
            "UPDATE incidents SET status = ?, resolved_by = ?, resolved_at = ?, resolution_reason = ? "
            "WHERE id = ?",
            (STATUS_RESOLVED, operator, now, reason, incident_id),
        )
        self._conn.commit()
        log.info("Incident #%d resolved by %s (%s)", incident_id, operator, reason)

    def record(self, det, score, frame, crop_frame=None, burst_frames=None) -> int:
        timestamp = time.time()
        ts_label = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
        track_key = det.person_id if det.person_id is not None else det.track_id
        prefix = f"{score.tier}_{det.category()}_{track_key}_{ts_label}"

        snapshot_path = self._save_encrypted(frame, f"{prefix}_full.jpg.enc")

        crop_path = None
        if crop_frame is not None and crop_frame.size > 0:
            crop_path = self._save_encrypted(crop_frame, f"{prefix}_crop.jpg.enc")

        burst_paths = []
        for i, burst_frame in enumerate(burst_frames or []):
            path = self._save_encrypted(burst_frame, f"{prefix}_burst{i}.jpg.enc")
            if path:
                burst_paths.append(path)

        cur = self._conn.execute(
            """
            INSERT INTO incidents
                (track_id, person_id, category, zone_tier, score, tier, timestamp,
                 snapshot_path, crop_path, burst_paths, camera_name, breakdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                det.track_id, det.person_id, det.category(), det.zone_tier,
                score.total, score.tier, timestamp,
                snapshot_path, crop_path, json.dumps(burst_paths),
                # getattr: callers outside the live pipeline (tests, imports)
                # may pass objects that never had a camera attached.
                getattr(det, "camera_name", None),
                json.dumps(_score_breakdown(score)),
            ),
        )
        self._conn.commit()
        log.info("Recorded incident #%d (%s, score=%.0f)", cur.lastrowid, score.tier, score.total)
        return cur.lastrowid

    def _save_encrypted(self, frame, filename: str) -> "str | None":
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        encrypted = self._fernet.encrypt(buffer.tobytes())
        path = os.path.join(self.evidence_dir, filename)
        with open(path, "wb") as f:
            f.write(encrypted)
        return path

    def decrypt_image_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            encrypted = f.read()
        return self._fernet.decrypt(encrypted)

    def list_incidents(self, limit: int = 50) -> list:
        cur = self._conn.execute(
            "SELECT id, track_id, person_id, category, zone_tier, score, tier, timestamp, "
            "snapshot_path, crop_path, burst_paths, status, acknowledged_by, acknowledged_at, "
            "resolved_by, resolved_at, resolution_reason, camera_name, breakdown "
            "FROM incidents ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_incident(self, incident_id: int) -> "dict | None":
        cur = self._conn.execute(
            "SELECT id, track_id, person_id, category, zone_tier, score, tier, timestamp, "
            "snapshot_path, crop_path, burst_paths, status, acknowledged_by, acknowledged_at, "
            "resolved_by, resolved_at, resolution_reason, camera_name, breakdown "
            "FROM incidents WHERE id = ?",
            (incident_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        columns = [d[0] for d in cur.description]
        return dict(zip(columns, row))

    def close(self) -> None:
        self._conn.close()
