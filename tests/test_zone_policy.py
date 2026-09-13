"""Tests for the configurable zone-policy and rule layer.

Covers:
- RED policy rules: restricted-zone intrusion, boundary crossing, wrong direction,
  short loiter threshold, curfew presence -> immediate CRITICAL alert
- YELLOW policy rules: entry, medium loitering, wrong direction, movement toward RED,
  repeated suspicious movement -> REVIEW/HIGH alert and escalation toward RED
- GREEN policy rules: normal movement (no alert), long loitering (REVIEW),
  abnormal direction (REVIEW), movement toward YELLOW (escalation)
- Backward compatibility: legacy zone without policy maintains exact default scoring
- End-to-end integration: AlertManager, IncidentStore, API serialization, and WebSocket delivery
"""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from alerts.alert_manager import AlertManager
from database.incident_store import IncidentStore
from detection.detector import Detection
import integration.api as api_module
from intelligence.threat_rules import ThreatRulesDB
from intelligence.threat_score import ThreatScore, ThreatScorer
from zones.zone_engine import Zone, ZoneEngine
from zones.zone_policy import (
    ZonePolicy,
    ZonePolicyRuleMatch,
    evaluate_zone_policy,
)


def make_test_detection(
    category: str = "person",
    speed: float = 4.0,
    direction: tuple = (0.0, 5.0),
    zone_tier: str = "red",
    zone_direction: str = "inward",
    track_id: int = 1,
    person_id: int = 1,
) -> Detection:
    det = Detection(class_id=0 if category == "person" else 2, class_name=category, confidence=0.9, box=(10, 10, 50, 80))
    det.speed = speed
    det.direction = direction
    det.zone_tier = zone_tier
    det.zone_direction = zone_direction
    det.track_id = track_id
    det.person_id = person_id
    det.camera_name = "cam0"
    return det


class TestRedZonePolicy(unittest.TestCase):
    def setUp(self):
        self.red_zone = Zone(
            "red",
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            id="zone-red-1",
            label="Border Perimeter Alpha",
            policy={"enabled": True, "short_loiter_seconds": 10.0},
        )

    def test_restricted_zone_intrusion(self):
        """Any entity inside RED zone without special vector triggers immediate restricted intrusion alert."""
        det = make_test_detection(zone_tier="red", zone_direction=None, speed=3.0)
        # Disable crossing and wrong direction to test basic intrusion
        zone = Zone(
            "red",
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            id="zone-red-1",
            label="Border Perimeter Alpha",
            policy={
                "enabled": True,
                "rules": {"boundary_crossing": False, "wrong_direction": False},
            },
        )
        match = evaluate_zone_policy(zone, det, dwell_seconds=2.0)
        self.assertIsNotNone(match)
        self.assertEqual(match.rule_name, "restricted_zone_intrusion")
        self.assertEqual(match.tier, "red")
        self.assertEqual(match.alert_severity, "CRITICAL")
        self.assertTrue(match.immediate)
        self.assertIn("Border Perimeter Alpha", match.reason)
        self.assertIn("restricted_zone_intrusion", match.reason)

    def test_boundary_crossing(self):
        """Crossing vector triggers boundary_crossing rule."""
        for crossing_dir in ("crossing", "inward", "outward"):
            det = make_test_detection(zone_tier="red", zone_direction=crossing_dir, speed=6.5)
            match = evaluate_zone_policy(self.red_zone, det, dwell_seconds=3.0)
            self.assertIsNotNone(match)
            self.assertEqual(match.rule_name, "boundary_crossing")
            self.assertEqual(match.tier, "red")
            self.assertEqual(match.alert_severity, "CRITICAL")
            self.assertTrue(match.immediate)
            self.assertIn("crossing boundary", match.reason)

    def test_wrong_direction(self):
        """Direction toward interior or designated prohibited direction triggers wrong_direction rule."""
        zone = Zone(
            "red",
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            label="Restricted Border Wall",
            policy={
                "enabled": True,
                "prohibited_direction": "inward",
                "rules": {"boundary_crossing": False},
            },
        )
        det = make_test_detection(zone_tier="red", zone_direction="inward", speed=4.0)
        match = evaluate_zone_policy(zone, det, dwell_seconds=1.0)
        self.assertIsNotNone(match)
        self.assertEqual(match.rule_name, "wrong_direction")
        self.assertTrue(match.immediate)
        self.assertEqual(match.tier, "red")

    def test_short_loiter_threshold(self):
        """Dwell >= short_loiter_seconds (10s) triggers loiter alarm."""
        zone = Zone(
            "red",
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            label="Border Fence Alpha",
            policy={
                "enabled": True,
                "short_loiter_seconds": 10.0,
                "rules": {"boundary_crossing": False, "wrong_direction": False},
            },
        )
        det = make_test_detection(zone_tier="red", zone_direction=None, speed=0.5)
        # Below threshold
        match_short = evaluate_zone_policy(zone, det, dwell_seconds=8.0)
        self.assertEqual(match_short.rule_name, "restricted_zone_intrusion")

        # At or above threshold
        match_loiter = evaluate_zone_policy(zone, det, dwell_seconds=11.5)
        self.assertEqual(match_loiter.rule_name, "short_loiter_threshold")
        self.assertTrue(match_loiter.immediate)
        self.assertEqual(match_loiter.tier, "red")
        self.assertIn("loitering in restricted area", match_loiter.reason)

    def test_curfew_presence(self):
        """Target in RED zone during curfew window triggers curfew rule."""
        zone = Zone(
            "red",
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            label="Night Patrol Zone",
            policy={
                "enabled": True,
                "curfew_start_hour": 23,
                "curfew_end_hour": 5,
                "rules": {
                    "boundary_crossing": False,
                    "wrong_direction": False,
                    "short_loiter_threshold": False,
                },
            },
        )
        det = make_test_detection(zone_tier="red", zone_direction=None, speed=2.0)
        # Day (14:00) -> normal restricted intrusion
        match_day = evaluate_zone_policy(zone, det, dwell_seconds=1.0, hour=14)
        self.assertEqual(match_day.rule_name, "restricted_zone_intrusion")

        # Night (02:00) -> curfew presence
        match_curfew = evaluate_zone_policy(zone, det, dwell_seconds=1.0, hour=2)
        self.assertEqual(match_curfew.rule_name, "curfew_presence")
        self.assertTrue(match_curfew.immediate)
        self.assertEqual(match_curfew.tier, "red")
        self.assertIn("curfew", match_curfew.reason)


class TestYellowZonePolicy(unittest.TestCase):
    def setUp(self):
        self.yellow_zone = Zone(
            "yellow",
            [(100, 0), (200, 0), (200, 100), (100, 100)],
            id="zone-yellow-1",
            label="Buffer Strip Bravo",
            policy={"enabled": True, "medium_loiter_seconds": 30.0},
        )

    def test_buffer_entry(self):
        """Entry in yellow zone without other triggers produces REVIEW entry alert."""
        det = make_test_detection(zone_tier="yellow", zone_direction=None, speed=4.0)
        match = evaluate_zone_policy(self.yellow_zone, det, dwell_seconds=5.0)
        self.assertIsNotNone(match)
        self.assertEqual(match.rule_name, "entry")
        self.assertEqual(match.tier, "yellow")
        self.assertEqual(match.alert_severity, "REVIEW")
        self.assertFalse(match.immediate)
        self.assertIn("entered sensitive buffer zone", match.reason)

    def test_medium_loitering(self):
        """Dwell >= medium_loiter_seconds (30s) produces HIGH loitering alert."""
        det = make_test_detection(zone_tier="yellow", zone_direction=None, speed=3.0)
        match = evaluate_zone_policy(self.yellow_zone, det, dwell_seconds=35.0)
        self.assertEqual(match.rule_name, "medium_loitering")
        self.assertEqual(match.tier, "yellow")
        self.assertEqual(match.alert_severity, "HIGH")
        self.assertIn("dwelling in buffer strip", match.reason)

    def test_movement_toward_red(self):
        """Direction inward toward RED produces HIGH alert."""
        det = make_test_detection(zone_tier="yellow", zone_direction="inward", speed=4.0)
        match = evaluate_zone_policy(self.yellow_zone, det, dwell_seconds=5.0)
        self.assertEqual(match.rule_name, "movement_toward_red")
        self.assertEqual(match.tier, "yellow")
        self.assertEqual(match.alert_severity, "HIGH")
        self.assertIn("moving toward RED", match.reason)

    def test_repeated_suspicious_movement(self):
        """Crouching (<= 1 px/f) or fast sprint (>= 12 px/f) or parallel recon triggers suspicious movement."""
        det_crouch = make_test_detection(zone_tier="yellow", zone_direction=None, speed=0.5)
        match_crouch = evaluate_zone_policy(self.yellow_zone, det_crouch, dwell_seconds=4.0)
        self.assertEqual(match_crouch.rule_name, "repeated_suspicious_movement")
        self.assertEqual(match_crouch.alert_severity, "HIGH")

        det_parallel = make_test_detection(zone_tier="yellow", zone_direction="parallel", speed=3.5)
        match_recon = evaluate_zone_policy(self.yellow_zone, det_parallel, dwell_seconds=4.0)
        self.assertEqual(match_recon.rule_name, "repeated_suspicious_movement")

    def test_escalation_toward_red(self):
        """Movement toward RED + loitering (or crawling/sprinting toward red) escalates to RED."""
        # Target moving inward toward red AND loitering for 32 seconds
        det = make_test_detection(zone_tier="yellow", zone_direction="inward", speed=4.0)
        match = evaluate_zone_policy(self.yellow_zone, det, dwell_seconds=32.0)
        self.assertEqual(match.rule_name, "movement_toward_red")
        self.assertEqual(match.tier, "red")
        self.assertTrue(match.escalated)
        self.assertTrue(match.immediate)
        self.assertEqual(match.alert_severity, "CRITICAL")
        self.assertIn("ESCALATED TO RED", match.reason)


class TestGreenZonePolicy(unittest.TestCase):
    def setUp(self):
        self.green_zone = Zone(
            "green",
            [(200, 0), (300, 0), (300, 100), (200, 100)],
            id="zone-green-1",
            label="Internal Camp Corridor",
            policy={"enabled": True, "long_loiter_seconds": 60.0},
        )

    def test_normal_movement_no_alert(self):
        """Standard movement in green zone produces normal_movement with no alert."""
        det = make_test_detection(zone_tier="green", zone_direction=None, speed=4.0)
        match = evaluate_zone_policy(self.green_zone, det, dwell_seconds=5.0)
        self.assertEqual(match.rule_name, "normal_movement")
        self.assertEqual(match.tier, "green")
        self.assertEqual(match.alert_severity, "NONE")

    def test_long_loitering_review(self):
        """Dwell >= long_loiter_seconds (60s) in green zone escalates to REVIEW (yellow tier)."""
        det = make_test_detection(zone_tier="green", zone_direction=None, speed=2.0)
        match = evaluate_zone_policy(self.green_zone, det, dwell_seconds=65.0)
        self.assertEqual(match.rule_name, "long_loitering")
        self.assertEqual(match.tier, "yellow")
        self.assertEqual(match.alert_severity, "REVIEW")
        self.assertTrue(match.escalated)
        self.assertIn("extended dwell", match.reason)

    def test_abnormal_direction_review(self):
        """Anomalous direction vector in green zone triggers REVIEW."""
        det = make_test_detection(zone_tier="green", zone_direction="parallel", speed=3.0)
        match = evaluate_zone_policy(self.green_zone, det, dwell_seconds=5.0)
        self.assertEqual(match.rule_name, "abnormal_direction")
        self.assertEqual(match.tier, "yellow")
        self.assertEqual(match.alert_severity, "REVIEW")

    def test_movement_toward_yellow_escalation(self):
        """Movement toward yellow buffer zone escalates to REVIEW."""
        det = make_test_detection(zone_tier="green", zone_direction="outward", speed=4.5)
        match = evaluate_zone_policy(self.green_zone, det, dwell_seconds=5.0)
        self.assertEqual(match.rule_name, "movement_toward_yellow")
        self.assertEqual(match.tier, "yellow")
        self.assertEqual(match.alert_severity, "REVIEW")
        self.assertTrue(match.escalated)


class TestBackwardCompatibility(unittest.TestCase):
    def test_zone_without_policy_behaves_normally(self):
        """Legacy zones with no policy attribute pass through unchanged."""
        legacy_zone = Zone("green", [(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertIsNone(legacy_zone.policy)

        tmp_dir = tempfile.mkdtemp()
        rules_db = ThreatRulesDB(db_path=str(Path(tmp_dir) / "threat_rules.db"))
        scorer = ThreatScorer(rules_db)

        # Legacy score invocation without zone:
        score_legacy = scorer.score(
            zone_tier="green",
            hour=12,
            speed_px_per_frame=4.0,
            category="person",
            zone_direction=None,
            dwell_seconds=5.0,
        )
        self.assertEqual(score_legacy.tier, "green")
        self.assertIsNone(score_legacy.override_reason)
        self.assertIsNone(score_legacy.rule_name)

    def test_zone_engine_classify_backward_compatible(self):
        """ZoneEngine.classify() provides tier and direction as before."""
        tmp_dir = tempfile.mkdtemp()
        engine = ZoneEngine(config_path=str(Path(tmp_dir) / "zones.json"))
        zone = Zone("red", [(0, 0), (100, 0), (100, 100), (0, 100)])
        engine.add_zone(zone)

        result = engine.classify((50, 50))
        self.assertEqual(result["tier"], "red")
        self.assertIn("direction", result)
        # Extra fields present for modern consumers
        self.assertEqual(result["zone"], zone)


class TestEndToEndAlertDelivery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rules_db = ThreatRulesDB(db_path=str(Path(self.tmp) / "threat_rules.db"))
        self.scorer = ThreatScorer(self.rules_db)
        self.store = IncidentStore(
            db_path=str(Path(self.tmp) / "incidents.db"),
            evidence_dir=str(Path(self.tmp) / "snapshots"),
            key_path=str(Path(self.tmp) / "evidence.key"),
        )
        self.received_webhooks = []
        mock_webhook = types.SimpleNamespace(notify=lambda payload: self.received_webhooks.append(payload))
        self.received_syslogs = []
        mock_syslog = types.SimpleNamespace(emit=lambda msg: self.received_syslogs.append(msg))
        self.alert_manager = AlertManager(
            snapshot_dir=str(Path(self.tmp) / "snapshots"),
            incident_store=self.store,
            webhook=mock_webhook,
            syslog=mock_syslog,
            cooldown_seconds=5.0,
            confirm_n=2,
            confirm_window=3,
        )

    def tearDown(self):
        self.rules_db.close()
        self.store.close()

    def test_red_rule_immediate_alert_and_websocket_serialization(self):
        """RED rule trigger bypasses confirm_n hysteresis, stores explainable event, and serializes for /ws/alerts."""
        zone = Zone(
            "red",
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            id="zone-red-1",
            label="Restricted Fence Perimeter",
            policy={"enabled": True},
        )
        det = make_test_detection(zone_tier="red", zone_direction="crossing", speed=6.0, track_id=42, person_id=101)
        det.camera_name = "cam0"

        # Threat scorer evaluates zone policy
        score = self.scorer.score(
            zone_tier=det.zone_tier,
            hour=14,
            speed_px_per_frame=det.speed,
            category=det.category(),
            zone_direction=det.zone_direction,
            dwell_seconds=5.0,
            zone=zone,
        )
        self.assertEqual(score.tier, "red")
        self.assertTrue(score.immediate)
        self.assertEqual(score.rule_name, "boundary_crossing")
        self.assertIn("boundary_crossing", score.override_reason)

        # AlertManager handles detection with frame
        import numpy as np
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        # Verify it confirms on the VERY FIRST frame (due to score.immediate = True)
        self.alert_manager.handle(det, score, [frame])

        # Verify incident was persisted in database
        incidents = self.store.list_incidents(limit=10)
        self.assertEqual(len(incidents), 1)
        incident_row = incidents[0]
        self.assertEqual(incident_row["track_id"], 42)
        self.assertEqual(incident_row["person_id"], 101)
        self.assertEqual(incident_row["tier"], "red")

        # Verify rule breakdown stored in breakdown JSON
        breakdown_json = json.loads(incident_row["breakdown"])
        self.assertEqual(breakdown_json.get("rule_name"), "boundary_crossing")
        self.assertEqual(breakdown_json.get("zone_label"), "Restricted Fence Perimeter")
        self.assertIn("crossing boundary", breakdown_json.get("rule_evidence", ""))

        # Verify API serialisation (the exact shape delivered over WebSocket and REST)
        serialised = api_module._serialise(incident_row)
        self.assertEqual(serialised["tier"], "red")
        self.assertIn("breakdown", serialised)
        self.assertEqual(serialised["breakdown"]["ruleName"], "boundary_crossing")
        self.assertEqual(serialised["breakdown"]["zoneLabel"], "Restricted Fence Perimeter")
        self.assertIn("boundary_crossing", serialised["breakdown"]["overrideReason"])

        # Verify outbound integrations received explainable fields
        self.assertEqual(len(self.received_webhooks), 1)
        self.assertEqual(self.received_webhooks[0]["rule_name"], "boundary_crossing")
        self.assertIn("boundary_crossing", self.received_webhooks[0]["override_reason"])

        self.assertEqual(len(self.received_syslogs), 1)
        self.assertIn("rule=boundary_crossing", self.received_syslogs[0])

    def test_api_zone_policy_endpoints(self):
        """Verify GET /api/v1/zones/{camera_id}/policy and PUT /api/v1/zones/{camera_id}/policy."""
        from fastapi.testclient import TestClient
        client = TestClient(api_module.app)
        token = "test-token"
        with patch.object(api_module, "IBVAP_API_TOKEN", token), \
             patch.object(api_module, "_camera_sources", return_value={"cam0": 0}):
            # Setup mock zone file
            zones_data = [
                {
                    "zone_type": "red",
                    "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)],
                    "id": "zone-cam0-0",
                    "label": "Front Gate",
                    "policy": {"enabled": True, "short_loiter_seconds": 12.0},
                }
            ]
            path = "config/zones_cam0.json"
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(zones_data, f)

            try:
                headers = {"Authorization": f"Bearer {token}"}
                res = client.get("/api/v1/zones/cam0/policy", headers=headers)
                self.assertEqual(res.status_code, 200)
                body = res.json()
                self.assertEqual(body["cameraId"], "cam0")
                self.assertEqual(len(body["zones"]), 1)
                self.assertEqual(body["zones"][0]["policy"]["short_loiter_seconds"], 12.0)

                # Update policy via PUT
                update_res = client.put(
                    "/api/v1/zones/cam0/policy",
                    json={"zones": {"zone-cam0-0": {"enabled": True, "short_loiter_seconds": 15.0}}},
                    headers=headers,
                )
                self.assertEqual(update_res.status_code, 200)
                self.assertEqual(update_res.json()["updatedCount"], 1)

                # Verify persistence
                with open(path) as f:
                    updated = json.load(f)
                self.assertEqual(updated[0]["policy"]["short_loiter_seconds"], 15.0)
            finally:
                if Path(path).exists():
                    Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
