import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np

log = logging.getLogger("ibvap.zones")

ZONE_PRIORITY = {"red": 3, "yellow": 2, "green": 1}


class Zone:
    def __init__(
        self,
        zone_type: str,
        polygon: list,
        id: "str | None" = None,
        label: "str | None" = None,
        direction: "str | None" = None,
        policy: "dict | Any | None" = None,
        **extra,
    ):
        self.zone_type = zone_type  # "red" | "yellow" | "green"
        self.polygon = polygon  # list of (x, y) pixel points
        self.id = id
        self.label = label
        self.direction = direction
        self.policy = policy
        self.extra = extra
        for k, v in extra.items():
            setattr(self, k, v)

    def contains(self, point: tuple) -> bool:
        if not self.polygon:
            return False
        contour = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def centroid(self) -> tuple:
        if not self.polygon:
            return (0.0, 0.0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def to_dict(self) -> dict:
        d = {"zone_type": self.zone_type, "polygon": self.polygon}
        if self.id is not None:
            d["id"] = self.id
        if self.label is not None:
            d["label"] = self.label
        if self.direction is not None:
            d["direction"] = self.direction
        if self.policy is not None:
            d["policy"] = self.policy.to_dict() if hasattr(self.policy, "to_dict") else self.policy
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        data_copy = dict(data)
        zone_type = data_copy.pop("zone_type")
        polygon = [tuple(p) for p in data_copy.pop("polygon", [])]
        id_ = data_copy.pop("id", None)
        label = data_copy.pop("label", None)
        direction = data_copy.pop("direction", None)
        policy = data_copy.pop("policy", None)
        return cls(zone_type, polygon, id=id_, label=label, direction=direction, policy=policy, **data_copy)


class ZoneEngine:
    """Classifies a detection's ground position into a Red/Yellow/Green zone
    tier. Zones are simple polygons (drawn interactively — see
    zones/drawer.py) persisted to a small JSON file per camera, so this is
    the "virtual fence" the roadmap describes, minus the dashboard UI (that's
    Phase 13; this engine's logic doesn't change when that UI arrives).

    - Red: highest priority when zones overlap (border line).
    - Yellow: direction-aware — a track's movement direction is compared
      against the vector from this zone's centroid toward the nearest Red
      zone's centroid, to classify the movement as inward (toward the
      border) or outward.
    - Green: re-tiered to "yellow" during the configured curfew window, since
      an otherwise-safe interior area is more suspicious overnight.
    """

    def __init__(
        self,
        config_path: str,
        curfew_start_hour: int = 23,
        curfew_end_hour: int = 5,
        now_fn=datetime.now,
        fixed_tier: "str | None" = None,
    ):
        self.config_path = config_path
        self.curfew_start_hour = curfew_start_hour
        self.curfew_end_hour = curfew_end_hour
        self._now_fn = now_fn
        # A camera mounted at one point along the border sees one tier for
        # its whole frame - set this to skip polygon classification entirely
        # (see classify()) for a camera with no drawn zones.json at all.
        self.fixed_tier = fixed_tier
        self.zones: list = []
        self.load()

    def add_zone(self, zone: Zone) -> None:
        self.zones.append(zone)
        self.save()

    def clear(self) -> None:
        self.zones = []
        self.save()

    def load(self) -> None:
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path) as f:
            data = json.load(f)
        self.zones = [Zone.from_dict(z) for z in data]
        log.info("Loaded %d zone(s) from %s", len(self.zones), self.config_path)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump([z.to_dict() for z in self.zones], f, indent=2)

    def _is_curfew(self) -> bool:
        hour = self._now_fn().hour
        if self.curfew_start_hour > self.curfew_end_hour:
            return hour >= self.curfew_start_hour or hour < self.curfew_end_hour
        return self.curfew_start_hour <= hour < self.curfew_end_hour

    def classify(self, ground_point: tuple, direction: "tuple | None" = None) -> dict:
        """Returns {"tier": "red"|"yellow"|"green"|"none", "direction": "inward"|"outward"|None}"""
        if self.fixed_tier is not None:
            tier = self.fixed_tier
            direction_label = self._direction_label_fixed(direction)
            if tier == "green" and self._is_curfew():
                tier = "yellow"
            fixed_zone = getattr(self, "_fixed_zone", None)
            if fixed_zone is None or fixed_zone.zone_type != tier:
                fixed_zone = Zone(tier, [], label=f"Fixed {tier.upper()} Camera Tier")
                self._fixed_zone = fixed_zone
            return {
                "tier": tier,
                "direction": direction_label,
                "zone": fixed_zone,
                "zone_id": None,
                "zone_label": fixed_zone.label,
            }

        matches = [z for z in self.zones if z.contains(ground_point)]
        if not matches:
            return {"tier": "none", "direction": None, "zone": None, "zone_id": None, "zone_label": None}

        best = max(matches, key=lambda z: ZONE_PRIORITY[z.zone_type])
        tier = best.zone_type
        direction_label = self._direction_label(best, tier, direction)

        if tier == "green" and self._is_curfew():
            tier = "yellow"  # curfew re-tiering

        return {
            "tier": tier,
            "direction": direction_label,
            "zone": best,
            "zone_id": getattr(best, "id", None),
            "zone_label": getattr(best, "label", None),
        }

    def evaluate_policy(
        self,
        det,
        dwell_seconds: float = 0.0,
        hour: "int | None" = None,
        is_curfew: "bool | None" = None,
        threat_score=None,
        zone: "Zone | None" = None,
    ):
        """Evaluates zone policy rules against a detection."""
        if zone is None:
            box = getattr(det, "box", None)
            if box is not None:
                x1, y1, x2, y2 = box
                ground_point = ((x1 + x2) // 2, y2)
                cls_res = self.classify(ground_point, getattr(det, "direction", None))
                zone = cls_res.get("zone")
        if zone is None:
            return None
        from zones.zone_policy import evaluate_zone_policy

        if is_curfew is None:
            is_curfew = self._is_curfew()
        if hour is None:
            hour = self._now_fn().hour
        return evaluate_zone_policy(
            zone=zone,
            det=det,
            dwell_seconds=dwell_seconds,
            hour=hour,
            is_curfew=is_curfew,
            threat_score=threat_score,
        )

    # A track's direction vector is a displacement summed over the recent
    # history window, so it is never exactly zero — camera shake and box jitter
    # alone produce a pixel or two. Below this, treat the subject as stationary
    # and report no direction, rather than reading a crossing out of noise (and
    # tripping the border-crossing override on someone standing still).
    MIN_DIRECTION_MAGNITUDE = 4.0

    def _direction_label(self, zone, tier: str, direction) -> "str | None":
        """Movement relative to the border line.

        Red zone = the line itself, so movement through it is a crossing. The
        sign comes from the nearest Green zone (own territory): moving away
        from own side is "outward" (exfiltration/smuggling), toward it is
        "inward" (infiltration). With no Green zone drawn there is no way to
        tell the two apart, so it is reported as a plain "crossing" — still a
        breach, just without the sense.

        Yellow zone = the approach strip, so the reference is the nearest Red
        zone: moving toward the line is "inward".

        Movement across neither axis is "parallel" — travelling along the
        fence rather than at it, which is what reconnaissance looks like.
        """
        if direction is None:
            return None
        magnitude = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        if magnitude < self.MIN_DIRECTION_MAGNITUDE:
            return None  # stationary

        if tier == "red":
            reference = self._nearest_centroid(zone, "green")
            if reference is None:
                return "crossing"
            zx, zy = zone.centroid()
            # Vector pointing from own territory out toward the line.
            axis = (zx - reference[0], zy - reference[1])
            label_positive, label_negative = "outward", "inward"
        elif tier == "yellow":
            reference = self._nearest_centroid(zone, "red")
            if reference is None:
                return None
            zx, zy = zone.centroid()
            axis = (reference[0] - zx, reference[1] - zy)
            label_positive, label_negative = "inward", "outward"
        else:
            return None

        axis_magnitude = (axis[0] ** 2 + axis[1] ** 2) ** 0.5
        if axis_magnitude == 0:
            return None
        # Cosine between travel and the border axis. Near zero means moving
        # along the line rather than across it.
        cosine = (
            direction[0] * axis[0] + direction[1] * axis[1]
        ) / (magnitude * axis_magnitude)
        if abs(cosine) < 0.35:  # within ~20 degrees of parallel to the line
            return "parallel"
        return label_positive if cosine > 0 else label_negative

    def _direction_label_fixed(self, direction) -> "str | None":
        """Direction heuristic for a camera fixed to one zone tier: there is
        no other zone's centroid to reference, so this reads raw on-screen
        motion instead. A border camera faces across the line, so a ground
        point descending in frame (larger y = closer to the lens) reads as
        approaching the camera - "inward" - and rising reads as "outward",
        the same vocabulary _direction_label produces from zone geometry."""
        if direction is None:
            return None
        dx, dy = direction
        magnitude = (dx ** 2 + dy ** 2) ** 0.5
        if magnitude < self.MIN_DIRECTION_MAGNITUDE:
            return None
        if abs(dy) < abs(dx):
            return "parallel"
        return "inward" if dy > 0 else "outward"

    def _nearest_centroid(self, zone, zone_type: str) -> "tuple | None":
        candidates = [z for z in self.zones if z.zone_type == zone_type and z is not zone]
        if not candidates:
            return None
        zx, zy = zone.centroid()
        return min(
            (z.centroid() for z in candidates),
            key=lambda c: (c[0] - zx) ** 2 + (c[1] - zy) ** 2,
        )
