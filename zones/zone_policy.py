"""Configurable zone policy and rule evaluation engine.

Implements operator-configured rules per zone tier:
- RED (critical/restricted):
  - restricted-zone intrusion
  - boundary crossing
  - wrong direction
  - short loiter threshold
  - curfew presence
  -> immediate CRITICAL alert

- YELLOW (sensitive/buffer):
  - entry
  - medium loitering
  - wrong direction
  - movement toward RED
  - repeated suspicious movement
  -> REVIEW/HIGH alert and escalate toward RED

- GREEN (normal):
  - normal movement -> no alert
  - long loitering -> REVIEW
  - abnormal direction -> REVIEW
  - movement toward YELLOW -> escalation
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

log = logging.getLogger("ibvap.zones.policy")

DEFAULT_SHORT_LOITER_SECONDS = 10.0
DEFAULT_MEDIUM_LOITER_SECONDS = 30.0
DEFAULT_LONG_LOITER_SECONDS = 60.0
DEFAULT_CURFEW_START_HOUR = 23
DEFAULT_CURFEW_END_HOUR = 5


@dataclass
class ZonePolicyRuleMatch:
    matched: bool
    rule_name: str
    tier: str  # "red" | "yellow" | "green"
    alert_severity: str  # "CRITICAL" | "HIGH" | "REVIEW" | "NONE"
    immediate: bool = False
    escalated: bool = False
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    score_floor: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ZonePolicy:
    """Configurable policy settings and rules for a specific zone."""

    def __init__(
        self,
        tier: str = "green",
        enabled: bool = True,
        short_loiter_seconds: float = DEFAULT_SHORT_LOITER_SECONDS,
        medium_loiter_seconds: float = DEFAULT_MEDIUM_LOITER_SECONDS,
        long_loiter_seconds: float = DEFAULT_LONG_LOITER_SECONDS,
        curfew_start_hour: int = DEFAULT_CURFEW_START_HOUR,
        curfew_end_hour: int = DEFAULT_CURFEW_END_HOUR,
        allowed_direction: Optional[str] = None,
        prohibited_direction: Optional[str] = None,
        escalate_to_red: bool = True,
        rules: Optional[dict] = None,
    ):
        self.tier = tier.lower()
        self.enabled = enabled
        self.short_loiter_seconds = float(short_loiter_seconds)
        self.medium_loiter_seconds = float(medium_loiter_seconds)
        self.long_loiter_seconds = float(long_loiter_seconds)
        self.curfew_start_hour = int(curfew_start_hour)
        self.curfew_end_hour = int(curfew_end_hour)
        self.allowed_direction = allowed_direction
        self.prohibited_direction = prohibited_direction
        self.escalate_to_red = escalate_to_red
        self.rules = rules or {}

    def is_rule_enabled(self, rule_name: str) -> bool:
        """Returns whether a specific named rule is active (defaults to True)."""
        if not self.enabled:
            return False
        return bool(self.rules.get(rule_name, True))

    def is_curfew_hour(self, hour: int) -> bool:
        if self.curfew_start_hour > self.curfew_end_hour:
            return hour >= self.curfew_start_hour or hour < self.curfew_end_hour
        return self.curfew_start_hour <= hour < self.curfew_end_hour

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "enabled": self.enabled,
            "short_loiter_seconds": self.short_loiter_seconds,
            "medium_loiter_seconds": self.medium_loiter_seconds,
            "long_loiter_seconds": self.long_loiter_seconds,
            "curfew_start_hour": self.curfew_start_hour,
            "curfew_end_hour": self.curfew_end_hour,
            "allowed_direction": self.allowed_direction,
            "prohibited_direction": self.prohibited_direction,
            "escalate_to_red": self.escalate_to_red,
            "rules": self.rules,
        }

    @classmethod
    def from_dict(cls, data: dict, default_tier: str = "green") -> "ZonePolicy":
        if not isinstance(data, dict):
            return cls(tier=default_tier)
        return cls(
            tier=data.get("tier", default_tier),
            enabled=data.get("enabled", True),
            short_loiter_seconds=data.get("short_loiter_seconds", DEFAULT_SHORT_LOITER_SECONDS),
            medium_loiter_seconds=data.get("medium_loiter_seconds", DEFAULT_MEDIUM_LOITER_SECONDS),
            long_loiter_seconds=data.get("long_loiter_seconds", DEFAULT_LONG_LOITER_SECONDS),
            curfew_start_hour=data.get("curfew_start_hour", DEFAULT_CURFEW_START_HOUR),
            curfew_end_hour=data.get("curfew_end_hour", DEFAULT_CURFEW_END_HOUR),
            allowed_direction=data.get("allowed_direction"),
            prohibited_direction=data.get("prohibited_direction"),
            escalate_to_red=data.get("escalate_to_red", True),
            rules=data.get("rules", {}),
        )


def evaluate_zone_policy(
    zone: Any,
    det: Any,
    dwell_seconds: float = 0.0,
    hour: Optional[int] = None,
    is_curfew: Optional[bool] = None,
    threat_score: Optional[Any] = None,
) -> Optional[ZonePolicyRuleMatch]:
    """Evaluates the configured policy for a zone using existing pipeline outputs:
    - detection: category, speed, direction, zone_direction, box
    - tracking: dwell_seconds, person_id, track_id
    - zone: zone_type, id, label, policy
    - time: hour, curfew
    - threat outputs: kinematics risk, threat score total

    Returns a ZonePolicyRuleMatch with explainable details if a rule fires,
    or None if no rule matches or policy is disabled.
    """
    if zone is None:
        return None

    zone_tier = getattr(zone, "zone_type", None) or "none"
    zone_tier = zone_tier.lower()
    if zone_tier not in ("red", "yellow", "green"):
        return None

    # Retrieve or construct ZonePolicy
    raw_policy = getattr(zone, "policy", None)
    if raw_policy is None:
        # Unconfigured / legacy zone preserves existing behavior
        return None
    elif isinstance(raw_policy, ZonePolicy):
        policy = raw_policy
    elif isinstance(raw_policy, dict):
        policy = ZonePolicy.from_dict(raw_policy, default_tier=zone_tier)
    else:
        policy = ZonePolicy(tier=zone_tier)

    if not policy.enabled:
        return None

    # Check zone-level overrides if specified on zone (e.g. loiteringThresholdSeconds)
    loiter_override = getattr(zone, "loiteringThresholdSeconds", None)
    short_loiter_threshold = (
        float(loiter_override)
        if loiter_override and zone_tier == "red"
        else policy.short_loiter_seconds
    )
    medium_loiter_threshold = (
        float(loiter_override)
        if loiter_override and zone_tier == "yellow"
        else policy.medium_loiter_seconds
    )
    long_loiter_threshold = (
        float(loiter_override)
        if loiter_override and zone_tier == "green"
        else policy.long_loiter_seconds
    )

    zone_label = getattr(zone, "label", None) or getattr(zone, "id", None) or f"{zone_tier.upper()} zone"
    category = det.category() if hasattr(det, "category") else getattr(det, "class_name", "target")
    speed = float(getattr(det, "speed", 0.0) or 0.0)
    zone_direction = getattr(det, "zone_direction", None)
    raw_direction = getattr(det, "direction", None)

    # Curfew determination
    if is_curfew is None:
        if hour is not None:
            curfew_active = policy.is_curfew_hour(hour)
        else:
            curfew_active = False
    else:
        curfew_active = bool(is_curfew)

    context = {
        "zone_tier": zone_tier,
        "zone_label": zone_label,
        "category": category,
        "speed": round(speed, 2),
        "zone_direction": zone_direction,
        "dwell_seconds": round(dwell_seconds, 1),
        "is_curfew": curfew_active,
        "hour": hour,
    }

    # =========================================================================
    # RED = Critical / Restricted Policy
    # =========================================================================
    if zone_tier == "red":
        # 1. Boundary crossing
        if policy.is_rule_enabled("boundary_crossing") and zone_direction in ("crossing", "inward", "outward"):
            reason = (
                f"Zone '{zone_label}' (RED) Rule 'boundary_crossing': "
                f"{category.capitalize()} crossing boundary ({zone_direction}, speed={speed:.1f}px/f)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="boundary_crossing",
                tier="red",
                alert_severity="CRITICAL",
                immediate=True,
                reason=reason,
                evidence={**context, "rule": "boundary_crossing"},
                score_floor=90.0,
            )

        # 2. Wrong direction (moving inward/toward protected boundary)
        wrong_dir = policy.prohibited_direction or "inward"
        if policy.is_rule_enabled("wrong_direction") and zone_direction == wrong_dir:
            reason = (
                f"Zone '{zone_label}' (RED) Rule 'wrong_direction': "
                f"{category.capitalize()} moving in prohibited direction ({zone_direction})"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="wrong_direction",
                tier="red",
                alert_severity="CRITICAL",
                immediate=True,
                reason=reason,
                evidence={**context, "rule": "wrong_direction"},
                score_floor=88.0,
            )

        # 3. Short loiter threshold
        if policy.is_rule_enabled("short_loiter_threshold") and dwell_seconds >= short_loiter_threshold:
            reason = (
                f"Zone '{zone_label}' (RED) Rule 'short_loiter_threshold': "
                f"{category.capitalize()} loitering in restricted area ({dwell_seconds:.1f}s >= {short_loiter_threshold:.1f}s)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="short_loiter_threshold",
                tier="red",
                alert_severity="CRITICAL",
                immediate=True,
                reason=reason,
                evidence={**context, "rule": "short_loiter_threshold", "threshold": short_loiter_threshold},
                score_floor=85.0,
            )

        # 4. Curfew presence
        if policy.is_rule_enabled("curfew_presence") and curfew_active:
            hour_str = f"{hour:02d}:00" if hour is not None else "curfew hours"
            reason = (
                f"Zone '{zone_label}' (RED) Rule 'curfew_presence': "
                f"{category.capitalize()} presence in restricted zone during curfew ({hour_str})"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="curfew_presence",
                tier="red",
                alert_severity="CRITICAL",
                immediate=True,
                reason=reason,
                evidence={**context, "rule": "curfew_presence"},
                score_floor=88.0,
            )

        # 5. Restricted-zone intrusion (any confirmed presence in red)
        if policy.is_rule_enabled("restricted_zone_intrusion"):
            reason = (
                f"Zone '{zone_label}' (RED) Rule 'restricted_zone_intrusion': "
                f"{category.capitalize()} detected inside restricted zone (speed={speed:.1f}px/f)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="restricted_zone_intrusion",
                tier="red",
                alert_severity="CRITICAL",
                immediate=True,
                reason=reason,
                evidence={**context, "rule": "restricted_zone_intrusion"},
                score_floor=80.0,
            )

    # =========================================================================
    # YELLOW = Sensitive / Buffer Policy
    # =========================================================================
    elif zone_tier == "yellow":
        is_toward_red = zone_direction == "inward"
        is_wrong_dir = (
            (policy.prohibited_direction and zone_direction == policy.prohibited_direction)
            or (policy.allowed_direction and zone_direction and zone_direction != policy.allowed_direction)
            or (zone_direction == "inward")
        )
        is_medium_loiter = dwell_seconds >= medium_loiter_threshold

        # Repeated suspicious movement: crouching (speed <= 1.0) or sprinting (speed >= 12.0)
        # or parallel reconnaissance along border fence
        is_suspicious_motion = (
            speed <= 1.0 or speed >= 12.0 or zone_direction == "parallel"
        )
        has_high_kinematics = (
            threat_score is not None and getattr(threat_score, "kinematics_risk", 0.0) >= 8.0
        )
        is_repeated_suspicious = is_suspicious_motion or has_high_kinematics

        # ESCALATION TOWARD RED:
        # If moving toward RED + loitering, OR wrong direction + repeated suspicious movement
        escalate = policy.escalate_to_red and (
            (is_toward_red and is_medium_loiter)
            or (is_wrong_dir and is_repeated_suspicious)
            or (is_toward_red and speed >= 10.0)
        )
        if escalate and policy.is_rule_enabled("movement_toward_red"):
            reason = (
                f"Zone '{zone_label}' (YELLOW) Rule 'movement_toward_red' [ESCALATED TO RED]: "
                f"{category.capitalize()} moving toward RED boundary with suspicious indicators "
                f"(direction={zone_direction}, dwell={dwell_seconds:.1f}s, speed={speed:.1f}px/f)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="movement_toward_red",
                tier="red",
                alert_severity="CRITICAL",
                immediate=True,
                escalated=True,
                reason=reason,
                evidence={**context, "rule": "movement_toward_red", "escalated": True},
                score_floor=75.0,
            )

        # 1. Movement toward RED (without full escalation)
        if policy.is_rule_enabled("movement_toward_red") and is_toward_red:
            reason = (
                f"Zone '{zone_label}' (YELLOW) Rule 'movement_toward_red': "
                f"{category.capitalize()} moving toward RED restricted boundary (speed={speed:.1f}px/f)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="movement_toward_red",
                tier="yellow",
                alert_severity="HIGH",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "movement_toward_red"},
                score_floor=60.0,
            )

        # 2. Medium loitering
        if policy.is_rule_enabled("medium_loitering") and is_medium_loiter:
            reason = (
                f"Zone '{zone_label}' (YELLOW) Rule 'medium_loitering': "
                f"{category.capitalize()} dwelling in buffer strip ({dwell_seconds:.1f}s >= {medium_loiter_threshold:.1f}s)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="medium_loitering",
                tier="yellow",
                alert_severity="HIGH",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "medium_loitering", "threshold": medium_loiter_threshold},
                score_floor=55.0,
            )

        # 3. Wrong direction
        if policy.is_rule_enabled("wrong_direction") and is_wrong_dir:
            reason = (
                f"Zone '{zone_label}' (YELLOW) Rule 'wrong_direction': "
                f"{category.capitalize()} travel direction ({zone_direction}) violates buffer corridor policy"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="wrong_direction",
                tier="yellow",
                alert_severity="HIGH",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "wrong_direction"},
                score_floor=50.0,
            )

        # 4. Repeated suspicious movement
        if policy.is_rule_enabled("repeated_suspicious_movement") and is_repeated_suspicious:
            reason = (
                f"Zone '{zone_label}' (YELLOW) Rule 'repeated_suspicious_movement': "
                f"{category.capitalize()} anomalous movement pattern in buffer zone (speed={speed:.1f}px/f, dir={zone_direction})"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="repeated_suspicious_movement",
                tier="yellow",
                alert_severity="HIGH",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "repeated_suspicious_movement"},
                score_floor=48.0,
            )

        # 5. Buffer zone entry
        if policy.is_rule_enabled("entry"):
            reason = (
                f"Zone '{zone_label}' (YELLOW) Rule 'entry': "
                f"{category.capitalize()} entered sensitive buffer zone (speed={speed:.1f}px/f)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="entry",
                tier="yellow",
                alert_severity="REVIEW",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "entry"},
                score_floor=40.0,
            )

    # =========================================================================
    # GREEN = Normal Policy
    # =========================================================================
    elif zone_tier == "green":
        # 1. Long loitering -> REVIEW
        if policy.is_rule_enabled("long_loitering") and dwell_seconds >= long_loiter_threshold:
            reason = (
                f"Zone '{zone_label}' (GREEN) Rule 'long_loitering': "
                f"{category.capitalize()} extended dwell in normal zone ({dwell_seconds:.1f}s >= {long_loiter_threshold:.1f}s)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="long_loitering",
                tier="yellow",  # Elevates to REVIEW
                alert_severity="REVIEW",
                immediate=False,
                escalated=True,
                reason=reason,
                evidence={**context, "rule": "long_loitering", "threshold": long_loiter_threshold},
                score_floor=38.0,
            )

        # 2. Movement toward YELLOW buffer zone -> Escalation to REVIEW
        # If moving outward/toward yellow perimeter
        if policy.is_rule_enabled("movement_toward_yellow") and zone_direction in ("outward", "inward"):
            reason = (
                f"Zone '{zone_label}' (GREEN) Rule 'movement_toward_yellow': "
                f"{category.capitalize()} moving toward sensitive buffer zone (dir={zone_direction}, speed={speed:.1f}px/f)"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="movement_toward_yellow",
                tier="yellow",
                alert_severity="REVIEW",
                immediate=False,
                escalated=True,
                reason=reason,
                evidence={**context, "rule": "movement_toward_yellow"},
                score_floor=35.0,
            )

        # 3. Abnormal direction -> REVIEW
        if policy.is_rule_enabled("abnormal_direction") and zone_direction == "parallel":
            reason = (
                f"Zone '{zone_label}' (GREEN) Rule 'abnormal_direction': "
                f"{category.capitalize()} atypical parallel movement in green zone"
            )
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="abnormal_direction",
                tier="yellow",
                alert_severity="REVIEW",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "abnormal_direction"},
                score_floor=34.0,
            )

        # 4. Normal movement -> NO ALERT
        if policy.is_rule_enabled("normal_movement"):
            reason = f"Zone '{zone_label}' (GREEN): Normal routine movement"
            return ZonePolicyRuleMatch(
                matched=True,
                rule_name="normal_movement",
                tier="green",
                alert_severity="NONE",
                immediate=False,
                reason=reason,
                evidence={**context, "rule": "normal_movement"},
                score_floor=0.0,
            )

    return None
