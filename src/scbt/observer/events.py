"""Failure event e = (v_f, c) (paper Def. 1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class TickRecord:
    tick: int
    t: float
    node: str
    tag: str
    status: str  # SUCCESS | FAILURE | RUNNING | INVALID


@dataclass
class FailureEvent:
    vehicle: str
    vehicle_class: str
    failed_node: str            # v_f: name of the culprit leaf
    failed_tag: str             # its node type
    failed_ports: Dict[str, Any]
    feedback: str               # leaf feedback_message
    anchor_path: List[str]      # names from root to v_f
    tick: int
    t: float
    blackboard: Dict[str, Any]  # b: telemetry snapshot
    tick_trace: List[TickRecord]  # tau: last k records of the subtree
    scene_description: str = ""  # s: textual context (telemetry-derived in headless)
    subtree_xml: str = ""        # XML of the anchor's parent subtree, culprit marked
    injected: Tuple[str, ...] = field(default_factory=tuple)  # ground-truth labels (for metrics only)

    def context_text(self) -> str:
        """Scene description used for embedding. Headless mode: derived from
        telemetry, never from the injected-failure label."""
        if self.scene_description:
            return self.scene_description
        b = self.blackboard
        # The vehicle class is deliberately NOT part of the text: retrieval
        # must surface cross-class candidates (paper Sec. IV-C).
        parts = [
            f"node {self.failed_tag} '{self.failed_node}' failed",
            f"feedback: {self.feedback}" if self.feedback else "",
            f"altitude {b.get('alt', 0):.1f} m",
            f"speed {b.get('speed', 0):.1f} m/s",
            f"wind {b.get('wind_speed', 0):.1f} m/s",
            f"battery {b.get('battery_pct', 0):.0f}%",
            "gps ok" if b.get("gps_ok", True) else "gps degraded",
            f"link quality {b.get('link_quality', 1.0):.2f}",
        ]
        return "; ".join(p for p in parts if p)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, default=str)
