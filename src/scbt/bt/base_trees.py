"""Hand-written base trees T_m (paper Sec. V-B).

Structure (identical for all classes; only waypoints/limits differ):

    Fallback root
    ├── MissionCritical mission            <- failure events are raised here
    │   └── Sequence
    │       ├── Takeoff
    │       ├── ReactiveSequence guarded_flight
    │       │   ├── IsBatteryOk  battery_guard
    │       │   ├── IsInsideGeofence geofence_guard
    │       │   └── Sequence waypoints
    │       │       ├── GoToNED wp1
    │       │       └── ...
    │       └── Land
    └── Sequence abort                      <- hand-coded safe fallback
        ├── ReturnToLaunch
        └── Land

The base trees are deliberately *minimal*: they succeed in nominal
conditions and expose mission-critical failure at ``mission``. Recovery
logic is what SCBT learns; baseline B0 adds it by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from ..sim.config import ClassSpec

Waypoint = Tuple[float, float, float]  # NED


@dataclass
class MissionPlan:
    """Base tree plus the goal structure the success metric is defined on.

    required_goals: all must be reached for mission success.
    optional_goals: contribute to mission value; may be skipped by a recovery.
    """
    mission: str
    xml: ET.Element
    required_goals: List[str]
    optional_goals: List[str] = field(default_factory=list)
    goal_targets: Dict[str, Waypoint] = field(default_factory=dict)   # goal_id -> NED target

    def value(self, done: set) -> float:
        """Fraction of goal value achieved (required weight 1, optional 0.5)."""
        tot = len(self.required_goals) + 0.5 * len(self.optional_goals)
        got = sum(1 for g in self.required_goals if g in done) + 0.5 * sum(1 for g in self.optional_goals if g in done)
        return got / tot if tot else 1.0


def _wp_elems(wps: List[Tuple[Waypoint, Optional[str]]], alt_m: float, tol: float) -> List[ET.Element]:
    out = []
    for i, ((n, e, _d), goal) in enumerate(wps, 1):
        attrs = {"name": f"wp{i}", "north": f"{n}", "east": f"{e}", "down": f"{-alt_m}",
                 "arrive_tol_m": f"{tol}", "timeout_s": "40"}
        if goal:
            attrs["goal_id"] = goal
        out.append(ET.Element("GoToNED", attrs))
    return out


def _skeleton(cls: ClassSpec, wps: List[Tuple[Waypoint, Optional[str]]], alt_m: float, min_batt_pct: float = 25.0) -> ET.Element:
    root = ET.Element("Fallback", {"name": "root"})

    mission = ET.SubElement(root, "MissionCritical", {"name": "mission"})
    seq = ET.SubElement(mission, "Sequence", {"name": "mission_seq"})
    ET.SubElement(seq, "Takeoff", {"name": "takeoff", "timeout_s": "20", "min_alt_m": "1.5"})
    ET.SubElement(seq, "SetSpeed", {"name": "cruise", "speed": f"{min(3.0, cls.max_speed_mps)}"})
    guarded = ET.SubElement(seq, "ReactiveSequence", {"name": "guarded_flight"})
    ET.SubElement(guarded, "IsBatteryOk", {"name": "battery_guard", "min_pct": f"{min_batt_pct}"})
    ET.SubElement(guarded, "IsInsideGeofence", {"name": "geofence_guard", "margin_m": "0"})
    wpseq = ET.SubElement(guarded, "Sequence", {"name": "waypoints"})
    for e in _wp_elems(wps, alt_m, tol=1.5):
        wpseq.append(e)
    ET.SubElement(seq, "Land", {"name": "land", "timeout_s": "60"})

    abort = ET.SubElement(root, "Sequence", {"name": "abort"})
    ET.SubElement(abort, "ReturnToLaunch", {"name": "rtl", "velocity": f"{min(3.0, cls.max_speed_mps)}"})
    ET.SubElement(abort, "Land", {"name": "abort_land", "timeout_s": "60"})
    return root


# ---- mission templates ----------------------------------------------------

def m1_transport(cls: ClassSpec, home: Waypoint, length_m: float = 40.0, alt_m: float = 8.0) -> MissionPlan:
    """M1: deliver at ``length_m`` north (required goal), return, land."""
    n0, e0, _ = home
    wps = [((n0 + length_m, e0, 0.0), "deliver"), ((n0, e0, 0.0), None)]
    return MissionPlan("M1", _skeleton(cls, wps, alt_m), required_goals=["deliver"],
                       goal_targets={"deliver": (n0 + length_m, e0, -alt_m)})


def m2_survey(cls: ClassSpec, home: Waypoint, size_m: float = 30.0, lanes: int = 3, alt_m: float = 10.0) -> MissionPlan:
    """M2: lawn-mower survey. Lane 1 is the priority area (required); the
    remaining lanes are optional and may be skipped under energy pressure."""
    n0, e0, _ = home
    wps: List[Tuple[Waypoint, Optional[str]]] = []
    spacing = size_m / max(lanes - 1, 1)
    goals = []
    for i in range(lanes):
        e = e0 + i * spacing
        g = f"lane{i + 1}"
        goals.append(g)
        if i % 2 == 0:
            wps += [((n0, e, 0.0), None), ((n0 + size_m, e, 0.0), g)]
        else:
            wps += [((n0 + size_m, e, 0.0), None), ((n0, e, 0.0), g)]
    wps.append(((n0, e0, 0.0), None))
    targets = {g: (wp[0], wp[1], -alt_m) for wp, g in wps if g}
    return MissionPlan("M2", _skeleton(cls, wps, alt_m), required_goals=goals[:1], optional_goals=goals[1:],
                       goal_targets=targets)


MISSION_TEMPLATES: Dict[str, Callable[..., MissionPlan]] = {
    "M1": m1_transport,
    "M2": m2_survey,
}


def build_base_tree(mission: str, cls: ClassSpec, home: Waypoint, **kw) -> MissionPlan:
    return MISSION_TEMPLATES[mission](cls, home, **kw)
