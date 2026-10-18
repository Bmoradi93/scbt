"""B0: hand-coded expert tree with recovery branches for the curriculum.

Written from the failure *descriptions* (Sec. V-A of the paper), before
inspecting any learned library, by an engineer who knows the leaf set:

  F1 wind gust   -> each waypoint leg: Fallback(leg, Sequence(Hover until
                    IsWindOk polls true, retry leg at reduced speed))
  F7 battery     -> guarded flight: Fallback(guarded, degraded mode: lower
                    reserve guard, fly to the next required goal only, return)
  F2 GPS         -> guard: if GPS degraded, hover and wait up to T, then
                    continue (loss is transient in the curriculum)
  F6 no-fly zone -> geofence guard failure: return home (nothing better
                    without a planner)
  F8 link        -> link guard: hover until link recovers, then continue

Parameters are an engineer's reasonable guesses (not tuned). Patching is
disabled on MissionCritical: B0 never learns.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

from ..sim.config import ClassSpec
from .base_trees import MissionPlan, Waypoint, _wp_elems, build_base_tree


def _wind_wrap(leg: ET.Element) -> ET.Element:
    fb = ET.Element("Fallback", {"name": f"{leg.get('name')}_windfb"})
    fb.append(leg)
    rec = ET.SubElement(fb, "Sequence", {"name": f"{leg.get('name')}_windrec"})
    ET.SubElement(rec, "Hover", {"name": f"{leg.get('name')}_hold", "duration_s": "10"})
    retry = ET.SubElement(rec, "Retry", {"name": f"{leg.get('name')}_retry", "num_attempts": "6"})
    seq = ET.SubElement(retry, "Sequence", {"name": f"{leg.get('name')}_poll"})
    ET.SubElement(seq, "WaitSimTime", {"name": f"{leg.get('name')}_wait", "duration_s": "5"})
    ET.SubElement(seq, "IsWindOk", {"name": f"{leg.get('name')}_windok", "max_wind_mps": "8"})
    again = ET.SubElement(seq, "GoToNED", dict(leg.attrib))
    again.set("name", f"{leg.get('name')}_again")
    again.set("velocity", "2.0")
    again.set("timeout_s", "60")
    return fb


def expert_tree(mission: str, cls: ClassSpec, home: Waypoint, **kw) -> MissionPlan:
    base = build_base_tree(mission, cls, home, **kw)
    root = base.xml
    mission_seq = root.find("./MissionCritical/Sequence")
    guarded = mission_seq.find("./ReactiveSequence[@name='guarded_flight']")
    wps = guarded.find("./Sequence[@name='waypoints']")

    # F1: wrap every leg
    legs = list(wps)
    for leg in legs:
        wps.remove(leg)
    for leg in legs:
        wps.append(_wind_wrap(leg))

    # F2 / F8: transient guards that wait rather than fail
    guards = ET.Element("ReactiveSequence", {"name": "guarded_flight"})
    gps = ET.SubElement(guards, "Fallback", {"name": "gps_guard"})
    ET.SubElement(gps, "IsGpsOk", {"name": "gps_ok"})
    gps_wait = ET.SubElement(gps, "Sequence", {"name": "gps_wait"})
    ET.SubElement(gps_wait, "Hover", {"name": "gps_hold", "duration_s": "5"})
    ET.SubElement(gps_wait, "IsGpsOk", {"name": "gps_ok2"})
    link = ET.SubElement(guards, "Fallback", {"name": "link_guard"})
    ET.SubElement(link, "IsLinkOk", {"name": "link_ok", "min_quality": "0.3"})
    link_wait = ET.SubElement(link, "Sequence", {"name": "link_wait"})
    ET.SubElement(link_wait, "Hover", {"name": "link_hold", "duration_s": "5"})
    ET.SubElement(link_wait, "IsLinkOk", {"name": "link_ok2", "min_quality": "0.3"})
    for child in list(guarded):
        guards.append(child)

    # F7: degraded mode around the guarded flight
    fb = ET.Element("Fallback", {"name": "guarded_fb"})
    fb.append(guards)
    degraded = ET.SubElement(fb, "ReactiveSequence", {"name": "degraded"})
    ET.SubElement(degraded, "IsBatteryOk", {"name": "reserve_guard", "min_pct": "12"})
    ET.SubElement(degraded, "IsInsideGeofence", {"name": "geofence_guard2", "margin_m": "0"})
    dseq = ET.SubElement(degraded, "Sequence", {"name": "degraded_seq"})
    ET.SubElement(dseq, "SetSpeed", {"name": "eco_speed", "speed": "4"})
    ET.SubElement(dseq, "GoToRequiredGoal", {"name": "finish_required", "velocity": "4", "timeout_s": "90", "arrive_tol_m": "1.5"})
    ET.SubElement(dseq, "ReturnToLaunch", {"name": "degraded_rtl", "velocity": "4", "timeout_s": "120", "arrive_tol_m": "2"})

    idx = list(mission_seq).index(guarded)
    mission_seq.remove(guarded)
    mission_seq.insert(idx, fb)
    for mc in root.iter("MissionCritical"):
        mc.set("patching", "false")
    return MissionPlan(mission, root, base.required_goals, base.optional_goals, base.goal_targets)
