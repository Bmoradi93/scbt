"""Decision-trace determinism under replayed telemetry.

The simulator's command-execution thread is wall-clock driven, so two
end-to-end runs of the same scenario can differ at the 1e-4 m level even
with no BT in the loop (measured; see README). That jitter occasionally
flips a threshold decision by one tick. The claim that matters for the
control logic is narrower and exact: *given the same observation sequence,
the frozen tree makes the same decisions*. This module re-ticks a tree
offline against a recorded per-tick telemetry log with a fake vehicle whose
commands are no-ops (all action leaves complete by telemetry), and returns
the trace hash. It is used (a) to show the offline hash of a run equals its
online hash (the fake is faithful) and (b) that N offline re-ticks of the
same log give 1 distinct hash.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import py_trees

from ..bt.base_trees import MissionPlan
from ..bt.leaves import NAMESPACE
from ..bt.registry import from_xml, find_node
from ..observer.trace import TickTrace, trace_hash
from ..sim.config import ClassSpec


class _NeverDone:
    """Awaitable returning a task that never completes (commands are no-ops)."""

    def __call__(self, *a, **k):
        async def _coro():
            loop = asyncio.get_running_loop()
            return loop.create_future()   # never resolved
        return _coro()


class FakeVehicle:
    def __init__(self, spec: ClassSpec, name: str, home_ned):
        self.spec, self.name, self.home_ned = spec, name, tuple(home_ned)
        self.telemetry: Dict[str, Any] = {}
        self.takeoff = self.land = self.go_home = self.hover = self.move_to_ned = _NeverDone()
        self.link_quality = 1.0; self.gps_ok = True; self.gps_noise_m = 0.0

    @property
    def ned(self):
        return tuple(self.telemetry.get("pos", (0, 0, 0)))

    def dist_to(self, n, e, d):
        pn, pe, pd = self.ned
        return ((pn - n) ** 2 + (pe - e) ** 2 + (pd - d) ** 2) ** 0.5

    def cancel(self):
        pass


class FakeSession:
    def __init__(self, geofence: Dict[str, float], instance: int):
        self.geofence = geofence; self.instance = instance; self.pending_starts: list = []
        self.lockstep = True

    def separation_violations(self):
        return []


def replay(tree_xml: ET.Element, plan: MissionPlan, spec: ClassSpec, telemetry_log: List[Dict[str, Any]],
           geofence: Dict[str, float], vehicle_name: str = "Drone1", instance: int = 99) -> Tuple[str, int]:
    """Tick ``tree_xml`` against ``telemetry_log``; return (trace_hash, ticks)."""
    ns = f"/replay{instance}/{vehicle_name}"
    v = FakeVehicle(spec, vehicle_name, telemetry_log[0].get("pos", (0, 0, 0)))
    sess = FakeSession(geofence, instance)

    async def run():
        tok = NAMESPACE.set(ns)
        try:
            root = from_xml(ET.fromstring(ET.tostring(tree_xml)))
        finally:
            NAMESPACE.reset(tok)
        bb = py_trees.blackboard.Client(name=f"replay_{vehicle_name}", namespace=ns)
        for key in ("vehicle", "session", "speed", "failure_pending", "give_up", "goals_done", "mission_goals", "required_goals"):
            bb.register_key(key, access=py_trees.common.Access.WRITE)
        bb.vehicle = v; bb.session = sess; bb.speed = 3.0; bb.failure_pending = False; bb.give_up = False
        bb.goals_done = set(); bb.mission_goals = dict(plan.goal_targets); bb.required_goals = list(plan.required_goals)
        tree = py_trees.trees.BehaviourTree(root)
        tr = TickTrace(); tree.add_visitor(tr); tree.setup()
        v.home_ned = tuple(telemetry_log[0].get("home_ned", v.home_ned))
        for i, tel in enumerate(telemetry_log):
            v.telemetry = {k: (tuple(x) if isinstance(x, list) else x) for k, x in tel.items()}
            tr.tick = i + 1; tr.t = tel["t"]
            tree.tick()
            await asyncio.sleep(0)
            if root.status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
                break
        return trace_hash(tr.records), tr.tick

    return asyncio.run(run())
