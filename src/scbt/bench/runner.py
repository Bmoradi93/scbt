"""Episode runner: tick trees for one or more vehicles, inject failures,
intercept failure events, consult a patch policy, hot-patch, and record
metrics.

The *patch policy* is any callable ``policy(event, runner) -> Patch | None``.
Step 1 ships ``scbt.bench.policies`` with a hand-written lookup (stands in
for the Patch Manager) and ``no_patch`` (baseline behaviour: give up ->
abort branch). The learned Patch Manager plugs in here unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import py_trees
from py_trees.common import Status

from ..bt.base_trees import MissionPlan, build_base_tree
from ..bt.leaves import NAMESPACE
from ..bt.registry import find_node, from_xml, iter_nodes, to_xml_string
from ..inject.curriculum import Injection
from ..observer.events import FailureEvent
from ..observer.trace import TickTrace, trace_hash
from ..patch.apply import apply_patch
from ..patch.grammar import Patch
from ..patch.static_check import StaticCheckError, static_check
from ..sim.session import SimSession, Vehicle

LOG = logging.getLogger("scbt")

PatchPolicy = Callable[[FailureEvent, "VehicleRunner"], Optional[Patch]]


@dataclass
class EpisodeResult:
    vehicle: str
    vehicle_class: str
    mission: str
    success: bool                     # all required goals reached and vehicle landed (outcome mission|abort)
    outcome: str                      # "mission" | "abort" | "failed" | "timeout"
    goals_done: List[str]
    mission_value: float              # fraction of goal value achieved (required 1, optional 0.5)
    ticks: int
    sim_time_s: float
    wall_time_s: float
    failure_events: int
    patches_applied: int
    patches_rejected: int
    llm_calls: int
    energy_pct: float                  # battery % consumed
    trace_hash: str
    violations: Dict[str, int] = field(default_factory=dict)   # invariant monitor counts (ticks)
    min_battery_pct: float = 100.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    final_tree_xml: str = ""
    injected: List[str] = field(default_factory=list)


class VehicleRunner:
    """One vehicle, one tree, one tick loop."""

    def __init__(self, session: SimSession, vehicle: Vehicle, plan: MissionPlan,
                 policy: PatchPolicy, injections: Optional[List[Injection]] = None, max_sim_time_s: float = 300.0,
                 k_trace: int = 12, max_patches: int = 3, max_same_node_failures: int = 2):
        self.session = session
        self.vehicle = vehicle
        self.plan = plan
        self.mission = plan.mission
        tree_xml = plan.xml
        self.policy = policy
        self.injections = injections or []
        self.max_sim_time_s = max_sim_time_s
        self.k_trace = k_trace
        self.max_patches = max_patches
        self.max_same_node_failures = max_same_node_failures
        self.applied_patch_ids: List[str] = []

        # Blackboard namespace is per (simulator instance, vehicle): the replay
        # farm runs vehicles with the same names in another instance.
        self.ns = f"/sim{session.instance}/{vehicle.name}"
        token = NAMESPACE.set(self.ns)
        try:
            self.root = from_xml(tree_xml)
        finally:
            NAMESPACE.reset(token)
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.trace = TickTrace(full=False)
        self.tree.add_visitor(self.trace)

        # Blackboard: expose vehicle/session to all leaves of this tree.
        # py_trees blackboards are global per process; namespace by vehicle.
        self.bb = py_trees.blackboard.Client(name=f"runner_{vehicle.name}", namespace=self.ns)
        for key in ("vehicle", "session", "speed", "failure_pending", "give_up", "goals_done",
                    "mission_goals", "required_goals"):
            self.bb.register_key(key, access=py_trees.common.Access.WRITE)
        self.bb.vehicle = vehicle
        self.bb.session = session
        self.bb.speed = 3.0
        self.bb.goals_done = set()
        self.bb.mission_goals = dict(plan.goal_targets)
        self.bb.required_goals = list(plan.required_goals)
        self.bb.failure_pending = False
        self.bb.give_up = False
        self.tree.setup()   # infinite timeout: no signal handler, works off the main thread

        self.mission_node = find_node(self.root, "mission")
        self.events: List[FailureEvent] = []
        self.patches_applied = 0
        self.patches_rejected = 0
        self.llm_calls = 0
        self.t_start = 0.0
        self.battery_start = 100.0
        self.energy_used_pct = 0.0     # integrated per-tick decrements (robust to F7 resets)
        self._batt_prev: Optional[float] = None
        self.violations: Dict[str, int] = {"geofence": 0, "altitude": 0, "speed": 0, "separation": 0}
        self.min_battery_pct = 100.0
        self.record_telemetry = False
        self.telemetry_log: List[Dict[str, Any]] = []   # one snapshot per tick (for offline re-ticking)

    # ----- failure event construction -------------------------------------
    def _make_event(self) -> FailureEvent:
        culprit = self.trace.culprit_leaf()
        tel = dict(self.vehicle.telemetry)
        if culprit is None:
            # Composite failed without a failing leaf this tick (should not
            # happen with our trees); fall back to the mission node.
            culprit = self.mission_node
        path = []
        n = culprit
        while n is not None:
            path.append(n.name)
            n = n.parent
        path.reverse()
        parent = culprit.parent if culprit.parent is not None else culprit
        sub_names = {b.name for b in iter_nodes(parent)}
        subtree_xml = to_xml_string(parent).replace(
            f'name="{culprit.name}"', f'name="{culprit.name}" FAILED="true"', 1
        )
        return FailureEvent(
            vehicle=self.vehicle.name,
            vehicle_class=self.vehicle.spec.name,
            failed_node=culprit.name,
            failed_tag=getattr(culprit, "xml_tag", type(culprit).__name__),
            failed_ports=getattr(culprit, "port_values", lambda: {})(),
            feedback=getattr(culprit, "feedback_message", ""),
            anchor_path=path,
            tick=self.trace.tick,
            t=tel.get("t", 0.0),
            blackboard={k: v for k, v in tel.items() if k not in ("wind",)},
            tick_trace=self.trace.tail(self.k_trace, within=sub_names),
            subtree_xml=subtree_xml,
            injected=tuple(i.code for i in self.injections if i.armed),
        )

    # ----- invariant monitors (admission stage 2) ----------------------------
    def _monitor(self, tel: Dict[str, Any]) -> None:
        g = self.session.geofence
        n, e, d = tel["pos"]
        alt = -d
        if not (g["n_min"] <= n <= g["n_max"] and g["e_min"] <= e <= g["e_max"]):
            self.violations["geofence"] += 1
        if alt > g["alt_max"]:
            self.violations["altitude"] += 1
        if tel["speed"] > self.vehicle.spec.max_speed_mps * 1.3 + 1.0:   # controller overshoot margin
            self.violations["speed"] += 1
        if self.session.separation_violations():
            self.violations["separation"] += 1
        if not tel.get("landed"):
            self.min_battery_pct = min(self.min_battery_pct, b_ if (b_ := tel["battery_pct"]) > 0 else self.min_battery_pct)

    # ----- patch application ---------------------------------------------
    def try_patch(self, patch: Patch, theta: Optional[List[float]] = None) -> bool:
        try:
            static_check(patch, self.root, self.vehicle.spec)
        except StaticCheckError as exc:
            self.patches_rejected += 1
            LOG.warning(f"[{self.vehicle.name}] patch rejected: {exc}")
            return False
        token = NAMESPACE.set(self.ns)
        try:
                apply_patch(self.root, patch, theta, setup_fn=lambda n: n.setup())
        finally:
            NAMESPACE.reset(token)
        self.patches_applied += 1
        self.applied_patch_ids.append(patch.patch_id or "")
        return True

    # ----- main loop --------------------------------------------------------
    async def run(self) -> EpisodeResult:
        s = self.session
        wall0 = time.time()
        s.refresh()
        self.t_start = self.vehicle.telemetry["t"]
        self.battery_start = self.vehicle.telemetry["battery_pct"]
        self._llm_calls_at_start = getattr(getattr(self.policy, "stats", None), "llm_calls", 0)
        outcome = "timeout"
        while True:
            s.refresh()
            tel = self.vehicle.telemetry
            t_rel = tel["t"] - self.t_start
            for inj in self.injections:
                if inj.maybe_fire(s, self.vehicle, t_rel):
                    LOG.info(f"[{self.vehicle.name}] injected {inj.code} {inj.params}")

            b = tel["battery_pct"]
            if self._batt_prev is not None and 0.0 < self._batt_prev - b < 5.0:
                self.energy_used_pct += self._batt_prev - b
            self._batt_prev = b
            self._monitor(tel)

            if self.record_telemetry:
                self.telemetry_log.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in tel.items()})
            self.trace.tick += 1
            self.trace.t = tel["t"]
            self.tree.tick()

            if self.bb.failure_pending:
                self.bb.failure_pending = False
                ev = self._make_event()
                self.events.append(ev)
                same = sum(1 for e in self.events if e.failed_node == ev.failed_node)
                budget_ok = self.patches_applied < self.max_patches and same <= self.max_same_node_failures
                patch = None
                if budget_ok:
                    s.pause()                     # System 2 runs with the world frozen
                    try:
                        patch = self.policy(ev, self)
                    finally:
                        s.resume()
                else:
                    LOG.info(f"[{self.vehicle.name}] patch budget exhausted (patches={self.patches_applied}, same-node={same})")
                if patch is not None and self.try_patch(patch, getattr(patch, "_theta", None)):
                    pass                      # mission restarts on next tick
                else:
                    self.bb.give_up = True

            st = self.root.status
            if st == Status.SUCCESS:
                outcome = "mission" if self.mission_node.status == Status.SUCCESS else "abort"
                break
            if st == Status.FAILURE:
                outcome = "failed"
                break
            if t_rel > self.max_sim_time_s:
                outcome = "timeout"
                self.vehicle.cancel()
                break
            await s.sleep_tick()

        for inj in self.injections:
            inj.clear(s, self.vehicle)
        s.refresh()
        done = set(self.bb.goals_done)
        self.llm_calls = getattr(getattr(self.policy, "stats", None), "llm_calls", 0) - self._llm_calls_at_start
        landed = outcome in ("mission", "abort") and (self.vehicle.telemetry.get("landed") or self.vehicle.telemetry.get("alt", 9) < 0.5)
        success = bool(landed) and all(g in done for g in self.plan.required_goals)
        result = EpisodeResult(
            vehicle=self.vehicle.name,
            vehicle_class=self.vehicle.spec.name,
            mission=self.mission,
            success=success,
            outcome=outcome,
            goals_done=sorted(done),
            mission_value=self.plan.value(done),
            ticks=self.trace.tick,
            sim_time_s=self.vehicle.telemetry["t"] - self.t_start,
            wall_time_s=time.time() - wall0,
            failure_events=len(self.events),
            patches_applied=self.patches_applied,
            patches_rejected=self.patches_rejected,
            llm_calls=self.llm_calls,
            energy_pct=self.energy_used_pct,
            trace_hash=trace_hash(self.trace.records),
            violations=dict(self.violations),
            min_battery_pct=self.min_battery_pct,
            events=[e.__dict__ | {"tick_trace": [r.__dict__ for r in e.tick_trace]} for e in self.events],
            final_tree_xml=to_xml_string(self.root),
            injected=[i.code for i in self.injections],
        )
        hook = getattr(self.policy, "on_episode_end", None)
        if hook is not None:
            hook(result)
        return result


async def run_episode(session: SimSession, vehicle_name: str, mission: str, policy: PatchPolicy,
                      injections: Optional[List[Injection]] = None, plan: Optional[MissionPlan] = None,
                      **mission_kw) -> EpisodeResult:
    v = session.vehicles[vehicle_name]
    plan = plan if plan is not None else build_base_tree(mission, v.spec, v.home_ned, **mission_kw)
    runner = VehicleRunner(session, v, plan, policy, injections)
    return await runner.run()
