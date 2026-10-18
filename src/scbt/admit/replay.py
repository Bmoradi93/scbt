"""Admission stages 2-3 and CMA-ES parameter tuning by simulated replay
(paper Secs. IV-D, IV-F).

A *replay* re-runs the originating episode (same mission, same failure
type) with the candidate patch compiled into the base tree and patching
disabled (the tree behaves as a frozen tree). Injection parameters and
trigger time are perturbed per replay to form the event neighbourhood
N(e). Each replay reports mission success, invariant violations, time and
energy, from which:

  stage 2  (constraint)  any violation in any replay -> reject
  stage 3  (empirical)   success rate >= rho_min and >= baseline + delta
  tuning   (CMA-ES)      maximize J(theta) = success - lam*dt - mu*dE - nu*violations
                         over the patch's tunable box, budget B evaluations

Replaying from mission start (rather than re-spawning at the recorded
state) is slower but faithful: the patched tree must work through the
same takeoff and approach, and sensor/controller state is consistent.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..bt.base_trees import MissionPlan, build_base_tree
from ..bt.leaves import NAMESPACE
from ..bt.registry import from_xml, to_xml
from ..inject.curriculum import RANGES, Injection, make_injection
from ..patch.apply import apply_patch
from ..patch.grammar import Patch
from ..sim.config import FleetSpec, default_fleet
from ..sim.session import SimSession
from ..bench import policies
from ..bench.runner import EpisodeResult, VehicleRunner

LOG = logging.getLogger("scbt.admit")


@dataclass
class AdmissionConfig:
    K: int = 4                 # replays per evaluation
    rho_min: float = 0.75      # required success rate of the patched tree
    delta: float = 0.25        # required margin over the unpatched tree
    perturb: float = 0.20      # +-fraction on injection params, +-2 s trigger
    tune_budget: int = 0       # CMA-ES evaluations (0 -> no tuning, use theta0)
    cma_popsize: int = 4
    lam: float = 0.10
    mu: float = 0.10
    nu: float = 1.00
    max_sim_time_s: float = 300.0


@dataclass
class ReplayStats:
    n: int = 0
    successes: int = 0
    violations: int = 0
    mean_time_s: float = 0.0
    mean_energy_pct: float = 0.0
    min_battery_pct: float = 100.0
    mean_value: float = 0.0
    results: List[dict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0


class ReplayFarm:
    """Owns one (or more) dedicated simulator instances for replays."""

    def __init__(self, instances: List[int], fleet: Optional[FleetSpec] = None, cfg: Optional[AdmissionConfig] = None,
                 time_scale: float = 10.0):
        self.cfg = cfg or AdmissionConfig()
        self.fleet = fleet or default_fleet(1, time_scale=time_scale)
        self.sessions = [SimSession(fleet=self.fleet, instance=i, lockstep=True) for i in instances]
        self._started = False
        self.evaluations = 0

    def start(self) -> "ReplayFarm":
        for s in self.sessions:
            s.start()
        self._started = True
        return self

    def close(self) -> None:
        for s in self.sessions:
            s.close()

    # ------------------------------------------------------------------
    @staticmethod
    def compile_tree(plan: MissionPlan, patch: Optional[Patch], theta: Optional[List[float]], vehicle_name: str) -> ET.Element:
        """Base tree + patch, serialized, with patching disabled."""
        tok = NAMESPACE.set(f"/compile/{vehicle_name}")
        try:
            root = from_xml(copy.deepcopy(plan.xml))
            if patch is not None:
                apply_patch(root, patch, theta)
            el = to_xml(root)
        finally:
            NAMESPACE.reset(tok)
        for mc in el.iter("MissionCritical"):
            mc.set("patching", "false")
        return el

    @staticmethod
    def perturbed(inj: Injection, rng: random.Random, frac: float) -> Injection:
        params = {}
        ranges = RANGES.get(inj.code, {})
        for k, v in inj.params.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                nv = v * rng.uniform(1 - frac, 1 + frac)
                if k in ranges:
                    nv = min(ranges[k][1], max(ranges[k][0], nv))
                params[k] = nv
            else:
                params[k] = v
        return make_injection(inj.code, rng, trigger_t=max(2.0, inj.trigger_t + rng.uniform(-2.0, 2.0)), params=params)

    def _run_one(self, session: SimSession, plan: MissionPlan, tree: ET.Element, inj: Optional[Injection]) -> EpisodeResult:
        """Run one replay on ``session`` (its own asyncio loop on this thread)."""
        session.load_scene()
        v = session.vehicles[self.fleet.vehicles[0].name]
        frozen = MissionPlan(plan.mission, ET.fromstring(ET.tostring(tree)), list(plan.required_goals),
                             list(plan.optional_goals), dict(plan.goal_targets))
        runner = VehicleRunner(session, v, frozen, policies.no_patch, [inj] if inj else [],
                               max_sim_time_s=self.cfg.max_sim_time_s)
        self.evaluations += 1
        return asyncio.run(runner.run())

    def _run_many(self, plan: MissionPlan, tree: ET.Element, injs: List[Optional[Injection]]) -> List[EpisodeResult]:
        """Distribute replays over the farm's simulator instances, one worker
        thread per instance. Admission is called synchronously from inside the
        live episode's event loop, so every replay runs on a worker thread
        with its own loop; results are returned in input order."""
        import concurrent.futures, queue
        free: "queue.Queue[SimSession]" = queue.Queue()
        for s_ in self.sessions:
            free.put(s_)

        def job(inj):
            sess = free.get()
            try:
                return self._run_one(sess, plan, tree, inj)
            finally:
                free.put(sess)

        with concurrent.futures.ThreadPoolExecutor(len(self.sessions), thread_name_prefix="scbt-replay") as ex:
            return list(ex.map(job, injs))

    def evaluate(self, plan: MissionPlan, patch: Optional[Patch], theta: Optional[List[float]], inj: Optional[Injection],
                 seed: int, K: Optional[int] = None) -> ReplayStats:
        """K perturbed replays (or one nominal replay if ``inj`` is None: the
        simulator is deterministic in lockstep, so one suffices)."""
        rng = random.Random(seed)
        vehicle_name = self.fleet.vehicles[0].name
        tree = self.compile_tree(plan, patch, theta, vehicle_name)
        st = ReplayStats()
        if inj is None:
            injs: List[Optional[Injection]] = [None]
        else:
            K = K or self.cfg.K
            injs = [self.perturbed(inj, rng, self.cfg.perturb) for _ in range(K)]
        K = len(injs)
        for pinj, res in zip(injs, self._run_many(plan, tree, injs)):
            viol = sum(res.violations.values())
            st.n += 1
            st.successes += int(res.success)
            st.violations += viol
            st.mean_time_s += res.sim_time_s / K
            st.mean_energy_pct += res.energy_pct / K
            st.min_battery_pct = min(st.min_battery_pct, res.min_battery_pct)
            st.mean_value += res.mission_value / K
            st.results.append({"success": res.success, "outcome": res.outcome, "value": res.mission_value,
                               "violations": res.violations, "min_batt": res.min_battery_pct, "sim_s": res.sim_time_s,
                               "inj": pinj.params if pinj else None, "trigger": pinj.trigger_t if pinj else None})
        return st

    # ------------------------------------------------------------------
    def objective(self, st: ReplayStats) -> float:
        """J(theta): stage-2 invariants (incl. the 10 % landing reserve) enter as
        a penalty so CMA-ES does not trade safety margin for success rate."""
        c = self.cfg
        unsafe = st.violations > 0 or st.min_battery_pct < 10.0
        return st.rate - c.lam * st.mean_time_s / 100.0 - c.mu * st.mean_energy_pct / 10.0 - c.nu * unsafe

    def tune(self, plan: MissionPlan, patch: Patch, inj: Injection, seed: int) -> Tuple[List[float], float]:
        """CMA-ES over the patch's tunable box; returns (theta*, J*)."""
        if patch.dim == 0 or self.cfg.tune_budget <= 0:
            st = self.evaluate(plan, patch, patch.theta0(), inj, seed)
            return patch.theta0(), self.objective(st)
        import cma
        lo = [b[0] for b in patch.bounds()]; hi = [b[1] for b in patch.bounds()]
        x0 = patch.theta0()
        sigma0 = 0.3
        # Optimize in the unit box; CMA-ES needs dim >= 2 -> pad a dummy coordinate.
        def to_unit(x): return [(xi - l) / (h - l) for xi, l, h in zip(x, lo, hi)]
        def from_unit(u): return [l + min(1.0, max(0.0, ui)) * (h - l) for ui, l, h in zip(u, lo, hi)]
        pad = 1 if patch.dim == 1 else 0
        u0 = to_unit(x0) + [0.5] * pad
        es = cma.CMAEvolutionStrategy(u0, sigma0, {"popsize": self.cfg.cma_popsize, "bounds": [0, 1],
                                                    "seed": seed + 1, "verbose": -9, "maxfevals": self.cfg.tune_budget})
        best_theta, best_J = x0, float("-inf")
        evals = 0
        while not es.stop() and evals < self.cfg.tune_budget:
            cands = es.ask()
            cands = cands[: max(1, self.cfg.tune_budget - evals)]
            Js = []
            for u in cands:
                theta = from_unit(u[: patch.dim])
                st = self.evaluate(plan, patch, theta, inj, seed + evals, K=self.cfg.K)
                J = self.objective(st)
                Js.append(-J)
                evals += 1
                if J > best_J:
                    best_J, best_theta = J, theta
            es.tell(cands, Js)
        return best_theta, best_J

    _base_nominal_cache: Dict[str, ReplayStats] = {}

    def _base_nominal(self, plan: MissionPlan, seed: int) -> ReplayStats:
        """Nominal run of the current live tree (no candidate); cached per tree."""
        key = ET.tostring(plan.xml, encoding="unicode")
        if key not in self._base_nominal_cache:
            self._base_nominal_cache[key] = self.evaluate(plan, None, None, None, seed)
        return self._base_nominal_cache[key]

    # ------------------------------------------------------------------
    def admission_fn(self, entry, patch: Patch, event, runner) -> Tuple[bool, str, Optional[List[float]]]:
        """PatchManager hook: tune, then stages 2-3. Returns (ok, reason, theta).
        Replays use the tree *as it currently stands* in the live episode
        (base + previously applied patches), since a candidate may anchor on
        a node introduced by an earlier patch; the baseline is that same tree
        without the candidate."""
        try:
            return self._admission(entry, patch, event, runner)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"admission error for {entry.patch_id}: {exc}")
            return False, f"admission error: {type(exc).__name__}: {exc}", None

    def _admission(self, entry, patch: Patch, event, runner) -> Tuple[bool, str, Optional[List[float]]]:
        live_xml = to_xml(runner.root)
        plan = MissionPlan(runner.plan.mission, live_xml, list(runner.plan.required_goals),
                           list(runner.plan.optional_goals), dict(runner.plan.goal_targets))
        inj = next((i for i in runner.injections if i.armed), None)
        if inj is None:
            return True, "no injection active; static-only", None
        seed = hash((entry.patch_id, event.vehicle)) & 0xFFFF
        theta, J = self.tune(plan, patch, inj, seed)
        patched = self.evaluate(plan, patch, theta, inj, seed)
        baseline = self.evaluate(plan, None, None, inj, seed)      # same perturbations (same seed)
        # Non-regression: the patch must not change nominal behaviour. A
        # ReplaceSubtree that deletes optional goals "recovers" every failure
        # and scores perfectly on required goals; this is where it is caught.
        nominal = self.evaluate(plan, patch, theta, None, seed)
        base_nominal = self._base_nominal(plan, seed)
        c = self.cfg
        record = {"theta": [round(float(t), 3) for t in theta],
                  "nominal_success": nominal.rate, "nominal_value": round(nominal.mean_value, 3),
                  "base_nominal_value": round(base_nominal.mean_value, 3), "J": round(J, 3), "patched_rate": patched.rate, "baseline_rate": baseline.rate,
                  "violations": patched.violations, "min_batt": patched.min_battery_pct, "K": patched.n,
                  "evaluations": self.evaluations}
        entry.provenance["admission_record"] = record
        LOG.info(f"admission {entry.patch_id}: {record}")
        if nominal.violations > 0 or nominal.rate < 1.0 or nominal.mean_value + 1e-6 < base_nominal.mean_value:
            return False, (f"stage3: nominal regression (success {nominal.rate:.0f}, value {nominal.mean_value:.2f} "
                           f"vs base {base_nominal.mean_value:.2f}) ({record})"), theta
        if patched.violations > 0:
            return False, f"stage2: {patched.violations} invariant violation ticks ({record})", theta
        if patched.min_battery_pct < 10.0:
            return False, f"stage2: battery reserve {patched.min_battery_pct:.1f}% < 10% ({record})", theta
        if patched.rate < c.rho_min:
            return False, f"stage3: success rate {patched.rate:.2f} < rho_min {c.rho_min} ({record})", theta
        if patched.rate < baseline.rate + c.delta:
            return False, f"stage3: no margin over unpatched ({patched.rate:.2f} vs {baseline.rate:.2f}) ({record})", theta
        return True, f"admitted ({record})", theta
