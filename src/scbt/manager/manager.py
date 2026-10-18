"""PatchManager: System 2 (paper Sec. IV).

    event -> phi(c) -> retrieve(L, rho) -> bandit{stored patches, CALL_LLM}
          -> [CALL_LLM: proposer -> static check -> repair loop -> admission -> store]
          -> patch (+ theta for this class) -> runner hot-patches the tree

Rewards (paper Eq. 4) are computed at episode end from the runner's
result and fed back to the bandit via ``on_episode_end``.

Admission stages 2-3 (simulated replays, CMA-ES) plug in through
``admission_fn``; the default admits on the static check alone and records
that in provenance so results can never be mistaken for fully-admitted ones.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from xml.etree import ElementTree as ET

import numpy as np

from ..observer.events import FailureEvent
from ..patch.grammar import Patch
from ..patch.static_check import StaticCheckError, static_check
from ..sim.config import DEFAULT_CLASSES
from .bandit import CALL_LLM, LinearTS
from .embedding import ContextEmbedder
from .library import LibraryEntry, PatchLibrary
from .proposer import ProposalError, Proposer, make_proposer

LOG = logging.getLogger("scbt.manager")


@dataclass
class ManagerConfig:
    rho: float = 0.80                 # retrieval threshold
    kappa: float = 0.30               # LLM call cost in reward units
    lam: float = 0.10                 # time cost per 100 s of recovery
    mu: float = 0.10                  # energy cost per 10 % battery
    max_repairs: int = 2              # re-prompt attempts after static-check rejection
    examples_k: int = 3
    use_library: bool = True          # False -> B2 (patch without library)
    use_bandit: bool = True           # False -> B4 (nearest neighbour / else LLM)
    cross_class: bool = True          # False -> B6 (per-class libraries)
    regenerate: bool = False          # True -> B1 (whole mission subtree, ReplaceSubtree on mission_seq)
    tune: bool = True                 # False -> B3 (LLM initial values, no CMA-ES)
    max_depth: int = 6
    seed: int = 0


@dataclass
class PendingDecision:
    event_index: int
    arm: str
    phi: np.ndarray
    t: float
    t_rel: float          # sim seconds since mission start
    energy_at: float
    llm_called: bool
    cross_class: bool


@dataclass
class ManagerStats:
    events: int = 0
    llm_calls: int = 0
    llm_latency_s: float = 0.0
    library_hits_same: int = 0
    library_hits_cross: int = 0
    rejected_static: int = 0
    rejected_admission: int = 0
    gave_up: int = 0
    rewards: List[float] = field(default_factory=list)
    decisions: List[dict] = field(default_factory=list)


class PatchManager:
    def __init__(self, proposer: Proposer | str = "mock", library: Optional[PatchLibrary] = None,
                 embedder: Optional[ContextEmbedder] = None, config: Optional[ManagerConfig] = None,
                 admission_fn: Optional[Callable] = None):
        self.cfg = config or ManagerConfig()
        self.proposer = make_proposer(proposer, regenerate=self.cfg.regenerate) if isinstance(proposer, str) else proposer
        if self.cfg.regenerate:
            self.cfg.use_library = False
            self.cfg.max_depth = max(self.cfg.max_depth, 10)
        self.library = library or PatchLibrary()
        self.embed = embedder or ContextEmbedder()
        self.bandit = LinearTS(self.embed.dim, seed=self.cfg.seed)
        self.admission_fn = admission_fn           # (entry, patch, event, runner) -> (ok, reason, theta)
        self.stats = ManagerStats()
        self._pending: List[PendingDecision] = []
        self.frozen = False

    # ------------------------------------------------------------------
    # policy interface used by VehicleRunner
    # ------------------------------------------------------------------
    def __call__(self, event: FailureEvent, runner) -> Optional[Patch]:
        self.stats.events += 1
        event._runner = runner                      # for MockProposer
        cls = runner.vehicle.spec
        phi = self.embed(event)
        phi_r = self.embed.retrieval_part(phi)

        retrieved = []
        if self.cfg.use_library:
            retrieved = self.library.retrieve(phi_r, self.cfg.rho)
            if not self.cfg.cross_class:
                retrieved = [(e, s) for e, s in retrieved if e.origin_class == cls.name]

        # --- choose an arm --------------------------------------------------
        applied = set(getattr(runner, "applied_patch_ids", []))
        candidates = [e.patch_id for e, _ in retrieved if e.patch_id not in applied]
        if self.frozen:
            arm = candidates[0] if candidates else None
            scores = {}
        elif not self.cfg.use_bandit:
            arm = candidates[0] if candidates else CALL_LLM
            scores = {}
        else:
            arm, scores = self.bandit.select(phi, candidates + [CALL_LLM])
        if arm is None:
            self.stats.gave_up += 1
            return None

        patch: Optional[Patch] = None
        entry: Optional[LibraryEntry] = None
        llm_called = False
        if arm == CALL_LLM:
            llm_called = True
            entry = self._call_llm(event, runner, cls, phi, phi_r)
            if entry is None and candidates:
                arm = candidates[0]                  # fall back to best stored patch
                entry = self.library.get(arm)
            elif entry is not None:
                arm = entry.patch_id
        else:
            entry = self.library.get(arm)

        if entry is None:
            self.stats.gave_up += 1
            self._record(event, arm, phi, llm_called, cross=False, runner=runner, gave_up=True)
            return None

        patch = entry.patch()
        cross = cls.name not in entry.admitted_for and entry.origin_class != cls.name
        theta = entry.theta.get(cls.name) or entry.theta.get(entry.origin_class) or patch.theta0()
        patch._theta = theta                         # consumed by runner.try_patch
        entry.uses += 1
        if not llm_called:
            if cross:
                self.stats.library_hits_cross += 1
            else:
                self.stats.library_hits_same += 1
        self._record(event, arm, phi, llm_called, cross, runner)
        return patch

    def _record(self, event, arm, phi, llm_called, cross, runner, gave_up=False):
        self._pending.append(PendingDecision(
            event_index=len(runner.events) - 1, arm=arm, phi=phi, t=event.t,
            t_rel=event.t - getattr(runner, "t_start", 0.0), energy_at=runner.energy_used_pct, llm_called=llm_called, cross_class=cross))
        self.stats.decisions.append({"vehicle": event.vehicle, "failed_node": event.failed_node,
                                     "arm": arm, "llm": llm_called, "cross": cross, "gave_up": gave_up})

    # ------------------------------------------------------------------
    def _call_llm(self, event, runner, cls, phi, phi_r) -> Optional[LibraryEntry]:
        examples = [(e.context_text(), e.patch_xml,
                     f"{e.status}; used {e.uses}x, {e.successes} successful"
                     + (f"; rejected: {e.reject_reason}" if e.reject_reason else ""))
                    for e, _ in self.library.nearest(phi_r, self.cfg.examples_k)]
        plan_hint = f"required {runner.plan.required_goals}, optional {runner.plan.optional_goals}, done {sorted(runner.bb.goals_done)}"
        feedback = None
        t0 = time.time()
        proposal = patch = None
        attempts = 0
        for attempt in range(1 + self.cfg.max_repairs):
            attempts = attempt + 1
            try:
                proposal = self.proposer.propose(event, cls, examples, feedback, plan_hint)
                self.stats.llm_calls += 1
            except ProposalError as exc:
                LOG.warning(f"proposer failed: {exc}")
                self.stats.llm_calls += 1
                proposal = None
                break
            pid = self.library.new_id()
            try:
                patch = proposal.to_patch(pid)
                warnings = static_check(patch, runner.root, cls, max_depth=self.cfg.max_depth)
                break
            except (StaticCheckError, ValueError) as exc:
                self.stats.rejected_static += 1
                feedback = str(exc)
                LOG.info(f"static check rejected attempt {attempt + 1}: {exc}")
                self.library.add(LibraryEntry(
                    patch_id=pid, patch_xml=patch.to_xml_string() if patch else proposal.subtree_xml,
                    phi=phi.tolist(), phi_retrieval=phi_r.tolist(), origin_class=cls.name,
                    status="rejected", reject_reason=f"static: {exc}",
                    provenance={"context_text": event.context_text(), "rationale": proposal.rationale,
                                "proposer": self.proposer.name, "attempt": attempt + 1}))
                patch = None
        self.stats.llm_latency_s += time.time() - t0
        if patch is None:
            return None

        entry = LibraryEntry(
            patch_id=patch.patch_id, patch_xml=patch.to_xml_string(), phi=phi.tolist(),
            phi_retrieval=phi_r.tolist(), origin_class=cls.name,
            provenance={"context_text": event.context_text(), "rationale": patch.rationale,
                        "proposer": self.proposer.name, "attempts": attempts,
                        "event": {"vehicle": event.vehicle, "failed_node": event.failed_node, "t": event.t,
                                  "injected": list(event.injected)},
                        "static_warnings": warnings})
        # Admission stages 2-3
        if self.admission_fn is not None:
            ok, reason, theta = self.admission_fn(entry, patch, event, runner)
            entry.provenance["admission"] = reason
            LOG.info(f"{'ADMIT' if ok else 'REJECT'} {entry.patch_id}: {reason[:300]}")
            if not ok:
                entry.status = "rejected"; entry.reject_reason = f"admission: {reason}"
                self.stats.rejected_admission += 1
                self.library.add(entry)
                return None
            if theta is not None:
                entry.theta[cls.name] = list(theta)
        else:
            entry.provenance["admission"] = "static_only"
        entry.status = "admitted"
        entry.admitted_for.append(cls.name)
        self.library.add(entry)
        self.bandit.ensure_arm(entry.patch_id)
        return entry

    # ------------------------------------------------------------------
    # reward feedback (paper Eq. 4)
    # ------------------------------------------------------------------
    def on_episode_end(self, result) -> None:
        for d in self._pending:
            dt = max(0.0, result.sim_time_s - d.t_rel)
            dE = max(0.0, result.energy_pct - d.energy_at)
            r = (1.0 if result.success else 0.0) - self.cfg.lam * dt / 100.0 - self.cfg.mu * dE / 10.0 \
                - (self.cfg.kappa if d.llm_called else 0.0)
            arm_for_update = CALL_LLM if d.llm_called else d.arm
            self.bandit.update(arm_for_update, d.phi, r)
            if d.llm_called and d.arm != CALL_LLM:
                # The produced patch also gets credit as its own arm.
                self.bandit.update(d.arm, d.phi, r + self.cfg.kappa)
            if d.arm in self.library.entries:
                e = self.library.entries[d.arm]
                if result.success:
                    e.successes += 1
            self.stats.rewards.append(r)
        self._pending.clear()
        self.library.save()

    # ------------------------------------------------------------------
    def save_state(self, path: str) -> None:
        """Persist bandit posteriors + stats next to the library."""
        import json, os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"bandit": self.bandit.to_dict(), "config": self.cfg.__dict__,
                       "stats": self.summary(), "decisions": self.stats.decisions}, f, indent=1)

    def load_state(self, path: str) -> None:
        import json
        with open(path) as f:
            d = json.load(f)
        self.bandit = LinearTS.from_dict(d["bandit"], seed=self.cfg.seed)

    def freeze(self) -> None:
        self.frozen = True
        self.bandit.frozen = True

    def summary(self) -> dict:
        s = self.stats
        return {"events": s.events, "llm_calls": s.llm_calls, "llm_latency_s": round(s.llm_latency_s, 2),
                "hits_same": s.library_hits_same, "hits_cross": s.library_hits_cross,
                "rejected_static": s.rejected_static, "rejected_admission": s.rejected_admission,
                "gave_up": s.gave_up, "library": self.library.stats(),
                "mean_reward": round(float(np.mean(s.rewards)), 3) if s.rewards else None}
