"""Learning-phase loop: N episodes with random failures through the
PatchManager, logging the amortization curve.

    pixi run scbt-learn -- --episodes 12 --proposer mock --inject F1 F7 --library /tmp/lib.json
    pixi run scbt-learn -- --episodes 30 --proposer gemini-2.5-flash ...

Each episode: mission ~ U{M1,M2}, failure ~ U{--inject}, trigger ~ U[6,14] s.
The same seed reproduces the same episode sequence for every method.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time

from ..inject.curriculum import make_injection
from ..manager import ManagerConfig, PatchLibrary, PatchManager
from ..sim.config import default_fleet
from ..sim.session import SimSession
from .runner import run_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--inject", nargs="+", default=["F1", "F7"])
    ap.add_argument("--missions", nargs="+", default=["M1", "M2"])
    ap.add_argument("--proposer", default="mock", help="mock | gemini-3.1-pro-preview | gemini-2.5-flash | openai/<model> ...")
    ap.add_argument("--library", default=None, help="JSON path; persists across runs")
    ap.add_argument("--state", default=None, help="bandit/state JSON (default: <library>.state.json)")
    ap.add_argument("--rho", type=float, default=0.80)
    ap.add_argument("--kappa", type=float, default=0.30)
    ap.add_argument("--method", default="scbt", choices=["scbt", "B1", "B2", "B3", "B4", "B5", "B6"],
                    help="baseline preset (overrides the --no-* flags)")
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--no-bandit", action="store_true")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--admit", action="store_true", help="enable admission stages 2-3 (+ CMA-ES) by replay")
    ap.add_argument("--farm-instance", type=int, default=6, help="first runtime instance used by the replay farm")
    ap.add_argument("--farm-size", type=int, default=1, help="number of parallel replay instances")
    ap.add_argument("--K", type=int, default=4, help="replays per admission evaluation")
    ap.add_argument("--tune-budget", type=int, default=0, help="CMA-ES evaluations per patch (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--time-scale", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)

    preset = {"scbt": {}, "B1": {"regenerate": True}, "B2": {"use_library": False}, "B3": {"tune": False},
              "B4": {"use_bandit": False}, "B5": {}, "B6": {"cross_class": False}}[args.method]
    if args.method == "B5":
        args.admit = False                     # static admission only
    if args.method == "B3":
        args.tune_budget = 0
    cfg = ManagerConfig(rho=args.rho, kappa=args.kappa, use_library=not args.no_library,
                        use_bandit=not args.no_bandit, cross_class=not args.no_transfer, seed=args.seed)
    for k, v in preset.items():
        setattr(cfg, k, v)
    farm = None
    admission_fn = None
    if args.admit:
        from ..admit import AdmissionConfig, ReplayFarm
        farm = ReplayFarm(list(range(args.farm_instance, args.farm_instance + args.farm_size)),
                          fleet=default_fleet(1, time_scale=args.time_scale),
                          cfg=AdmissionConfig(K=args.K, tune_budget=args.tune_budget)).start()
        admission_fn = farm.admission_fn
    manager = PatchManager(proposer=args.proposer, library=PatchLibrary(args.library), config=cfg,
                           admission_fn=admission_fn)
    state_path = args.state or (args.library.replace(".json", "") + ".state.json" if args.library else None)
    if state_path and os.path.exists(state_path):
        manager.load_state(state_path)

    rng = random.Random(args.seed)
    session = SimSession(fleet=default_fleet(1, time_scale=args.time_scale), instance=args.instance, lockstep=True)
    session.start()
    rows = []
    try:
        for ep in range(args.episodes):
            mission = rng.choice(args.missions)
            code = rng.choice(args.inject)
            inj = make_injection(code, rng, trigger_t=rng.uniform(6.0, 14.0))
            calls_before = manager.stats.llm_calls
            t0 = time.time()
            if ep > 0:
                session.load_scene()           # fresh vehicle state per episode
            res = asyncio.run(run_episode(session, "Drone1", mission, manager, [inj]))
            row = {"episode": ep, "mission": mission, "failure": code, "success": res.success,
                   "outcome": res.outcome, "value": round(res.mission_value, 2),
                   "events": res.failure_events, "patches": res.patches_applied,
                   "llm_calls": manager.stats.llm_calls - calls_before,
                   "hits_same": manager.stats.library_hits_same, "hits_cross": manager.stats.library_hits_cross,
                   "sim_s": round(res.sim_time_s, 1), "wall_s": round(time.time() - t0, 1),
                   "trace_hash": res.trace_hash[:12]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        session.close()
        if farm is not None:
            farm.close()
    summary = manager.summary()
    summary["replay_evaluations"] = farm.evaluations if farm is not None else 0
    if state_path:
        manager.save_state(state_path)
    summary["cumulative_llm_calls"] = [sum(r["llm_calls"] for r in rows[: i + 1]) for i in range(len(rows))]
    summary["success_rate"] = sum(r["success"] for r in rows) / max(1, len(rows))
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": summary, "args": vars(args)}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
