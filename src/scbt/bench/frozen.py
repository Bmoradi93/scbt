"""Frozen-phase evaluation (paper Sec. V-C): crystallized trees, no learning.

    python -m scbt.bench.frozen --library runs/lib.json --episodes 20 --inject F1 F7 \\
        [--heldout] [--ndet 5] [--seed 0] [--out runs/frozen.json] [--tree-dir runs/frozen]

No PatchManager, replay farm, LLM or bandit object is created in this
process. Per episode the crystallized tree for (class, mission) is built
at the vehicle's actual home, ticked with MissionCritical patching off, and
scored exactly like the learning phase. ``--ndet N`` additionally repeats
the first scenario N times with the same seed and reports the number of
distinct tick-trace hashes (1 = deterministic). ``--base`` evaluates the
unpatched base tree instead (the B0-less lower bound).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import xml.etree.ElementTree as ET
from typing import Optional

from ..bt.base_trees import MissionPlan, build_base_tree
from ..crystallize import crystallize
from ..inject.curriculum import make_injection
from ..manager.bandit import LinearTS
from ..manager.library import PatchLibrary
from ..sim.config import default_fleet
from ..sim.session import SimSession
from . import policies
from .runner import run_episode


def frozen_plan(lib: Optional[PatchLibrary], bandit: Optional[LinearTS], cls_name: str, mission: str, home,
                expert: bool = False) -> MissionPlan:
    from ..sim.config import DEFAULT_CLASSES
    if expert:
        from ..bt.expert_tree import expert_tree
        return expert_tree(mission, DEFAULT_CLASSES[cls_name], home)
    base = build_base_tree(mission, DEFAULT_CLASSES[cls_name], home)
    if lib is None:
        el = ET.fromstring(ET.tostring(base.xml))
        for mc in el.iter("MissionCritical"):
            mc.set("patching", "false")
        return MissionPlan(mission, el, base.required_goals, base.optional_goals, base.goal_targets)
    el, _info = crystallize(lib, cls_name, mission, bandit, home=home)
    return MissionPlan(mission, el, base.required_goals, base.optional_goals, base.goal_targets)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=None, help="library JSON (omit with --base)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--base", action="store_true", help="evaluate the unpatched base tree")
    ap.add_argument("--expert", action="store_true", help="evaluate the hand-coded expert tree (B0)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--inject", nargs="+", default=["F1", "F7"])
    ap.add_argument("--missions", nargs="+", default=["M1", "M2"])
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--ndet", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--time-scale", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tree-dir", default=None, help="also write the crystallized XML used (canonical home)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)

    lib = bandit = None
    if not args.base and not args.expert:
        if not args.library:
            ap.error("--library required unless --base")
        lib = PatchLibrary(args.library)
        state = args.state or args.library.replace(".json", "") + ".state.json"
        if os.path.exists(state):
            with open(state) as f:
                bandit = LinearTS.from_dict(json.load(f)["bandit"])
            bandit.frozen = True

    rng = random.Random(args.seed)
    session = SimSession(fleet=default_fleet(1, time_scale=args.time_scale), instance=args.instance, lockstep=True)
    session.start()
    rows, det_hashes = [], []
    offline = None
    try:
        v = session.vehicles["Drone1"]
        if args.tree_dir and lib is not None:
            os.makedirs(args.tree_dir, exist_ok=True)
            for m in args.missions:
                el, info = crystallize(lib, v.spec.name, m, bandit)
                ET.ElementTree(el).write(os.path.join(args.tree_dir, f"T_star_{v.spec.name}_{m}.xml"), encoding="unicode")
                with open(os.path.join(args.tree_dir, f"crystallize_{v.spec.name}_{m}.json"), "w") as f:
                    json.dump(info, f, indent=1)
        scenarios = []
        for ep in range(args.episodes):
            mission = rng.choice(args.missions)
            code = rng.choice(args.inject)
            inj = make_injection(code, rng, trigger_t=rng.uniform(6.0, 14.0), heldout=args.heldout)
            scenarios.append((mission, code, inj))

        det_traces = []
        det_logs = []

        def run_scenario(mission, inj, keep_trace=False):
            from .runner import VehicleRunner
            session.load_scene()
            v = session.vehicles["Drone1"]
            plan = frozen_plan(lib, bandit, v.spec.name, mission, v.home_ned, expert=args.expert)
            inj.armed = False
            runner = VehicleRunner(session, v, plan, policies.no_patch, [inj])
            runner.record_telemetry = keep_trace
            res = asyncio.run(runner.run())
            if keep_trace:
                det_traces.append([(r.tick, r.node, r.status) for r in runner.trace.records])
                for t_ in runner.telemetry_log:
                    t_["home_ned"] = list(v.home_ned)
                det_logs.append((plan, runner.telemetry_log, res.trace_hash, res.ticks))
            return res

        for ep, (mission, code, inj) in enumerate(scenarios):
            res = run_scenario(mission, inj)
            row = {"episode": ep, "mission": mission, "failure": code, "heldout": args.heldout, "success": res.success,
                   "outcome": res.outcome, "value": round(res.mission_value, 2), "events": res.failure_events,
                   "patches": res.patches_applied, "llm_calls": 0, "violations": res.violations,
                   "min_batt": round(res.min_battery_pct, 1), "sim_s": round(res.sim_time_s, 1),
                   "trace_hash": res.trace_hash[:16], "params": inj.params, "trigger": round(inj.trigger_t, 2)}
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)

        if args.ndet > 0 and scenarios:
            mission, code, inj = scenarios[0]
            for i in range(args.ndet):
                inj2 = make_injection(code, random.Random(0), trigger_t=inj.trigger_t, params=inj.params)
                res = run_scenario(mission, inj2, keep_trace=True)
                det_hashes.append(res.trace_hash)
                print(json.dumps({"det_rep": i, "trace_hash": res.trace_hash[:16], "ticks": res.ticks}), flush=True)
    finally:
        session.close()

    # Offline decision determinism: re-tick each recorded log N times.
    if det_logs:
        from .replay_det import replay
        fleet = default_fleet(1)
        spec = fleet.classes[fleet.vehicles[0].vehicle_class]
        faithful, max_distinct = 0, 0
        for plan, log, online_hash, _ticks in det_logs:
            hs = [replay(plan.xml, plan, spec, log, session.geofence)[0] for _ in range(3)]
            max_distinct = max(max_distinct, len(set(hs)))
            faithful += int(hs[0] == online_hash)
        offline = {"logs": len(det_logs), "reticks_per_log": 3, "max_distinct_per_log": max_distinct,
                   "online_equals_offline": faithful}
        print(json.dumps({"offline_determinism": offline}), flush=True)
    by_f = {}
    for r in rows:
        d = by_f.setdefault(r["failure"], {"n": 0, "success": 0, "value": 0.0, "aborts": 0})
        d["n"] += 1; d["success"] += int(r["success"]); d["value"] += r["value"]; d["aborts"] += int(r["outcome"] == "abort")
    summary = {"method": "base" if args.base else ("expert" if args.expert else "frozen"),
               "heldout": args.heldout, "episodes": len(rows),
               "success_rate": sum(r["success"] for r in rows) / max(1, len(rows)),
               "mean_value": sum(r["value"] for r in rows) / max(1, len(rows)),
               "by_failure": {k: {**d, "value": round(d["value"] / d["n"], 3)} for k, d in by_f.items()},
               "determinism": ({"reps": len(det_hashes), "distinct_hashes": len(set(det_hashes)),
                                "max_tick_spread": (max(t for _, _, _, t in det_logs) - min(t for _, _, _, t in det_logs)) if det_logs else None,
                                "offline": offline} if det_hashes else None),
               "llm_calls": 0}
    print(json.dumps(summary, indent=2))
    if det_hashes and len(set(det_hashes)) > 1:
        base = det_traces[0]
        for i, tr in enumerate(det_traces[1:], 1):
            if tr == base:
                continue
            for j, (a, b) in enumerate(zip(base, tr)):
                if a != b:
                    print(f"det rep {i} diverges at record {j}: rep0 {base[max(0,j-1):j+2]} | rep{i} {tr[max(0,j-1):j+2]}")
                    break
            else:
                print(f"det rep {i}: equal prefix, lengths {len(base)} vs {len(tr)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": summary, "args": vars(args), "det_hashes": det_hashes}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
