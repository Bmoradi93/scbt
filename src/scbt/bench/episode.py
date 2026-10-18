"""Run one episode against the headless runtime.

    pixi run scbt-episode -- --mission M1 --inject F1 --policy handwritten --seed 1
    pixi run scbt-episode -- --mission M1 --inject F1 --policy none
    pixi run scbt-episode -- --mission M1                # nominal, no injection

Add ``--no-launch`` to connect to an already running runtime (e.g. one
started with ``pixi run scbt-runtime``). ``--instance N`` selects ports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys

import py_trees

from ..inject.curriculum import make_injection
from ..sim.config import default_fleet
from ..sim.session import SimSession
from . import policies
from .runner import run_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1", choices=["M1", "M2"])
    ap.add_argument("--vehicle", default="Drone1")
    ap.add_argument("--fleet-size", type=int, default=1)
    ap.add_argument("--inject", action="append", default=[], help="failure code, repeatable (F1, F7, ...)")
    ap.add_argument("--trigger", type=float, default=None, help="injection time (s after start)")
    ap.add_argument("--policy", default="handwritten", choices=["handwritten", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--max-sim-time", type=float, default=300.0)
    ap.add_argument("--tick-dt", type=float, default=0.05)
    ap.add_argument("--realtime", action="store_true", help="run the sim in real time instead of lockstep")
    ap.add_argument("--time-scale", type=float, default=10.0, help="sim speed-up (steppable clock pacing)")
    ap.add_argument("--out", default=None, help="write EpisodeResult JSON here")
    ap.add_argument("--print-tree", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    py_trees.logging.level = py_trees.logging.Level.INFO if args.verbose else py_trees.logging.Level.WARN

    rng = random.Random(args.seed)
    injections = [make_injection(code, rng, trigger_t=args.trigger) for code in args.inject]
    policy = policies.handwritten if args.policy == "handwritten" else policies.no_patch

    session = SimSession(fleet=default_fleet(args.fleet_size, time_scale=args.time_scale), instance=args.instance,
                         launch_runtime=not args.no_launch, tick_dt=args.tick_dt,
                         lockstep=not args.realtime)
    session.start()
    try:
        result = asyncio.run(run_episode(session, args.vehicle, args.mission, policy, injections))
    finally:
        session.close()

    summary = {k: v for k, v in result.__dict__.items() if k not in ("events", "final_tree_xml")}
    summary["events"] = [
        {"failed_node": e["failed_node"], "failed_tag": e["failed_tag"], "feedback": e["feedback"],
         "t": round(e["t"], 2), "alt": round(e["blackboard"].get("alt", 0), 2),
         "wind": round(e["blackboard"].get("wind_speed", 0), 2), "battery": round(e["blackboard"].get("battery_pct", 0), 1)}
        for e in result.events
    ]
    print(json.dumps(summary, indent=2))
    if args.print_tree:
        print(result.final_tree_xml)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result.__dict__, f, indent=2, default=str)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
