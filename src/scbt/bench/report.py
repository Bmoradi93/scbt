"""Aggregate matrix outputs into the paper's tables and the amortization figure.

    python -m scbt.bench.report --dir runs/matrix
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    d = args.dir

    # ---- learning phase -------------------------------------------------
    learn = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, "learn_*_s*.json"))):
        m = re.match(r"learn_(.+)_s(\d+)\.json", os.path.basename(p)).group(1)
        learn[m].append(load(p))
    print("\n== Table I: learning phase (mean ± sd over seeds) ==")
    print(f"{'method':8s} {'seeds':>5s} {'success':>12s} {'value':>12s} {'LLM calls':>12s} {'calls/ep last half':>20s} {'admit/rej':>10s} {'hits':>6s}")
    curves = {}
    for m, runs in learn.items():
        succ = [r["summary"]["success_rate"] for r in runs]
        val = [np.mean([x["value"] for x in r["rows"]]) for r in runs]
        calls = [r["summary"]["llm_calls"] for r in runs]
        half = [np.mean([x["llm_calls"] for x in r["rows"][len(r["rows"]) // 2:]]) for r in runs]
        adm = [r["summary"]["library"].get("admitted", 0) for r in runs]
        rej = [r["summary"]["library"].get("rejected", 0) for r in runs]
        hits = [r["summary"]["hits_same"] + r["summary"]["hits_cross"] for r in runs]
        curves[m] = np.mean([r["summary"]["cumulative_llm_calls"] for r in runs], axis=0)
        print(f"{m:8s} {len(runs):5d} {np.mean(succ):6.2f}±{np.std(succ):4.2f} {np.mean(val):6.2f}±{np.std(val):4.2f} "
              f"{np.mean(calls):6.1f}±{np.std(calls):4.1f} {np.mean(half):10.2f}          {np.mean(adm):4.1f}/{np.mean(rej):4.1f} {np.mean(hits):5.1f}")

    # ---- frozen phase ---------------------------------------------------
    frozen = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(d, "frozen_*_s*.json"))):
        b = os.path.basename(p)
        mm = re.match(r"frozen_(.+)_s(\d+)(_heldout)?\.json", b)
        frozen[mm.group(1)].setdefault("heldout" if mm.group(3) else "curric", []).append(load(p))
    print("\n== Table II: frozen phase (learning disabled) ==")
    print(f"{'method':8s} {'curric succ':>12s} {'curric value':>13s} {'heldout succ':>13s} {'heldout value':>14s} "
          f"{'e2e distinct':>13s} {'tick spread':>11s} {'offline distinct':>16s} {'nodes M1/M2':>12s}")
    for m, dd in frozen.items():
        def agg(key, field):
            xs = [r["summary"][field] for r in dd.get(key, [])]
            return f"{np.mean(xs):6.2f}±{np.std(xs):4.2f}" if xs else "   n/a    "
        dets_ = [r["summary"]["determinism"] for r in dd.get("curric", []) if r["summary"].get("determinism")]
        if dets_:
            dets = f"{max(x['distinct_hashes'] for x in dets_)} of {dets_[0]['reps']}"
            spread = f"{max((x.get('max_tick_spread') or 0) for x in dets_)}"
            off = [x.get("offline") for x in dets_ if x.get("offline")]
            offline = (f"{max(o['max_distinct_per_log'] for o in off)} of {off[0]['reticks_per_log']} "
                       f"(faithful {sum(o['online_equals_offline'] for o in off)}/{sum(o['logs'] for o in off)})") if off else "n/a"
        else:
            dets = spread = offline = "n/a"
        nodes = ""
        for tp in sorted(glob.glob(os.path.join(d, f"trees_{m}_s*", "crystallize_*_M1.json")))[:1]:
            c1 = load(tp); c2 = load(tp.replace("_M1", "_M2"))
            nodes = f"{c1['nodes']}/{c2['nodes']} ({len(c1['applied'])}+{len(c2['applied'])} patches)"
        print(f"{m:8s} {agg('curric','success_rate')} {agg('curric','mean_value')}  {agg('heldout','success_rate')}  {agg('heldout','mean_value')}  "
              f"{dets:>13s} {spread:>11s} {offline:>16s} {nodes}")

    # ---- amortization figure --------------------------------------------
    if curves:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4.2, 2.8))
            for m, c in curves.items():
                ax.plot(range(1, len(c) + 1), c, marker="o", ms=3, label=m)
            ax.set_xlabel("episode"); ax.set_ylabel("cumulative LLM calls"); ax.grid(alpha=.3); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(d, "amortization.pdf")); fig.savefig(os.path.join(d, "amortization.png"), dpi=160)
            print(f"\nfigure: {os.path.join(d, 'amortization.pdf')}")
        except Exception as exc:  # noqa: BLE001
            print("figure skipped:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
