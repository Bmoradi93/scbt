"""Benchmark matrix driver (paper Tables I-II, Fig. amortization).

    python -m scbt.bench.matrix --out-dir runs/matrix --methods scbt B2 B3 B4 B5 B6 \\
        --seeds 0 1 2 --episodes 20 --proposer gemini-3.1-pro-preview --farm-size 4

Per (method, seed): learning run -> crystallize -> frozen evaluation on the
curriculum and on held-out variants (+ determinism reps). B0 (expert tree)
and "base" (unpatched) are evaluated in the frozen runner only. B1
(regenerate) has no frozen artifact: its "frozen" row is the learning run
itself with the LLM enabled, as in the paper. Each stage is a subprocess
so a crash in one cell does not lose the others; cells whose output exists
are skipped (re-runnable).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PY = [sys.executable, "-m"]


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", nargs="+", default=["scbt", "B2", "B3", "B4", "B5", "B6"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--ndet", type=int, default=5)
    ap.add_argument("--inject", nargs="+", default=["F1", "F7"])
    ap.add_argument("--proposer", default="gemini-3.1-pro-preview")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tune-budget", type=int, default=4)
    ap.add_argument("--farm-size", type=int, default=4)
    ap.add_argument("--instance", type=int, default=5)
    ap.add_argument("--farm-instance", type=int, default=6)
    ap.add_argument("--skip-reference", action="store_true", help="skip base/B0 frozen rows")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    log = os.path.join(args.out_dir, "matrix.log")
    inj = ["--inject", *args.inject]
    common_frozen = ["--episodes", str(args.test_episodes), "--instance", str(args.instance), *inj]

    # Reference rows (no learning): unpatched base tree and B0 expert tree.
    if not args.skip_reference:
        for name, flag in (("base", "--base"), ("B0", "--expert")):
            for seed in args.seeds:
                for ho in (False, True):
                    out = os.path.join(args.out_dir, f"frozen_{name}_s{seed}{'_heldout' if ho else ''}.json")
                    if os.path.exists(out):
                        continue
                    cmd = PY + ["scbt.bench.frozen", flag, "--seed", str(seed), "--out", out, *common_frozen]
                    if ho:
                        cmd.append("--heldout")
                    elif name == "B0":
                        cmd += ["--ndet", str(args.ndet)]
                    print(f"[matrix] {name} seed {seed} heldout={ho}", flush=True)
                    run(cmd, log)

    for method in args.methods:
        for seed in args.seeds:
            tag = f"{method}_s{seed}"
            lib = os.path.join(args.out_dir, f"lib_{tag}.json")
            learn_out = os.path.join(args.out_dir, f"learn_{tag}.json")
            if not os.path.exists(learn_out):
                print(f"[matrix] learn {tag}", flush=True)
                cmd = PY + ["scbt.bench.learn", "--method", method, "--episodes", str(args.episodes), "--seed", str(seed),
                            "--proposer", args.proposer, "--library", lib, "--out", learn_out, "--admit",
                            "--K", str(args.K), "--tune-budget", str(args.tune_budget), "--farm-size", str(args.farm_size),
                            "--instance", str(args.instance), "--farm-instance", str(args.farm_instance), *inj]
                rc = run(cmd, log)
                if rc != 0:
                    print(f"[matrix] learn {tag} FAILED rc={rc}", flush=True)
                    continue
            if method == "B1":
                continue  # no frozen artifact
            for ho in (False, True):
                out = os.path.join(args.out_dir, f"frozen_{tag}{'_heldout' if ho else ''}.json")
                if os.path.exists(out):
                    continue
                print(f"[matrix] frozen {tag} heldout={ho}", flush=True)
                cmd = PY + ["scbt.bench.frozen", "--library", lib, "--seed", str(seed), "--out", out,
                            "--tree-dir", os.path.join(args.out_dir, f"trees_{tag}"), *common_frozen]
                if ho:
                    cmd.append("--heldout")
                else:
                    cmd += ["--ndet", str(args.ndet)]
                run(cmd, log)
    print("[matrix] done; run: python -m scbt.bench.report --dir", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
