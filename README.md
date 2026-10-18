# SCBT: Self-Crystallizing Behavior Trees

Reference implementation and experimental data for:

> **Self-Crystallizing Behavior Trees: Amortizing Foundation-Model Reasoning into Deterministic Control for Autonomous Drones**
> Behnam Moradi, 2026. [paper/scbt_paper_final.pdf](paper/scbt_paper_final.pdf) ([arXiv link TBD])

## What this is

Classical robot control (Finite State Machines, Behavior Trees) is deterministic
and inspectable but fixed at design time. Foundation models can propose new
control logic at runtime, but at the cost of latency, token cost, and
non-determinism that are hard to accept in a safety-critical control loop.

**SCBT treats the LLM as a compiler, not a runtime.** When a node of a
vehicle's Behavior Tree fails:

1. An LLM (via a schema-constrained [Google ADK](https://github.com/google/adk-python)
   pipeline) proposes a minimal, mission-agnostic subtree **patch**.
2. **CMA-ES** tunes the patch's declared numeric parameters against simulated
   replays.
3. A **three-stage admission pipeline** (static check against a grammar and
   capability manifest, invariant-monitored replays, and an empirical success
   margin over the tree as it currently stands) gates entry into a
   fleet-shared, context-indexed **patch library**.
4. A **contextual bandit** (linear Thompson sampling) learns whether to reuse
   a stored patch or pay to call the LLM again.
5. Once the LLM call rate has decayed, the library **crystallizes** into an
   ordinary deterministic Behavior Tree that no human wrote, with zero
   learning components running at deployment.

Implemented and evaluated on [Microsoft Project AirSim](https://github.com/iamaisim/ProjectAirSim)
(headless runtime, Fast Physics, Simple Flight controller), with Gemini 3.1
Pro as the proposer. See the paper for the full method, a 3-seed learning
phase, a 6-baseline ablation matrix, and an exact offline determinism test.

## Repository layout

```
src/scbt/
  sim/        Project AirSim session, scene/robot config, lockstep clock, runtime launcher
  bt/         py_trees leaf nodes, XML dialect + registry, base trees, hand-coded expert tree (B0)
  observer/   failure interception (MissionCritical), tick-trace recording, failure events
  inject/     failure-injection curriculum (F1 wind, F7 battery, ...) and held-out variants
  patch/      patch grammar, static checker, hot-patch application to a live tree
  manager/    embedding, fleet library, Thompson-sampling bandit, ADK proposer, Patch Manager
  admit/      CMA-ES tuning + replay-based admission (stages 2-3), parallel replay farm
  crystallize.py          compile an admitted library into a frozen T*_m per class/mission
  bench/      episode runner, learning-phase loop, frozen-phase evaluator, baseline
              presets (B0-B6), benchmark matrix driver, results/figure report
tests/        offline unit tests (no simulator required)
data/matrix/  the exact run artifacts (libraries, learning logs, frozen-phase results,
              crystallized trees, report.txt, amortization figure) behind Tables I-II
              and Fig. 3 of the paper
paper/        paper source, compiled PDF, and the wind-calibration note (Sec. VI-A)
```

## Setup

Requires a **Project AirSim** checkout (the simulator itself is not vendored
here) with the headless runtime built:

```bash
git clone https://github.com/iamaisim/ProjectAirSim
cd ProjectAirSim && ./build.sh simlibs_release   # builds samples/projectairsim_runtime
```

Then, with this repo checked out **next to** `ProjectAirSim/` (or with
`$PROJECTAIRSIM_ROOT` pointed at your checkout):

```bash
pixi install
pixi run test          # offline unit tests, no simulator needed
```

`scbt` locates the Project AirSim checkout, in order: `$PROJECTAIRSIM_ROOT`,
the nearest ancestor directory containing `client/python`, or a sibling
directory named `ProjectAirSim`. The headless runtime binary is found the
same way, or via `$SCBT_RUNTIME_BIN` directly.

Without pixi: `pip install -e .[all]` plus
`pip install -e /path/to/ProjectAirSim/client/python/projectairsim`.

The live LLM proposer needs `GOOGLE_API_KEY` (Gemini) or another
[LiteLLM](https://github.com/BerriAI/litellm)-supported model string; without
a key, `--proposer mock` exercises the full pipeline with a fixed
hand-written proposer, and the sentence embedder falls back to a
deterministic hashing scheme if `sentence-transformers` is not installed.

## Quickstart

```bash
# One episode, nominal mission
pixi run episode -- --mission M1

# Inject a wind gust; no recovery vs. a hand-written hot-patch
pixi run episode -- --mission M1 --inject F1 --trigger 12 --seed 1 --policy none
pixi run episode -- --mission M1 --inject F1 --trigger 12 --seed 1 --policy handwritten --print-tree

# Learning phase against a live LLM (needs GOOGLE_API_KEY), with admission + tuning
pixi run learn -- --episodes 20 --proposer gemini-3.1-pro-preview --inject F1 F7 \
    --library runs/lib.json --out runs/learn.json --admit --K 3 --tune-budget 4 \
    --farm-size 4 --instance 5 --farm-instance 6

# Crystallize the library and evaluate the frozen tree, no LLM/bandit/optimizer running
pixi run crystallize -- --library runs/lib.json --out-dir runs/frozen
pixi run frozen -- --library runs/lib.json --episodes 20 --ndet 10 --out runs/frozen.json

# Full baseline matrix (B0-B6) and the paper's tables/figure
pixi run matrix -- --out-dir runs/matrix --methods scbt B1 B2 B3 B4 B5 B6 --seeds 0 1 2
pixi run report -- --dir runs/matrix
```

`--instance N` selects simulator ports `8989+2N` / `8990+2N`, so several
runtime instances (live episode, replay farm) run concurrently. Lockstep
mode (default) steps the simulator exactly one tick per BT tick at 10x real
time and is what makes episodes reproducible; `--realtime` runs the
simulator in real time instead.

## Reproducing the paper's numbers

`data/matrix/` is the unmodified output of the run behind Tables I-II and
Fig. 3 (library and learning-phase logs per method/seed, frozen-phase
results on curriculum and held-out failures, crystallized trees, and
`report.txt`). Regenerate the tables and figure from it directly:

```bash
pixi run report -- --dir data/matrix
```

To reproduce from scratch, see `pixi run matrix -- --help`; a full 3-seed
SCBT run plus the 5 single-seed ablations takes on the order of several
hours, dominated by LLM latency and CMA-ES replay evaluations.

## Known limitations (stated in the paper)

Two failure types evaluated end-to-end (wind gust, battery); single vehicle
per episode (the fleet, inter-vehicle separation, and cross-class patch
transfer are implemented in `scbt.sim`/`scbt.patch` but not exercised in the
reported experiments); ablations run on a single seed; headless simulator
host, so failure context is derived from telemetry rather than vision; a
known stage-2 admission limitation (it rejects invariant violations caused
by the injected failure itself, not only by the candidate patch) is
identified and left unfixed so the reported results match the method as
specified. See the paper's Discussion for the full list.

## Citation

```bibtex
@misc{moradi2026scbt,
  title  = {Self-Crystallizing Behavior Trees: Amortizing Foundation-Model
            Reasoning into Deterministic Control for Autonomous Drones},
  author = {Moradi, Behnam},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

Code: MIT (see `LICENSE`). The paper text and figures are not covered by
the code license; see `paper/`.

This project depends on, but does not redistribute, Microsoft/IAMAI's
[Project AirSim](https://github.com/iamaisim/ProjectAirSim) (MIT) and uses
[py_trees](https://github.com/splintered-reality/py_trees) (BSD-3-Clause),
[Google ADK](https://github.com/google/adk-python) (Apache-2.0),
[CMA-ES](https://github.com/CMA-ES/pycma) (BSD-3-Clause), and
[sentence-transformers](https://github.com/UKPLab/sentence-transformers)
(Apache-2.0). See `NOTICE.md`.
