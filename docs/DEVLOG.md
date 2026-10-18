# Development log

Implementation notes and decisions made while building SCBT, kept for
anyone extending this code. Not required reading to use the repository;
see the top-level `README.md` for that.

## Verified behaviour (incremental milestones)

* Nominal M1 (takeoff, 40 m out, back, land): mission SUCCESS, ~76 s sim.
* F1 wind gust (22-28 m/s) during leg 1, `--policy none`: `wp1` fails,
  failure event raised with `v_f = wp1 (GoToNED)`, give-up -> abort branch.
* Same with `--policy handwritten`: patch `WrapFallback(wp1, ...)` applied
  mid-flight without restarting; recovery waits out the gust; mission SUCCESS.
* Determinism: identical seed -> identical trace hash and tick count on
  repeated runs (with and without patch).
* F7 battery: `battery_guard (IsBatteryOk)` is the culprit; feedback carries
  the values.

See `paper/wind_probe.md` for the measured wind response that fixes the F1 range.

## Learning phase, first runs

Mock-proposer run (12 episodes, seed 0): 11/12 missions successful, 17
failure events, 7 LLM calls, 10 library hits; cumulative LLM calls per
episode `[1,1,3,4,5,5,6,6,6,7,7,7]` (zero calls in 6 of the last 8
episodes). The battery patch learned on M2 transferred to M1 because
patches are mission-agnostic (`{@port}` anchor references, `GoToRequiredGoal`).

First Gemini 3.1 Pro result (M2, F7): proposed degraded-mode patch; CMA-ES
theta = (reserve 13.9%, v 3.9 m/s); replays 3/3 vs 0/3 unpatched; landing
reserve 12.5%; admitted; live mission recovered (value 0.5). The same patch
with theta = (13.2, 3.6) landed at 9.8% and was correctly rejected at
stage 2 before the reserve invariant was added to the CMA-ES objective
(Eq. 2 in the paper).

With static-only admission, Gemini's first F7 patch was "just
ReturnToLaunch" (safe, abandons the required goal, 0/6 successes); the
bandit then correctly stopped paying for the LLM, which had produced the
same patch twice, and the system converged on a bad patch. This is the
paper's argument for empirical admission, observed directly (Sec. VII-D,
item 1).

## Non-regression admission

Crystallizing an early Gemini library produced a tree whose only patch was
a `ReplaceSubtree` that deleted the mission's optional waypoints and always
flew "required goal, return, land" -- it had passed admission because
success only checks required goals, and the bandit loved it (7/7 at the
time). Admission now also replays the patched tree with *no* failure
(one replay suffices: lockstep is deterministic) and rejects any drop in
nominal success or mission value relative to the unpatched current tree
(Sec. VII-D, item 3).

## Determinism: investigation after the first benchmark matrix

The first full matrix run reported "2 of 5 distinct traces" for several
frozen cells. Diagnosis, in order:

1. **Command dispatch race (fixed).** A command issued in tick *t* was sent
   to the server only when the asyncio loop next yielded, after the step;
   so it reached the server before step *t+1* or *t+2* depending on wall
   time. `sleep_tick` now drains pending sends and issues a synchronous
   barrier request before stepping.
2. **Completion-reply race (fixed).** Action leaves observed completion via
   the server's reply, which arrives at a wall-clock-dependent moment. All
   action leaves now decide completion from the per-tick telemetry
   snapshot (`complete_by_telemetry`); the reply is used for errors only.
3. **The simulator itself is not bit-reproducible (not fixable
   client-side).** With *no BT in the loop* (takeoff, one
   `move_to_position`, wind applied at a fixed step), five identical
   lockstep runs gave three distinct trajectories differing by ~1e-4 m: the
   server's command-execution thread is wall-clock scheduled. This
   occasionally flips a threshold decision (arrival, geofence) by one tick
   in a >2000-tick episode.

After (1) and (2): the BT is a pure function of telemetry. Re-ticking the
frozen tree offline against a recorded telemetry log reproduces the online
trace hash exactly, and gives one distinct hash over repeated re-ticks; see
`scbt/bench/replay_det.py` and the "offline" columns of Table II / the
`--ndet` output of `scbt.bench.frozen`.

## Other design decisions and bugs found by running

* **Success** = all required mission goals reached *and* the vehicle
  landed (outcome `mission` or `abort`). `mission_value` weights required
  goals 1 and optional goals 0.5. An abort after the required goals are
  met is not a failure; an abort before them is.
* **F7** = the battery reserve guard (25%) trips before the required goal;
  a useful recovery completes the required leg under a lower reserve and
  skips optional legs (`GoToRequiredGoal`). Admission enforces >= 10%
  battery at landing (stage 2).
* **Patches must be mission-agnostic**: literal coordinates in a patch
  subtree are a static-check smell; use `{@north}` etc. (references to the
  anchor leaf's own ports) or the `GoToRequiredGoal` leaf instead. Learned
  the hard way: an M2 patch with literal lane coordinates was reused on M1
  and flew to the wrong place.
* **Patch budget** per episode (default 3) and per failed node (default 2);
  already-applied patches are excluded from bandit fallback selection.
  Without these limits the manager re-applied the same patch on the same
  anchor repeatedly until the tree exceeded the interpreter's recursion
  limit.
* Energy metric integrates per-tick battery decrements rather than
  start-minus-end, so it is robust to the in-episode battery resets used by
  the F7 injection.
* `_replace` on a decorator parent (e.g. `MissionCritical`) must clear the
  old child's parent link; a patch anchored directly on `mission_seq`
  otherwise crashes the episode on apply. A failed apply now rejects the
  patch instead of aborting the episode.
* Blackboard namespaces are `/sim<instance>/<vehicle>`: the replay farm
  runs vehicles with the same names on other runtime instances; without the
  instance prefix the live episode's tree ended up commanding the farm's
  drone.
* Google ADK treats `{...}` in agent *instructions* as template variables;
  the patch grammar (which uses curly braces for tunable placeholders) is
  therefore passed in the user message, not the system instruction.
* LLM default is `gemini-3.1-pro-preview` (the newest explicitly-versioned
  Pro model at the time of writing); `gemini-pro-latest` is a moving alias
  and was avoided for reproducibility.

## Known issues not addressed in the reported experiments

* The headless Project AirSim runtime has no robot-to-robot mesh collision;
  inter-vehicle separation (failure F12) is monitored client-side
  (`SimSession.separation_violations`) but not exercised in a multi-vehicle
  episode in the reported results.
* "Wind telemetry" used for context embedding is the ground-truth wind
  vector from the simulator session, not an onboard estimate; a wind
  estimator leaf could replace it without touching the trees.
* Stage 2 of admission (Sec. IV-E) rejects a candidate patch if *any*
  replay violates an invariant, without reference to what the unpatched
  tree does under the same perturbation. Under wind failures, the gust
  itself can carry the vehicle across the geofence regardless of the patch;
  the correct rule is relative (reject only violations the patch
  introduces beyond the baseline, as stage 3 already does for success
  rate). See the paper's Discussion (Sec. VIII) for the consequence this
  had on the benchmark matrix (the B5 ablation).
