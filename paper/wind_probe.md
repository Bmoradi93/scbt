# Wind response of the Fast Physics quadrotor (headless runtime, Simple Flight)

Probe: hover at 8 m, apply uniform wind for 8 s, measure drift.

| wind (m/s) | drift after 8 s | note |
|---|---|---|
| 10 | 0.18 m | fully rejected |
| 20 | 0.78 m | fully rejected |
| 30 | 46.9 m | controller saturated; speed 7.6 m/s downwind |
| 25 (during MoveToPosition, 40 m leg) | 226 m from target after 19 s; command returned False | |

External force (`set_external_force`): 5 N -> 18 m / 6 s, 10 N -> 49 m / 6 s.

Conclusion: F1 uses wind in [22, 28] m/s for 15-30 s. The drone is displaced
tens of metres; whether the leg still completes within its timeout depends
on gust duration and timing, which is what the admission replays measure.
