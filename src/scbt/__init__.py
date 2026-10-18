"""Self-Crystallizing Behavior Trees (SCBT) on Project AirSim.

Layout
------
scbt.sim       launch/connect to projectairsim-runtime, scene config, vehicle wrapper
scbt.bt        py_trees leaves and base trees (System 1, deterministic)
scbt.observer  tick-trace logger, failure event extraction, determinism hash
scbt.inject    failure curriculum (F1..F12 headless subset)
scbt.patch     patch grammar, static checker, apply-to-live-tree
scbt.bench     episode runner and metrics
"""

__version__ = "0.1.0"
