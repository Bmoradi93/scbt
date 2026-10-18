# Third-party notices

This repository's own code (`src/scbt/`, `tests/`) is licensed under the MIT
License (see `LICENSE`). It depends on, but does not vendor or redistribute,
the following third-party projects:

| Project | License | Role |
|---|---|---|
| [Project AirSim](https://github.com/iamaisim/ProjectAirSim) (Microsoft / IAMAI Consulting Corp.) | MIT | Simulator and Python client; **not included**, must be installed separately (see README "Setup") |
| [py_trees](https://github.com/splintered-reality/py_trees) | BSD-3-Clause | Behavior Tree execution engine |
| [pycma](https://github.com/CMA-ES/pycma) | BSD-3-Clause | CMA-ES optimizer used for patch parameter tuning |
| [Google ADK (Agent Development Kit) for Python](https://github.com/google/adk-python) | Apache-2.0 | Schema-constrained LLM agent pipeline |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Apache-2.0 | `all-MiniLM-L6-v2` sentence embedding for failure-context retrieval |
| [commentjson](https://github.com/vaidik/commentjson) | MIT | JSONC parsing for Project AirSim scene/robot configs |

No source code from these projects is copied into this repository; they are
runtime dependencies installed via `pixi`/`pip`.
