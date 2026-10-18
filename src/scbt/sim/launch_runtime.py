"""Locate and launch ``projectairsim-runtime`` (headless host).

Usage (inside the pixi env):
    python -m scbt.sim.launch_runtime [--instance N]   # ports 8989+2N / 8990+2N

The runtime links against ``libonnxruntime.so.1.12.1`` but the build tree only
ships ``libonnxruntime.so``; we create a versioned symlink in a scratch dir
and prepend it to ``LD_LIBRARY_PATH`` so the binary runs unmodified.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

from .config import _find_project_airsim_root

REPO_ROOT = _find_project_airsim_root()
BASE_TOPICS_PORT = 8989
BASE_SERVICES_PORT = 8990


def instance_ports(instance: int) -> tuple[int, int]:
    return BASE_TOPICS_PORT + 2 * instance, BASE_SERVICES_PORT + 2 * instance


def find_runtime_binary() -> str:
    env = os.environ.get("SCBT_RUNTIME_BIN")
    if env and os.path.isfile(env):
        return env
    pattern = os.path.join(
        REPO_ROOT, "build", "linux64", "*", "*", "samples", "projectairsim_runtime", "projectairsim-runtime"
    )
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"projectairsim-runtime not found under '{REPO_ROOT}'. Build it with "
            "`./build.sh simlibs_release` inside your Project AirSim checkout, set "
            "$PROJECTAIRSIM_ROOT to that checkout, or set $SCBT_RUNTIME_BIN to the "
            "binary path directly. See README.md 'Setup'."
        )
    return candidates[0]


def _runtime_env(bin_path: str) -> dict:
    bin_dir = os.path.dirname(bin_path)
    shim = os.path.join(tempfile.gettempdir(), "scbt_rtlib")
    os.makedirs(shim, exist_ok=True)
    src = os.path.join(bin_dir, "libonnxruntime.so")
    dst = os.path.join(shim, "libonnxruntime.so.1.12.1")
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in [shim, bin_dir, env.get("LD_LIBRARY_PATH", "")] if p
    )
    return env


def launch(instance: int = 0, extra_args: Optional[List[str]] = None, quiet: bool = True) -> subprocess.Popen:
    bin_path = find_runtime_binary()
    topics, services = instance_ports(instance)
    cmd = [bin_path, "--topics-port", str(topics), "--services-port", str(services)]
    cmd += extra_args or []
    out = subprocess.DEVNULL if quiet else None
    return subprocess.Popen(cmd, env=_runtime_env(bin_path), stdout=out, stderr=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, default=0)
    args = ap.parse_args(argv)
    proc = launch(args.instance, quiet=False)
    t, s = instance_ports(args.instance)
    print(f"[scbt] runtime pid={proc.pid} topics={t} services={s}", file=sys.stderr)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
