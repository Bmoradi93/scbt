"""Failure-injection curriculum (paper Sec. V-C), headless subset.

Each failure is an ``Injection`` with ``arm(session, vehicle)`` called once
at a trigger condition and ``tick(session, vehicle)`` called every tick (for
time-varying faults). ``params`` holds the sampled parameter values so the
same failure can be replayed; ``sample(rng)`` draws them from the catalogue
ranges used both for injection and for admission randomization.

Headless-capable: F1 wind, F2 GPS, F5 pad shifted, F6 no-fly zone (geofence
shrink), F7 battery, F8 link, F12 separation conflict. F3/F4/F9/F10/F11 need
the Unreal host and are listed as ``requires_unreal=True`` for completeness.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..sim.session import SimSession, Vehicle


@dataclass
class Injection:
    code: str
    name: str
    params: Dict[str, Any]
    trigger_t: float                      # sim seconds after mission start
    requires_unreal: bool = False
    armed: bool = False
    _arm: Optional[Callable[[SimSession, Vehicle, Dict[str, Any]], None]] = None
    _tick: Optional[Callable[[SimSession, Vehicle, Dict[str, Any], float], None]] = None
    _clear: Optional[Callable[[SimSession, Vehicle, Dict[str, Any]], None]] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def maybe_fire(self, session: SimSession, vehicle: Vehicle, t_since_start: float) -> bool:
        if not self.armed and t_since_start >= self.trigger_t:
            self.armed = True
            if self._arm:
                self._arm(session, vehicle, self.params)
            return True
        if self.armed and self._tick:
            self._tick(session, vehicle, self.params, t_since_start - self.trigger_t)
        return False

    def clear(self, session: SimSession, vehicle: Vehicle) -> None:
        if self.armed and self._clear:
            self._clear(session, vehicle, self.params)
        self.armed = False


# ---- F1: sustained wind gust ------------------------------------------------

def _f1_sample(rng: random.Random) -> Dict[str, Any]:
    # Measured on the Fast Physics quadrotor (scbt/docs/wind_probe.md):
    # <=20 m/s is fully rejected by Simple Flight (<1 m drift), >=25 m/s
    # saturates the controller (tens of metres of drift in seconds). The
    # band below is therefore "beyond tracking capability" by construction.
    speed = rng.uniform(22.0, 28.0)
    heading = rng.uniform(0, 2 * math.pi)
    return {"speed": speed, "heading": heading, "duration_s": rng.uniform(15.0, 30.0)}


def _f1_arm(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    s.set_wind(p["speed"] * math.cos(p["heading"]), p["speed"] * math.sin(p["heading"]), 0.0)


def _f1_tick(s: SimSession, v: Vehicle, p: Dict[str, Any], dt: float) -> None:
    if dt >= p["duration_s"] and s.wind != (0.0, 0.0, 0.0):
        s.set_wind(0.0, 0.0, 0.0)


def _f1_clear(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    s.set_wind(0.0, 0.0, 0.0)


# ---- F2: GPS degradation then loss ----------------------------------------

def _f2_sample(rng: random.Random) -> Dict[str, Any]:
    return {"noise_m": rng.uniform(4.0, 10.0), "loss_after_s": rng.uniform(5.0, 15.0), "duration_s": rng.uniform(20.0, 40.0)}


def _f2_arm(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    v.gps_noise_m = p["noise_m"]


def _f2_tick(s: SimSession, v: Vehicle, p: Dict[str, Any], dt: float) -> None:
    if dt >= p["loss_after_s"]:
        v.gps_ok = False
    if dt >= p["duration_s"]:
        v.gps_ok = True
        v.gps_noise_m = 0.0


def _f2_clear(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    v.gps_ok = True
    v.gps_noise_m = 0.0


# ---- F5: landing pad / home shifted --------------------------------------

def _f5_sample(rng: random.Random) -> Dict[str, Any]:
    return {"shift_m": rng.uniform(8.0, 20.0), "heading": rng.uniform(0, 2 * math.pi)}


def _f5_arm(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    n, e, d = v.home_ned
    p["_old_home"] = v.home_ned
    v.home_ned = (n + p["shift_m"] * math.cos(p["heading"]), e + p["shift_m"] * math.sin(p["heading"]), d)


def _f5_clear(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    if "_old_home" in p:
        v.home_ned = p["_old_home"]


# ---- F6: no-fly zone announced (geofence shrinks across the route) -------

def _f6_sample(rng: random.Random) -> Dict[str, Any]:
    return {"n_max": rng.uniform(15.0, 30.0)}


def _f6_arm(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    p["_old"] = dict(s.geofence)
    s.geofence["n_max"] = p["n_max"]


def _f6_clear(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    if "_old" in p:
        s.geofence.update(p["_old"])


# ---- F7: battery capacity reduced ----------------------------------------

def _f7_sample(rng: random.Random) -> Dict[str, Any]:
    # Below the 25% reserve guard, but with enough charge to finish the
    # required leg in a degraded mode (see docs/wind_probe.md for energy
    # calibration). The admission invariant requires >= 10% at landing.
    return {"set_pct": rng.uniform(17.0, 24.0), "drain_mult": rng.uniform(1.3, 2.2)}


def _f7_arm(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    v.drone.set_battery_remaining(p["set_pct"])
    p["_old_drain"] = float(v.spec.battery_drain_rate)
    v.drone.set_battery_drain_rate(p["_old_drain"] * p["drain_mult"])


def _f7_clear(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    if "_old_drain" in p:
        v.drone.set_battery_drain_rate(p["_old_drain"])


# ---- F8: link dropout ----------------------------------------------------

def _f8_sample(rng: random.Random) -> Dict[str, Any]:
    return {"quality": rng.uniform(0.0, 0.2), "duration_s": rng.uniform(10.0, 30.0)}


def _f8_arm(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    v.link_quality = p["quality"]


def _f8_tick(s: SimSession, v: Vehicle, p: Dict[str, Any], dt: float) -> None:
    if dt >= p["duration_s"]:
        v.link_quality = 1.0


def _f8_clear(s: SimSession, v: Vehicle, p: Dict[str, Any]) -> None:
    v.link_quality = 1.0


# ---- F12: separation conflict (second vehicle crosses the route) ---------
# Implemented at the episode level (needs a second vehicle's tree); the
# injection only records the parameter. See scbt.bench.episode.

def _f12_sample(rng: random.Random) -> Dict[str, Any]:
    return {"cross_at_frac": rng.uniform(0.3, 0.7)}


# Catalogue ranges: used to clip event-neighbourhood perturbations so a
# replay stays inside the failure's defining band (e.g. F7 below the guard).
RANGES: Dict[str, Dict[str, tuple]] = {
    "F1": {"speed": (22.0, 28.0), "duration_s": (15.0, 30.0)},
    "F2": {"noise_m": (4.0, 10.0), "loss_after_s": (5.0, 15.0), "duration_s": (20.0, 40.0)},
    "F5": {"shift_m": (8.0, 20.0)},
    "F6": {"n_max": (15.0, 30.0)},
    "F7": {"set_pct": (17.0, 24.0), "drain_mult": (1.3, 2.2)},
    "F8": {"quality": (0.0, 0.2), "duration_s": (10.0, 30.0)},
}

# Held-out variants: disjoint sub-ranges, still inside the failing regime.
RANGES_HELDOUT: Dict[str, Dict[str, tuple]] = {
    "F1": {"speed": (28.0, 30.0), "duration_s": (30.0, 36.0)},
    "F2": {"noise_m": (10.0, 14.0), "loss_after_s": (3.0, 5.0), "duration_s": (40.0, 50.0)},
    "F5": {"shift_m": (20.0, 26.0)},
    "F6": {"n_max": (10.0, 15.0)},
    "F7": {"set_pct": (15.0, 17.0), "drain_mult": (1.2, 1.3)},
    "F8": {"quality": (0.2, 0.3), "duration_s": (30.0, 40.0)},
}


def sample_heldout(code: str, rng: random.Random) -> Dict[str, Any]:
    """Sample from the held-out sub-ranges (other params from the catalogue)."""
    p = FAILURES[code]["sample"](rng)
    for k, (lo, hi) in RANGES_HELDOUT.get(code, {}).items():
        p[k] = rng.uniform(lo, hi)
    return p


FAILURES: Dict[str, Dict[str, Any]] = {
    "F1": dict(name="wind_gust", sample=_f1_sample, arm=_f1_arm, tick=_f1_tick, clear=_f1_clear),
    "F2": dict(name="gps_degradation", sample=_f2_sample, arm=_f2_arm, tick=_f2_tick, clear=_f2_clear),
    "F3": dict(name="corridor_obstacle", sample=lambda r: {}, requires_unreal=True),
    "F4": dict(name="pad_occupied", sample=lambda r: {}, requires_unreal=True),
    "F5": dict(name="pad_shifted", sample=_f5_sample, arm=_f5_arm, clear=_f5_clear),
    "F6": dict(name="no_fly_zone", sample=_f6_sample, arm=_f6_arm, clear=_f6_clear),
    "F7": dict(name="battery_reduced", sample=_f7_sample, arm=_f7_arm, clear=_f7_clear),
    "F8": dict(name="link_dropout", sample=_f8_sample, arm=_f8_arm, tick=_f8_tick, clear=_f8_clear),
    "F9": dict(name="camera_dropout", sample=lambda r: {}, requires_unreal=True),
    "F10": dict(name="target_missing", sample=lambda r: {}, requires_unreal=True),
    "F11": dict(name="target_occluded", sample=lambda r: {}, requires_unreal=True),
    "F12": dict(name="separation_conflict", sample=_f12_sample),
}


def make_injection(code: str, rng: random.Random, trigger_t: Optional[float] = None,
                   params: Optional[Dict[str, Any]] = None, heldout: bool = False) -> Injection:
    spec = FAILURES[code]
    if params is not None:
        p = dict(params)
    elif heldout:
        p = sample_heldout(code, rng)
    else:
        p = spec["sample"](rng)
    return Injection(
        code=code,
        name=spec["name"],
        params=p,
        trigger_t=trigger_t if trigger_t is not None else rng.uniform(8.0, 20.0),
        requires_unreal=spec.get("requires_unreal", False),
        _arm=spec.get("arm"),
        _tick=spec.get("tick"),
        _clear=spec.get("clear"),
    )
