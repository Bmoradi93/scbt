"""py_trees leaves over ``scbt.sim.session.Vehicle``.

Conventions (mirror BehaviorTree.CPP so the paper's formulation applies):
* Every leaf has typed *ports* declared in ``PORTS`` as
  ``{name: (type, default, (min, max) | None)}``. Numeric ports with bounds
  are *tunable* and may be optimized by CMA-ES.
* Leaves read telemetry from ``vehicle.telemetry`` only (refreshed once per
  tick by the session), so the tick trace is deterministic given telemetry.
* Action leaves start an ``asyncio.Task`` on first tick and return RUNNING
  until it completes; a server-side error or a leaf-level timeout returns
  FAILURE. ``terminate()`` cancels the in-flight command on preemption.

A leaf's ``vehicle`` is resolved from the blackboard key ``vehicle`` at setup
time so that trees can be built from XML without a vehicle object in hand.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
from typing import Any, Dict, Optional, Tuple

import py_trees
from py_trees.common import Status

PortSpec = Dict[str, Tuple[type, Any, Optional[Tuple[float, float]]]]

# Blackboard namespace for the tree being built (one per vehicle so several
# vehicles can run in one process). Set by the runner around from_xml().
NAMESPACE: contextvars.ContextVar[str] = contextvars.ContextVar("scbt_bb_namespace", default="/default")


class VehicleLeaf(py_trees.behaviour.Behaviour):
    """Base class: binds to the vehicle on the shared blackboard."""

    PORTS: PortSpec = {}

    def __init__(self, name: str, **ports: Any):
        super().__init__(name=name)
        self.ports: Dict[str, Any] = {}
        for pname, (ptype, default, _bounds) in self.PORTS.items():
            raw = ports.pop(pname, default)
            self.ports[pname] = ptype(raw) if raw is not None else None
        if ports:
            raise TypeError(f"{type(self).__name__}: unknown ports {sorted(ports)}")
        self.bb = self.attach_blackboard_client(name=name, namespace=NAMESPACE.get())
        self.bb.register_key("vehicle", access=py_trees.common.Access.READ)
        self.bb.register_key("session", access=py_trees.common.Access.READ)
        self.bb.register_key("speed", access=py_trees.common.Access.WRITE)
        self.bb.register_key("goals_done", access=py_trees.common.Access.WRITE)

    @property
    def vehicle(self):
        return self.bb.vehicle

    @property
    def session(self):
        return self.bb.session

    @property
    def tel(self) -> Dict[str, Any]:
        return self.vehicle.telemetry

    # Default cruise speed is a blackboard value so SetSpeed can change it.
    @property
    def cruise_speed(self) -> float:
        try:
            return float(self.bb.speed)
        except KeyError:
            return 3.0

    # --- serialization for patches / crystallization ---
    def port_values(self) -> Dict[str, Any]:
        return dict(self.ports)


class AsyncActionLeaf(VehicleLeaf):
    """Runs one simulator command as an asyncio task."""

    PORTS: PortSpec = {"timeout_s": (float, 30.0, (5.0, 120.0))}

    def __init__(self, name: str, **ports: Any):
        super().__init__(name, **ports)
        self._task: Optional[asyncio.Task] = None
        self._starter: Optional[asyncio.Task] = None
        self._t0: float = 0.0

    async def start_command(self) -> asyncio.Task:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_done(self, result: Any) -> Status:
        """Map the command result to a status. Override for extra checks."""
        return Status.SUCCESS if result is not False else Status.FAILURE

    def complete_by_telemetry(self) -> bool:
        """Completion decided from the per-tick telemetry snapshot, so the
        tick trace is a function of simulator state and not of when the
        server's completion reply happens to arrive over the socket (which
        was the only source of trace non-determinism we found). The reply
        is still used for errors and as a fallback."""
        return False

    def initialise(self) -> None:
        self._t0 = self.tel.get("t", 0.0)
        self._task = None
        # The command itself is created by an awaitable (request_async); we
        # wrap it so the tick stays synchronous.
        self._starter = asyncio.get_running_loop().create_task(self._start())
        self.session.pending_starts.append(self._starter)

    async def _start(self) -> None:
        self._task = await self.start_command()

    def update(self) -> Status:
        if self._starter is not None and self._starter.done():
            exc = self._starter.exception()
            if exc is not None:
                self.feedback_message = f"start failed: {exc}"
                return Status.FAILURE
        if self._task is not None and self.complete_by_telemetry():
            return self.on_done(None)
        if self._task is None:
            return Status.RUNNING
        if self._task.done():
            if self._task.cancelled():
                return Status.FAILURE
            exc = self._task.exception()
            if exc is not None:
                self.feedback_message = f"command error: {exc}"
                return Status.FAILURE
            # A reply (success, or False = server-side timeout) that telemetry
            # does not confirm: keep RUNNING until the leaf's own sim-time
            # timeout, so the trace is independent of reply timing. The
            # server timeout equals the leaf timeout, so this costs at most a
            # tick. Exceptions and cancellation remain immediate failures.
        if self.tel.get("t", 0.0) - self._t0 > self.ports["timeout_s"]:
            self.feedback_message = "leaf timeout"
            return self.on_done(None) if self._task.done() else Status.FAILURE
        return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self.vehicle.cancel()
        self._task = None
        self._starter = None


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

class Takeoff(AsyncActionLeaf):
    """Takeoff to the controller's default altitude, then verify airborne."""

    PORTS: PortSpec = {
        "timeout_s": (float, 20.0, (5.0, 60.0)),
        "min_alt_m": (float, 1.5, (0.5, 5.0)),
    }

    def initialise(self) -> None:
        self._already_up = self.tel.get("alt", 0.0) >= self.ports["min_alt_m"] and not self.tel.get("landed", True)
        if self._already_up:
            self._task = None
            self._starter = None
            return
        super().initialise()

    async def start_command(self) -> asyncio.Task:
        return await self.vehicle.takeoff(timeout_sec=self.ports["timeout_s"])

    def complete_by_telemetry(self) -> bool:
        return (self.tel.get("alt", 0.0) >= self.ports["min_alt_m"] and not self.tel.get("landed", True)
                and abs(self.tel.get("vel", (0, 0, 0))[2]) < 0.2 and self.tel.get("t", 0.0) - self._t0 > 1.0)

    def update(self) -> Status:
        if getattr(self, "_already_up", False):
            return Status.SUCCESS  # idempotent: mission restarts after a patch
        return super().update()

    def on_done(self, result: Any) -> Status:
        ok = self.tel.get("alt", 0.0) >= self.ports["min_alt_m"]
        if not ok:
            self.feedback_message = f"alt {self.tel.get('alt', 0.0):.2f} < min_alt"
        return Status.SUCCESS if ok else Status.FAILURE


class Land(AsyncActionLeaf):
    PORTS: PortSpec = {"timeout_s": (float, 60.0, (10.0, 120.0))}

    async def start_command(self) -> asyncio.Task:
        return await self.vehicle.land(timeout_sec=self.ports["timeout_s"])

    def complete_by_telemetry(self) -> bool:
        return bool(self.tel.get("landed", False)) or (self.tel.get("alt", 1.0) < 0.3 and self.tel.get("speed", 1.0) < 0.2)

    def on_done(self, result: Any) -> Status:
        return Status.SUCCESS if self.tel.get("landed", False) or self.tel.get("alt", 1.0) < 0.3 else Status.FAILURE


class ReturnToLaunch(AsyncActionLeaf):
    PORTS: PortSpec = {
        "timeout_s": (float, 120.0, (20.0, 300.0)),
        "velocity": (float, 3.0, (0.5, 8.0)),
        "arrive_tol_m": (float, 2.0, (0.5, 5.0)),
    }

    async def start_command(self) -> asyncio.Task:
        return await self.vehicle.go_home(velocity=self.ports["velocity"], timeout_sec=self.ports["timeout_s"])

    def _home_dist(self) -> float:
        hn, he, _ = self.vehicle.home_ned
        pn, pe, _ = self.vehicle.ned
        return math.hypot(pn - hn, pe - he)

    def complete_by_telemetry(self) -> bool:
        return self._home_dist() <= self.ports["arrive_tol_m"] and self.tel.get("speed", 1.0) < 0.3

    def on_done(self, result: Any) -> Status:
        return Status.SUCCESS if self._home_dist() <= self.ports["arrive_tol_m"] else Status.FAILURE


class GoToNED(AsyncActionLeaf):
    """Fly to an absolute NED point; SUCCESS only if within ``arrive_tol_m``.

    This is the leaf that fails under F1 (wind) when the controller cannot
    hold the track within the timeout, and under F7 when the battery guard
    preempts it.
    """

    PORTS: PortSpec = {
        "north": (float, 0.0, None),
        "east": (float, 0.0, None),
        "down": (float, -5.0, None),
        "velocity": (float, 0.0, (0.5, 8.0)),      # 0 -> use blackboard cruise speed
        "timeout_s": (float, 40.0, (5.0, 120.0)),
        "arrive_tol_m": (float, 1.5, (0.5, 5.0)),
        "goal_id": (str, "", None),                 # non-empty: reaching it completes a mission goal
    }

    async def start_command(self) -> asyncio.Task:
        v = self.ports["velocity"] or self.cruise_speed
        return await self.vehicle.move_to_ned(
            self.ports["north"], self.ports["east"], self.ports["down"], v, self.ports["timeout_s"]
        )

    def complete_by_telemetry(self) -> bool:
        d = self.vehicle.dist_to(self.ports["north"], self.ports["east"], self.ports["down"])
        return d <= self.ports["arrive_tol_m"] and self.tel.get("speed", 1.0) < 0.3

    def on_done(self, result: Any) -> Status:
        d = self.vehicle.dist_to(self.ports["north"], self.ports["east"], self.ports["down"])
        if d > self.ports["arrive_tol_m"]:
            self.feedback_message = f"arrived {d:.2f} m from target"
            return Status.FAILURE
        if self.ports["goal_id"]:
            done = set(self.bb.goals_done) if self.bb.exists("goals_done") else set()
            done.add(self.ports["goal_id"])
            self.bb.goals_done = done
        return Status.SUCCESS


class GoToRequiredGoal(GoToNED):
    """Fly to the next outstanding *required* mission goal (read from the
    blackboard ``mission_goals``: {goal_id: (n, e, d)} and ``goals_done``).
    Mission-agnostic: lets a patch learned on one mission transfer to
    another. SUCCESS immediately if no required goal is outstanding."""

    PORTS: PortSpec = {
        "velocity": (float, 0.0, (0.5, 8.0)),
        "timeout_s": (float, 60.0, (5.0, 120.0)),
        "arrive_tol_m": (float, 1.5, (0.5, 5.0)),
    }

    def __init__(self, name: str, **ports: Any):
        super().__init__(name, **ports)
        self.bb.register_key("mission_goals", access=py_trees.common.Access.READ)
        self.bb.register_key("required_goals", access=py_trees.common.Access.READ)
        self.ports.setdefault("north", 0.0); self.ports.setdefault("east", 0.0)
        self.ports.setdefault("down", -5.0); self.ports.setdefault("goal_id", "")
        self._none_outstanding = False

    def initialise(self) -> None:
        done = set(self.bb.goals_done) if self.bb.exists("goals_done") else set()
        outstanding = [g for g in self.bb.required_goals if g not in done]
        if not outstanding:
            self._none_outstanding = True
            self._task = None; self._starter = None
            return
        self._none_outstanding = False
        n, e, d = self.bb.mission_goals[outstanding[0]]
        self.ports["north"], self.ports["east"], self.ports["down"] = float(n), float(e), float(d)
        self.ports["goal_id"] = outstanding[0]
        super().initialise()

    def update(self) -> Status:
        if self._none_outstanding:
            return Status.SUCCESS
        return super().update()

    def port_values(self) -> Dict[str, Any]:
        return {k: self.ports[k] for k in ("velocity", "timeout_s", "arrive_tol_m")}


class MoveRelative(AsyncActionLeaf):
    """Move by a body-independent NED offset from the current position."""

    PORTS: PortSpec = {
        "dn": (float, 0.0, (-20.0, 20.0)),
        "de": (float, 0.0, (-20.0, 20.0)),
        "dd": (float, 0.0, (-10.0, 10.0)),
        "velocity": (float, 2.0, (0.5, 6.0)),
        "timeout_s": (float, 30.0, (5.0, 60.0)),
        "arrive_tol_m": (float, 1.5, (0.5, 5.0)),
    }

    def initialise(self) -> None:
        n, e, d = self.vehicle.ned
        self._target = (n + self.ports["dn"], e + self.ports["de"], d + self.ports["dd"])
        super().initialise()

    async def start_command(self) -> asyncio.Task:
        n, e, d = self._target
        return await self.vehicle.move_to_ned(n, e, d, self.ports["velocity"], self.ports["timeout_s"])

    def complete_by_telemetry(self) -> bool:
        return self.vehicle.dist_to(*self._target) <= self.ports["arrive_tol_m"] and self.tel.get("speed", 1.0) < 0.3

    def on_done(self, result: Any) -> Status:
        return Status.SUCCESS if self.vehicle.dist_to(*self._target) <= self.ports["arrive_tol_m"] else Status.FAILURE


class Hover(VehicleLeaf):
    """Hold position for ``duration_s`` of sim time (issues hover once)."""

    PORTS: PortSpec = {"duration_s": (float, 2.0, (0.5, 30.0))}

    def __init__(self, name: str, **ports: Any):
        super().__init__(name, **ports)
        self._t0 = 0.0
        self._task: Optional[asyncio.Task] = None

    def initialise(self) -> None:
        self._t0 = self.tel.get("t", 0.0)
        self._task = asyncio.get_running_loop().create_task(self.vehicle.hover())
        self.session.pending_starts.append(self._task)

    def update(self) -> Status:
        if self.tel.get("t", 0.0) - self._t0 >= self.ports["duration_s"]:
            return Status.SUCCESS
        return Status.RUNNING


class WaitSimTime(VehicleLeaf):
    """Pure wait (no command). Typical LLM-proposed recovery: wait out a gust."""

    PORTS: PortSpec = {"duration_s": (float, 5.0, (0.5, 60.0))}

    def __init__(self, name: str, **ports: Any):
        super().__init__(name, **ports)
        self._t0 = 0.0

    def initialise(self) -> None:
        self._t0 = self.tel.get("t", 0.0)

    def update(self) -> Status:
        return Status.SUCCESS if self.tel.get("t", 0.0) - self._t0 >= self.ports["duration_s"] else Status.RUNNING


class SetSpeed(VehicleLeaf):
    """Set the blackboard cruise speed used by GoToNED when velocity=0."""

    PORTS: PortSpec = {"speed": (float, 3.0, (0.5, 8.0))}

    def update(self) -> Status:
        self.bb.speed = min(self.ports["speed"], self.vehicle.spec.max_speed_mps)
        return Status.SUCCESS


class InspectRegion(VehicleLeaf):
    """Placeholder perception leaf (scout only). Headless: reports 'clear'
    after ``duration_s``; the Unreal host replaces this with a camera/VLM
    call. Writes ``inspection_result`` to the blackboard."""

    PORTS: PortSpec = {
        "north": (float, 0.0, None),
        "east": (float, 0.0, None),
        "duration_s": (float, 3.0, (1.0, 15.0)),
    }

    def __init__(self, name: str, **ports: Any):
        super().__init__(name, **ports)
        self.bb.register_key("inspection_result", access=py_trees.common.Access.WRITE)
        self._t0 = 0.0

    def initialise(self) -> None:
        self._t0 = self.tel.get("t", 0.0)

    def update(self) -> Status:
        if self.tel.get("t", 0.0) - self._t0 >= self.ports["duration_s"]:
            self.bb.inspection_result = "clear"
            return Status.SUCCESS
        return Status.RUNNING


class RequestScout(InspectRegion):
    """Carrier/relay substitution for InspectRegion: delegate to a scout.
    Step-1 implementation: same behaviour as InspectRegion but logged as a
    delegation; fleet-level dispatch arrives with the Patch Manager."""


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

class IsAirborne(VehicleLeaf):
    PORTS: PortSpec = {"min_alt_m": (float, 1.0, (0.3, 3.0))}

    def update(self) -> Status:
        return Status.SUCCESS if self.tel.get("alt", 0.0) >= self.ports["min_alt_m"] and not self.tel.get("landed") else Status.FAILURE


class IsBatteryOk(VehicleLeaf):
    PORTS: PortSpec = {"min_pct": (float, 25.0, (10.0, 60.0))}

    def update(self) -> Status:
        ok = self.tel.get("battery_pct", 100.0) >= self.ports["min_pct"]
        if not ok:
            self.feedback_message = f"battery {self.tel.get('battery_pct', 0):.1f}% < {self.ports['min_pct']}%"
        return Status.SUCCESS if ok else Status.FAILURE


class IsWindOk(VehicleLeaf):
    PORTS: PortSpec = {"max_wind_mps": (float, 8.0, (2.0, 15.0))}

    def update(self) -> Status:
        return Status.SUCCESS if self.tel.get("wind_speed", 0.0) <= self.ports["max_wind_mps"] else Status.FAILURE


class IsGpsOk(VehicleLeaf):
    def update(self) -> Status:
        return Status.SUCCESS if self.tel.get("gps_ok", True) else Status.FAILURE


class IsLinkOk(VehicleLeaf):
    PORTS: PortSpec = {"min_quality": (float, 0.3, (0.05, 0.9))}

    def update(self) -> Status:
        return Status.SUCCESS if self.tel.get("link_quality", 1.0) >= self.ports["min_quality"] else Status.FAILURE


class IsInsideGeofence(VehicleLeaf):
    PORTS: PortSpec = {"margin_m": (float, 0.0, (0.0, 20.0))}

    def update(self) -> Status:
        g = self.session.geofence
        m = self.ports["margin_m"]
        n, e, d = self.vehicle.ned
        alt = -d
        ok = (g["n_min"] + m <= n <= g["n_max"] - m and g["e_min"] + m <= e <= g["e_max"] - m
              and alt <= g["alt_max"] - m)
        return Status.SUCCESS if ok else Status.FAILURE
