"""Simulation session: one runtime instance, one scene, N vehicles.

Design notes
------------
* The BT is ticked from a single asyncio loop. Leaves start Project AirSim
  async commands as ``asyncio.Task`` objects and poll ``.done()`` on each
  tick, so the tree never blocks on the simulator.
* Telemetry (pose, velocity, battery, GPS, wind) is read into a plain dict
  ``Vehicle.telemetry`` once per tick by ``SimSession.refresh()``; leaves and
  monitors read from that dict, never from the client directly. This makes
  the tick-trace a pure function of the telemetry sequence, which is what the
  determinism test hashes.
* All client calls go through this module so the sim backend can be swapped
  (Unreal host, PX4) without touching the BT code.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import LandedState
from projectairsim.utils import projectairsim_log

from .config import ClassSpec, FleetSpec, default_fleet, write_config_dir
from .launch_runtime import instance_ports, launch


@dataclass
class Vehicle:
    name: str
    spec: ClassSpec
    drone: Drone
    telemetry: Dict[str, Any] = field(default_factory=dict)
    home_ned: tuple = (0.0, 0.0, 0.0)
    # Client-side simulated link quality in [0,1]; the headless host has no
    # comms model. Failure injection F8 drives this.
    link_quality: float = 1.0
    # Client-side GPS health flag; F2 drives this (and perturbs reported pos).
    gps_ok: bool = True
    gps_noise_m: float = 0.0
    _battery_msg: Optional[Dict] = None
    _gps_msg: Optional[Dict] = None

    # ----- commands (return asyncio.Task) --------------------------------
    async def takeoff(self, timeout_sec: float = 20.0) -> asyncio.Task:
        return await self.drone.takeoff_async(timeout_sec=timeout_sec)

    async def land(self, timeout_sec: float = 60.0) -> asyncio.Task:
        return await self.drone.land_async(timeout_sec=timeout_sec)

    async def go_home(self, velocity: float, timeout_sec: float = 120.0) -> asyncio.Task:
        return await self.drone.go_home_async(timeout_sec=timeout_sec, velocity=velocity)

    async def hover(self) -> asyncio.Task:
        return await self.drone.hover_async()

    async def move_to_ned(self, n: float, e: float, d: float, velocity: float, timeout_sec: float) -> asyncio.Task:
        v = min(velocity, self.spec.max_speed_mps)
        return await self.drone.move_to_position_async(
            north=n, east=e, down=d, velocity=v, timeout_sec=timeout_sec
        )

    def cancel(self) -> None:
        try:
            self.drone.cancel_last_task()
        except Exception as exc:  # noqa: BLE001
            projectairsim_log().warning(f"{self.name}: cancel_last_task failed: {exc}")

    # ----- convenience ------------------------------------------------------
    @property
    def ned(self) -> tuple:
        p = self.telemetry.get("pos", (0.0, 0.0, 0.0))
        return p

    def dist_to(self, n: float, e: float, d: float) -> float:
        pn, pe, pd = self.ned
        return math.sqrt((pn - n) ** 2 + (pe - e) ** 2 + (pd - d) ** 2)


class SimSession:
    """Owns the runtime process (optional), client, world and vehicles."""

    def __init__(
        self,
        fleet: Optional[FleetSpec] = None,
        instance: int = 0,
        launch_runtime: bool = True,
        address: str = "127.0.0.1",
        tick_dt: float = 0.05,
        lockstep: bool = False,
    ):
        """``lockstep=True`` pauses the simulator and advances it by exactly
        ``tick_dt`` of sim time per BT tick (deterministic, and as fast as the
        physics can run). ``False`` runs the simulator in real time and the
        tick loop sleeps ``tick_dt`` of wall time between ticks."""
        self.fleet = fleet or default_fleet(1)
        self.instance = instance
        self.launch_runtime = launch_runtime
        self.address = address
        self.tick_dt = tick_dt
        self.lockstep = lockstep
        self.proc: Optional[subprocess.Popen] = None
        self.client: Optional[ProjectAirSimClient] = None
        self.world: Optional[World] = None
        self.vehicles: Dict[str, Vehicle] = {}
        self.wind: tuple = (0.0, 0.0, 0.0)
        self.geofence: Dict[str, float] = {  # NED box, metres; z is "down"
            "n_min": -200.0, "n_max": 200.0, "e_min": -200.0, "e_max": 200.0,
            "alt_min": 0.0, "alt_max": 60.0,
        }
        self.min_separation_m = 2.5
        # Command-start tasks created during the current tick (see sleep_tick).
        self.pending_starts: List[asyncio.Task] = []

    # ----- lifecycle ------------------------------------------------------
    def start(self, connect_retries: int = 40) -> "SimSession":
        if self.launch_runtime:
            self.proc = launch(self.instance)
        topics, services = instance_ports(self.instance)
        self.client = ProjectAirSimClient(self.address, port_topics=topics, port_services=services)
        last_exc: Optional[Exception] = None
        for _ in range(connect_retries):
            try:
                self.client.connect()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.25)
        if last_exc is not None:
            raise RuntimeError(f"could not connect to runtime on {topics}/{services}: {last_exc}")
        self.load_scene()
        return self

    def load_scene(self) -> None:
        cfg_dir = write_config_dir(self.fleet, os.path.join(tempfile.gettempdir(), f"scbt_cfg_{self.instance}"))
        # The stock constructor validates the configs against the schema and
        # loads the scene; the runtime has no "SceneBasicDrone" default.
        self.world = World(self.client, "scene.jsonc", delay_after_load_sec=0.5, sim_config_path=cfg_dir)
        self.vehicles = {}
        for vs in self.fleet.vehicles:
            drone = Drone(self.client, self.world, vs.name)
            v = Vehicle(name=vs.name, spec=self.fleet.classes[vs.vehicle_class], drone=drone)
            self._subscribe_telemetry(v)
            drone.enable_api_control()
            drone.arm()
            self.vehicles[vs.name] = v
        self.set_wind(0.0, 0.0, 0.0)
        if self.lockstep:
            self.world.pause()
            # Let sensors publish once so telemetry is populated.
            self.world.continue_for_sim_time(int(0.5e9), wait_until_complete=True)
        else:
            time.sleep(0.3)
        self.refresh()
        for v in self.vehicles.values():
            v.home_ned = v.ned

    def _subscribe_telemetry(self, v: Vehicle) -> None:
        s = v.drone.sensors
        if "Battery" in s:
            self.client.subscribe(s["Battery"]["battery"], lambda _t, m, v=v: setattr(v, "_battery_msg", m))
        if "GPS" in s:
            self.client.subscribe(s["GPS"]["gps"], lambda _t, m, v=v: setattr(v, "_gps_msg", m))

    def close(self) -> None:
        try:
            if self.client is not None:
                for v in self.vehicles.values():
                    try:
                        v.drone.disarm()
                        v.drone.disable_api_control()
                    except Exception:  # noqa: BLE001
                        pass
                self.client.disconnect()
        finally:
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

    # ----- world controls -------------------------------------------------
    def set_wind(self, n: float, e: float, d: float) -> None:
        self.wind = (n, e, d)
        self.world.set_wind_velocity(n, e, d)

    def sim_time_s(self) -> float:
        return self.world.get_sim_time() / 1e9

    def pause(self) -> None:
        if not self.lockstep:
            self.world.pause()

    def resume(self) -> None:
        if not self.lockstep:
            self.world.resume()

    # ----- telemetry ------------------------------------------------------
    def refresh(self) -> None:
        """Pull fresh telemetry for every vehicle into ``Vehicle.telemetry``."""
        t = self.sim_time_s()
        for v in self.vehicles.values():
            k = v.drone.get_ground_truth_kinematics()
            p = k["pose"]["position"]
            vel = k["twist"]["linear"]
            pos = (p["x"], p["y"], p["z"])
            if v.gps_noise_m > 0.0:
                # Deterministic pseudo-noise derived from sim time (reproducible).
                ph = (t * 7.3) % (2 * math.pi)
                pos = (pos[0] + v.gps_noise_m * math.cos(ph), pos[1] + v.gps_noise_m * math.sin(ph), pos[2])
            batt = v._battery_msg or {}
            v.telemetry = {
                "t": t,
                "pos": pos,
                "vel": (vel["x"], vel["y"], vel["z"]),
                "speed": math.sqrt(vel["x"] ** 2 + vel["y"] ** 2 + vel["z"] ** 2),
                "alt": -p["z"],
                "battery_pct": float(batt.get("battery_pct_remaining", 100.0)),
                "battery_time_s": float(batt.get("estimated_time_remaining", 1e9)),
                "gps_ok": v.gps_ok,
                "link_quality": v.link_quality,
                "wind": self.wind,
                "wind_speed": math.sqrt(sum(w * w for w in self.wind)),
                "landed": v.drone.get_landed_state() == LandedState.LANDED,
            }

    def separation_violations(self) -> List[tuple]:
        names = list(self.vehicles)
        out = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = self.vehicles[names[i]], self.vehicles[names[j]]
                if a.telemetry.get("landed") or b.telemetry.get("landed"):
                    continue
                if a.dist_to(*b.ned) < self.min_separation_m:
                    out.append((a.name, b.name))
        return out

    # ----- tick loop helper ----------------------------------------------
    async def sleep_tick(self) -> None:
        """Advance to the next BT tick. In lockstep mode this steps the
        simulator by ``tick_dt`` sim seconds; the asyncio loop is then yielded
        so pending command responses are processed before the next tick."""
        if self.lockstep:
            # Determinism: every command issued during this tick must reach
            # the server BEFORE the step. (1) drain the asyncio sends;
            # (2) a synchronous round-trip on the same request socket acts as
            # a barrier so the server has processed them; (3) step.
            await self._drain_pending_starts()
            self.world.get_sim_time()
            self.world.continue_for_sim_time(int(self.tick_dt * 1e9), wait_until_complete=True)
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(self.tick_dt)

    async def _drain_pending_starts(self, timeout_s: float = 5.0) -> None:
        pending = [t for t in self.pending_starts if not t.done()]
        self.pending_starts = []
        if pending:
            await asyncio.wait(pending, timeout=timeout_s)
        for _ in range(3):           # let any follow-up callbacks run
            await asyncio.sleep(0)
