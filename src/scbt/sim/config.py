"""Programmatic scene/robot configuration for the SCBT fleet.

We derive each vehicle's robot config from the Project AirSim example
``robot_quadrotor_fastphysics_battery_simple.jsonc`` (Fast Physics + Simple
Flight + simple battery), strip rendered sensors (headless runtime has no
cameras), enable GPS, and apply per-class physical differences.

Vehicle classes (paper Sec. V-A, headless subset):
  scout   - nominal mass, nominal speed cap
  carrier - heavier frame (payload), lower speed cap, larger battery drain
  relay   - nominal mass, link-quality model simulated client-side

Speed caps are *not* a Simple Flight setting; they are enforced by the BT
leaves through the capability manifest (``ClassSpec.max_speed``).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import commentjson


def _find_project_airsim_root() -> str:
    """Locate a Project AirSim checkout: $PROJECTAIRSIM_ROOT if set, else the
    nearest ancestor of this file containing ``client/python``, else a sibling
    ``ProjectAirSim`` directory next to this repo (the standalone-repo layout)."""
    env = os.environ.get("PROJECTAIRSIM_ROOT")
    if env:
        return os.path.normpath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "client", "python")):
            return d
        d = os.path.dirname(d)
    sibling = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "ProjectAirSim"))
    return sibling


# Path to the Project AirSim example configs (shipped with the client package).
# Override with $PROJECTAIRSIM_ROOT if this repo is checked out next to, rather
# than inside, a Project AirSim checkout.
_EXAMPLES_DIR = os.path.normpath(
    os.path.join(_find_project_airsim_root(), "client", "python", "example_user_scripts", "sim_config")
)
_BASE_ROBOT_CONFIG = "robot_quadrotor_fastphysics_battery_simple.jsonc"

STEP_NS = 3_000_000  # 3 ms physics step, as in the shipped scenes


@dataclass(frozen=True)
class ClassSpec:
    """Capability/physical profile of a vehicle class."""

    name: str
    frame_mass_kg: float
    max_speed_mps: float
    max_climb_mps: float
    battery_capacity: float          # sim units (see battery sensor config)
    battery_on_start: float
    battery_drain_rate: float
    leaves: List[str] = field(default_factory=list)   # allowed leaf node types
    substitutions: Dict[str, str] = field(default_factory=dict)


# Leaf types registered in scbt.bt.leaves. Kept here so the manifest and the
# implementation cannot silently diverge (tests assert they match).
COMMON_LEAVES = [
    "Takeoff", "Land", "ReturnToLaunch", "GoToNED", "GoToRequiredGoal", "Hover", "MoveRelative",
    "IsAirborne", "IsBatteryOk", "IsWindOk", "IsGpsOk", "IsInsideGeofence",
    "IsLinkOk", "WaitSimTime", "SetSpeed",
]

DEFAULT_CLASSES: Dict[str, ClassSpec] = {
    "scout": ClassSpec(
        name="scout", frame_mass_kg=1.0, max_speed_mps=6.0, max_climb_mps=2.0,
        battery_capacity=36000, battery_on_start=30000, battery_drain_rate=60.0,
        leaves=COMMON_LEAVES + ["InspectRegion"],
    ),
    "carrier": ClassSpec(
        name="carrier", frame_mass_kg=1.6, max_speed_mps=3.5, max_climb_mps=1.2,
        battery_capacity=36000, battery_on_start=30000, battery_drain_rate=90.0,
        leaves=COMMON_LEAVES + ["RequestScout"],
        substitutions={"InspectRegion": "RequestScout"},
    ),
    "relay": ClassSpec(
        name="relay", frame_mass_kg=1.0, max_speed_mps=5.0, max_climb_mps=2.0,
        battery_capacity=36000, battery_on_start=30000, battery_drain_rate=60.0,
        leaves=COMMON_LEAVES + ["RequestScout"],
        substitutions={"InspectRegion": "RequestScout"},
    ),
}


@dataclass(frozen=True)
class VehicleSpec:
    name: str                 # actor name in the scene, e.g. "Drone1"
    vehicle_class: str        # key into DEFAULT_CLASSES
    origin_ned: tuple         # (north, east, down) spawn position, metres


@dataclass(frozen=True)
class FleetSpec:
    vehicles: List[VehicleSpec]
    scene_id: str = "SceneSCBT"
    home_geo: tuple = (47.641468, -122.140165, 122.0)
    # Sim-time / wall-time ratio for the steppable clock (step-ns is fixed at
    # STEP_NS; the real-time execution period is STEP_NS / time_scale).
    time_scale: float = 1.0
    classes: Dict[str, ClassSpec] = field(default_factory=lambda: dict(DEFAULT_CLASSES))

    def vehicle(self, name: str) -> VehicleSpec:
        for v in self.vehicles:
            if v.name == name:
                return v
        raise KeyError(name)

    def class_of(self, name: str) -> ClassSpec:
        return self.classes[self.vehicle(name).vehicle_class]


def default_fleet(n: int = 1, time_scale: float = 1.0) -> FleetSpec:
    """Fleet used in the paper: 2 scouts, 1 carrier, 1 relay (truncated to n)."""
    roster = [
        VehicleSpec("Drone1", "scout", (0.0, 0.0, -0.0)),
        VehicleSpec("Drone2", "scout", (0.0, 4.0, -0.0)),
        VehicleSpec("Drone3", "carrier", (4.0, 0.0, -0.0)),
        VehicleSpec("Drone4", "relay", (4.0, 4.0, -0.0)),
    ]
    return FleetSpec(vehicles=roster[:n], time_scale=time_scale)


def _load_base_robot_config() -> Dict:
    with open(os.path.join(_EXAMPLES_DIR, _BASE_ROBOT_CONFIG)) as f:
        return commentjson.load(f)


def build_robot_config(cls: ClassSpec, base: Optional[Dict] = None) -> Dict:
    cfg = copy.deepcopy(base if base is not None else _load_base_robot_config())

    # Headless runtime: drop cameras, keep IMU, enable GPS, configure battery.
    sensors = []
    for s in cfg.get("sensors", []):
        if s["type"] == "camera":
            continue
        if s["type"] == "gps":
            s["enabled"] = True
        if s["type"] == "battery":
            # Robot.get_battery_state() hardcodes the sensor path ".../Battery".
            s["id"] = "Battery"
            s["total-battery-capacity"] = cls.battery_capacity
            s["battery-capacity-on-start"] = cls.battery_on_start
            s["battery-drain-rate-on-start"] = cls.battery_drain_rate
        sensors.append(s)
    if not any(s["type"] == "gps" for s in sensors):
        sensors.append({"id": "GPS", "type": "gps", "enabled": True, "parent-link": "Frame"})
    if not any(s["type"] == "barometer" for s in sensors):
        sensors.append({"id": "Baro", "type": "barometer", "enabled": True, "parent-link": "Frame"})
    cfg["sensors"] = sensors

    # Class mass on the frame link.
    for link in cfg["links"]:
        if link["name"] == "Frame":
            link["inertial"]["mass"] = cls.frame_mass_kg
    return cfg


def build_scene_config(fleet: FleetSpec, pause_on_start: bool = False, robot_config_as_file: bool = False) -> Dict:
    """Return a scene config dict. With ``robot_config_as_file=False`` the
    robot configs are inlined (ready for ``World.load_scene``); with ``True``
    they are referenced by file name (``robot_<class>.jsonc``) for use with
    ``write_config_dir`` + the stock ``World(client, scene, sim_config_path)``."""
    base = _load_base_robot_config()
    actors = []
    for v in fleet.vehicles:
        n, e, d = v.origin_ned
        actors.append(
            {
                "type": "robot",
                "name": v.name,
                "origin": {"xyz": f"{n} {e} {d}", "rpy-deg": "0 0 0"},
                "robot-config": (f"robot_{v.vehicle_class}.jsonc" if robot_config_as_file
                                 else build_robot_config(fleet.classes[v.vehicle_class], base)),
            }
        )
    lat, lon, alt = fleet.home_geo
    return {
        "id": fleet.scene_id,
        "actors": actors,
        "clock": {
            "type": "steppable",
            "step-ns": STEP_NS,
            "real-time-update-rate": max(1, int(STEP_NS / fleet.time_scale)),
            "pause-on-start": pause_on_start,
        },
        "home-geo-point": {"latitude": lat, "longitude": lon, "altitude": alt},
        "segmentation": {"initialize-ids": True, "ignore-existing": False, "use-owner-name": True},
        "scene-type": "UnrealNative",
    }


def write_config_dir(fleet: FleetSpec, out_dir: str) -> str:
    """Write ``scene.jsonc`` and one ``robot_<class>.jsonc`` per class into
    ``out_dir`` so the stock ``World`` constructor can load and validate them."""
    import json
    os.makedirs(out_dir, exist_ok=True)
    base = _load_base_robot_config()
    for cname, cls in fleet.classes.items():
        with open(os.path.join(out_dir, f"robot_{cname}.jsonc"), "w") as f:
            json.dump(build_robot_config(cls, base), f, indent=1)
    with open(os.path.join(out_dir, "scene.jsonc"), "w") as f:
        json.dump(build_scene_config(fleet, robot_config_as_file=True), f, indent=1)
    return out_dir
