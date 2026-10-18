"""Patch policies for the episode runner.

``no_patch``      - baseline: never patch (mission fails -> abort branch).
``handwritten``   - step-1 stand-in for the Patch Manager: a fixed lookup
                    from (failed node type, telemetry) to a patch. It exists
                    to exercise the hot-patch mechanism end to end; it is
                    *not* a baseline in the paper.
"""

from __future__ import annotations

from typing import Optional

from ..observer.events import FailureEvent
from ..patch.grammar import Patch, parse_patch_xml


def no_patch(event: FailureEvent, runner) -> Optional[Patch]:
    return None


def _wind_recovery(event: FailureEvent) -> Patch:
    # Hold position, then poll "wind ok" with a wait between attempts and
    # retry the same waypoint slower. (A first version used Retry directly
    # on IsWindOk; it burned its attempts in three ticks. The wait leaf is
    # what makes the retry meaningful -- the kind of structural detail the
    # verifier/bandit must catch when an LLM proposes it.)
    p = event.failed_ports
    xml = f"""
    <patch operator="WrapFallback" anchor="{event.failed_node}">
      <subtree>
        <Sequence>
          <Hover duration_s="{{hover_s}}"/>
          <Retry num_attempts="6">
            <Sequence>
              <WaitSimTime duration_s="{{wait_s}}"/>
              <IsWindOk max_wind_mps="{{max_wind}}"/>
              <GoToNED north="{{@north}}" east="{{@east}}" down="{{@down}}" goal_id="{{@goal_id}}"
                       velocity="{{v}}" timeout_s="60" arrive_tol_m="{{@arrive_tol_m}}"/>
            </Sequence>
          </Retry>
        </Sequence>
      </subtree>
      <tunable_ports>
        <port name="hover_s"  min="2"   max="30" initial="10"/>
        <port name="wait_s"   min="1"   max="20" initial="5"/>
        <port name="max_wind" min="2"   max="15" initial="8"/>
        <port name="v"        min="0.5" max="6"  initial="2"/>
      </tunable_ports>
      <rationale>Hold position until wind subsides, polling periodically, then retry the leg at reduced speed.</rationale>
    </patch>"""
    patch = parse_patch_xml(xml)
    patch.patch_id = "hw_wind"
    return patch


def _battery_recovery(event: FailureEvent, runner) -> Optional[Patch]:
    """Degraded mode: the 25% reserve guard tripped before the required goal.
    Complete only the required leg under a lower reserve at an efficient
    speed, then return and let the base tree land. Optional legs are
    skipped. Safety rests on the lower guard + the admission invariant
    (battery >= 10% at landing), not on this policy."""
    from ..bt.registry import find_node
    required = [g for g in runner.plan.required_goals if g not in set(runner.bb.goals_done)]
    if not required:
        return None
    xml = """
    <patch operator="WrapFallback" anchor="guarded_flight">
      <subtree>
        <ReactiveSequence>
          <IsBatteryOk min_pct="{reserve}"/>
          <Sequence>
            <GoToRequiredGoal velocity="{v}" timeout_s="90" arrive_tol_m="1.5"/>
            <ReturnToLaunch velocity="{v}" timeout_s="120" arrive_tol_m="2"/>
          </Sequence>
        </ReactiveSequence>
      </subtree>
      <tunable_ports>
        <port name="reserve" min="10" max="24" initial="12"/>
        <port name="v"       min="1"  max="6"  initial="4"/>
      </tunable_ports>
      <rationale>Reserve guard tripped early; finish the required delivery under a reduced reserve, skip optional legs, return home.</rationale>
    </patch>"""
    patch = parse_patch_xml(xml)
    patch.patch_id = "hw_batt"
    return patch


def handwritten(event: FailureEvent, runner) -> Optional[Patch]:
    """Lookup by failed node type. Returns None (give up) if unknown."""
    if event.failed_tag == "GoToNED" and event.blackboard.get("wind_speed", 0.0) > 5.0:
        return _wind_recovery(event)
    if event.failed_tag == "IsBatteryOk":
        return _battery_recovery(event, runner)
    return None
