"""Prompt material for the proposer agents: manifest, grammar, examples."""

from __future__ import annotations

from typing import Iterable, List

from ..bt.registry import LEAF_REGISTRY
from ..observer.events import FailureEvent
from ..sim.config import ClassSpec

GRAMMAR_TEXT = """\
You may emit exactly ONE patch, a local edit at an anchor node `a` of the current tree:
  WrapFallback(a, S)   : a  ->  Fallback(a, S)        try a first; if it fails run recovery S
  GuardWith(a, S)      : a  ->  Sequence(S, a)        establish a precondition S before a
  WrapRetry(a, N)      : a  ->  Retry(N)(a)           retry a up to N times (no subtree)
  InsertBefore(a, S)   : parent(a): [.., S, a, ..]
  InsertAfter(a, S)    : parent(a): [.., a, S, ..]
  ReplaceSubtree(a, S) : a  ->  S                     deletes existing logic; needs a strong rationale

Subtree XML dialect (BehaviorTree.CPP-like):
  control:   <Sequence> <ReactiveSequence> <Fallback> <ReactiveFallback> <Parallel policy="all|one">
  decorator: <Inverter> <ForceSuccess> <ForceFailure> <Retry num_attempts="N"> <Repeat num_cycles="N">
  leaves:    see the capability manifest; only those are allowed.
Semantics: Sequence/Fallback have memory (resume at the running child); Reactive* re-evaluate all
children every tick (use for guards). A condition leaf that fails returns FAILURE immediately (no
RUNNING), so Retry around a bare condition burns all attempts in consecutive ticks -- put a
WaitSimTime in front of it. Hold position with Hover (issues a hover command) or WaitSimTime (no command).

Patches must be mission-agnostic: never copy literal coordinates from the event. Use {@port} to
reference a port of the anchor leaf (e.g. north="{@north}" goal_id="{@goal_id}"), and
GoToRequiredGoal to fly to the next outstanding required mission goal.
Numeric ports you are unsure about MUST be written as placeholders {name} and declared in tunable_ports
with [min,max] and an initial guess; they will be optimized in simulation. Prefer 1-4 tunables.
"""


def render_manifest(cls: ClassSpec) -> str:
    lines = [f"Capability manifest for class '{cls.name}' (max_speed {cls.max_speed_mps} m/s):"]
    for tag in cls.leaves:
        leaf = LEAF_REGISTRY[tag]
        ports = []
        for pname, (ptype, default, bounds) in leaf.PORTS.items():
            b = f" in [{bounds[0]},{bounds[1]}]" if bounds else ""
            ports.append(f"{pname}:{ptype.__name__}={default}{b}")
        doc = (leaf.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  <{tag} {' '.join(ports)}/>  -- {doc}")
    if cls.substitutions:
        lines.append("  substitutions: " + ", ".join(f"{k} -> {v}" for k, v in cls.substitutions.items()))
    return "\n".join(lines)


def render_event(ev: FailureEvent) -> str:
    trace = "\n".join(f"    tick {r.tick}: {r.node} ({r.tag}) -> {r.status}" for r in ev.tick_trace)
    return (
        f"Vehicle {ev.vehicle} (class {ev.vehicle_class}) failed at node '{ev.failed_node}' of type "
        f"{ev.failed_tag} with ports {ev.failed_ports}.\n"
        f"Leaf feedback: {ev.feedback or '(none)'}\n"
        f"Telemetry at failure: {ev.blackboard}\n"
        f"Context: {ev.context_text()}\n"
        f"Recent tick trace of the enclosing subtree:\n{trace}\n"
        f"Enclosing subtree XML (failed node marked FAILED=\"true\"):\n{ev.subtree_xml}\n"
    )


def render_examples(examples: Iterable[tuple]) -> str:
    """examples: iterable of (context_text, patch_xml, outcome_str). The
    outcome string carries the admission status AND the success record so the
    model can see which earlier patches did not work."""
    out: List[str] = []
    for i, (ctx, xml, outcome) in enumerate(examples, 1):
        out.append(f"Example {i} [{outcome}]\n  context: {ctx}\n  patch: {xml}")
    return "\n".join(out) if out else "(no examples yet)"


OBJECTIVE_TEXT = """\
SUCCESS CRITERION (this is what the recovery is scored on): the mission counts as successful only if
ALL required goals are reached AND the vehicle lands. A recovery that merely returns to launch while a
required goal is still outstanding scores ZERO, the same as a crash-free abort. Prefer recoveries that
still complete the outstanding required goal(s) under a reduced but safe margin (e.g. a lower battery
reserve such as 10-15 %, reduced speed, skipping OPTIONAL goals), and only then return and land.
Hard safety invariants checked in simulation: stay inside the geofence, altitude within limits, speed
within the class limit, battery >= 10 % at landing. Do not remove or bypass guards without replacing
them with an equivalent guard at a justified threshold.
"""

ANALYST_INSTRUCTION = """\
You are the ContextAnalyst for an autonomous drone fleet. A node of a vehicle's Behavior Tree has
failed. Using only the telemetry, leaf feedback, tick trace and subtree given, diagnose the failure
and propose a recovery strategy expressible with the vehicle's capability manifest. Choose the anchor:
normally the failed leaf itself, or its parent composite if the recovery must bypass several siblings.
Respect the SUCCESS CRITERION given in the message: a recovery that abandons a required goal is a failure.
"""

PROPOSER_INSTRUCTION = """\
You are the PatchProposer. Turn the diagnosis into exactly one patch in the grammar below. The
subtree may use ONLY leaves from the capability manifest with ONLY their listed ports. Keep the edit
minimal and local. Numeric values you are unsure about must be tunable placeholders declared in
tunable_ports (syntax in the grammar section of the message). Do not change the anchor's own ports.
If a previous attempt was rejected, the verifier's message is given: fix exactly that problem.
NOTE: ADK treats curly braces in this instruction as template variables, so the grammar with its
brace syntax is given in the user message, not here.
"""


REGENERATE_INSTRUCTION = """\
You are the TreeRegenerator (baseline B1). A node of the vehicle's Behavior Tree has failed. Instead
of a local patch, write a COMPLETE replacement for the mission subtree: output exactly one patch with
operator ReplaceSubtree and anchor mission_seq, whose subtree is the entire new mission sequence
(takeoff, guarded flight with all mission waypoints and guards, recovery logic you consider
necessary, landing). The subtree may use ONLY leaves from the capability manifest with ONLY their
listed ports. Numeric values you are unsure about must be tunable placeholders declared in
tunable_ports (syntax in the grammar section of the message). Respect the SUCCESS CRITERION.
If a previous attempt was rejected, the verifier's message is given: fix exactly that problem.
"""
