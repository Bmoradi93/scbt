"""``MissionCritical`` decorator: the hook where failure events are raised.

Semantics
---------
* Passes the child's status through, except that when the child returns
  FAILURE and patching is enabled, it returns RUNNING for that tick and sets
  ``failure_pending`` on the blackboard. The episode runner sees the flag
  after the tick, pauses the simulator, and either applies a patch (the
  child restarts on the next tick) or sets ``give_up`` (the decorator then
  returns FAILURE without ticking the child, and the root Fallback proceeds
  to the hand-coded abort branch).
* With ``patching_enabled=False`` (frozen phase, baselines B0) it is a pure
  pass-through, so a crystallized tree contains no learning hook.
"""

from __future__ import annotations

import typing

import py_trees
from py_trees.common import Status


class MissionCritical(py_trees.decorators.Decorator):
    def __init__(self, name: str, child: py_trees.behaviour.Behaviour, patching_enabled: bool = True):
        super().__init__(name=name, child=child)
        self.patching_enabled = patching_enabled
        from ..bt.leaves import NAMESPACE
        self.bb = self.attach_blackboard_client(name=name, namespace=NAMESPACE.get())
        self.bb.register_key("failure_pending", access=py_trees.common.Access.WRITE)
        self.bb.register_key("give_up", access=py_trees.common.Access.WRITE)
        self.bb.failure_pending = False
        self.bb.give_up = False
        self.failures_intercepted = 0

    def tick(self) -> typing.Iterator[py_trees.behaviour.Behaviour]:
        if self.patching_enabled and self.bb.give_up:
            if self.decorated.status == Status.RUNNING:
                self.decorated.stop(Status.INVALID)
            self.status = Status.FAILURE
            yield self
            return
        yield from super().tick()

    def update(self) -> Status:
        st = self.decorated.status
        if st == Status.FAILURE and self.patching_enabled:
            self.failures_intercepted += 1
            self.bb.failure_pending = True
            return Status.RUNNING
        return st
