"""Tick-trace recording and the determinism hash (paper Sec. V-F)."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import py_trees
from py_trees.common import Status

from .events import TickRecord


class TickTrace(py_trees.visitors.VisitorBase):
    """Visitor that records (tick, node, status) for every node visited.

    ``full=False`` records only status *changes* per node (compact, what the
    determinism hash uses). ``full=True`` records every visit.
    """

    def __init__(self, full: bool = False):
        super().__init__(full=False)
        self.records: List[TickRecord] = []
        self.full_mode = full
        self.tick = 0
        self.t = 0.0
        self._last: Dict[str, Status] = {}
        self._visited_this_tick: List[py_trees.behaviour.Behaviour] = []

    def initialise(self) -> None:
        self._visited_this_tick = []

    def run(self, behaviour: py_trees.behaviour.Behaviour) -> None:
        self._visited_this_tick.append(behaviour)
        st = behaviour.status
        if self.full_mode or self._last.get(behaviour.name) != st:
            self.records.append(
                TickRecord(self.tick, self.t, behaviour.name, getattr(behaviour, "xml_tag", type(behaviour).__name__), st.name)
            )
        self._last[behaviour.name] = st

    def visited_this_tick(self) -> List[py_trees.behaviour.Behaviour]:
        return list(self._visited_this_tick)

    def culprit_leaf(self) -> Optional[py_trees.behaviour.Behaviour]:
        """Deepest leaf visited this tick with status FAILURE (last wins)."""
        c = None
        for b in self._visited_this_tick:
            if not b.children and b.status == Status.FAILURE:
                c = b
        return c

    def tail(self, k: int, within: Optional[set] = None) -> List[TickRecord]:
        recs = self.records if within is None else [r for r in self.records if r.node in within]
        return recs[-k:]


def trace_hash(records: List[TickRecord]) -> str:
    """SHA-256 over the ordered (node, status) sequence; time is excluded so
    wall-clock jitter cannot affect the hash, but tick index is included."""
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r.tick}:{r.node}:{r.status}\n".encode())
    return h.hexdigest()
