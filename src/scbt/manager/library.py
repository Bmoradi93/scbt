"""Fleet patch library L (paper Sec. IV-C), JSON-persisted."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..patch.grammar import Patch, parse_patch_xml
from .embedding import cosine


@dataclass
class LibraryEntry:
    patch_id: str
    patch_xml: str
    phi: List[float]                      # full context embedding of the originating event
    phi_retrieval: List[float]            # normalized retrieval block
    origin_class: str
    theta: Dict[str, List[float]] = field(default_factory=dict)      # per class
    admitted_for: List[str] = field(default_factory=list)            # classes passing admission
    status: str = "proposed"              # proposed | admitted | rejected
    reject_reason: str = ""
    provenance: Dict = field(default_factory=dict)
    uses: int = 0
    successes: int = 0
    created: float = field(default_factory=time.time)

    def patch(self) -> Patch:
        p = parse_patch_xml(self.patch_xml)
        p.patch_id = self.patch_id
        return p

    def context_text(self) -> str:
        return self.provenance.get("context_text", "")


class PatchLibrary:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.entries: Dict[str, LibraryEntry] = {}
        self._n = 0
        if path and os.path.exists(path):
            self.load()

    # ----- persistence --------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"n": self._n, "entries": [asdict(e) for e in self.entries.values()]}, f, indent=1)

    def load(self) -> None:
        with open(self.path) as f:
            d = json.load(f)
        self._n = d.get("n", 0)
        self.entries = {e["patch_id"]: LibraryEntry(**e) for e in d["entries"]}

    # ----- CRUD -----------------------------------------------------------
    def new_id(self, prefix: str = "p") -> str:
        self._n += 1
        return f"{prefix}{self._n:04d}"

    def add(self, entry: LibraryEntry) -> None:
        self.entries[entry.patch_id] = entry
        self.save()

    def get(self, patch_id: str) -> LibraryEntry:
        return self.entries[patch_id]

    # ----- retrieval ------------------------------------------------------
    def retrieve(self, phi_retrieval: np.ndarray, rho: float, admitted_only: bool = True,
                 for_class: Optional[str] = None) -> List[tuple]:
        """Return [(entry, similarity)] with similarity >= rho, best first.
        ``for_class`` filters to entries executable by that class (admitted
        for it, or admitted for another class -> cross-class candidate)."""
        out = []
        q = np.asarray(phi_retrieval, dtype=np.float32)
        for e in self.entries.values():
            if admitted_only and e.status != "admitted":
                continue
            s = cosine(q, np.asarray(e.phi_retrieval, dtype=np.float32))
            if s >= rho:
                out.append((e, s))
        out.sort(key=lambda t: -t[1])
        return out

    def nearest(self, phi_retrieval: np.ndarray, k: int = 3) -> List[tuple]:
        q = np.asarray(phi_retrieval, dtype=np.float32)
        scored = [(e, cosine(q, np.asarray(e.phi_retrieval, dtype=np.float32))) for e in self.entries.values()]
        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def stats(self) -> dict:
        st = {"total": len(self.entries), "admitted": 0, "rejected": 0, "proposed": 0}
        for e in self.entries.values():
            st[e.status] = st.get(e.status, 0) + 1
        return st
