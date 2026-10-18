"""Patch grammar O (paper Def. 2).

    <patch operator="WrapFallback" anchor="wp1">
      <subtree>
        <Sequence>
          <WaitSimTime duration_s="{wait_s}"/>
          <GoToNED north="40" east="0" down="-8" velocity="{v}"/>
        </Sequence>
      </subtree>
      <tunable_ports>
        <port name="wait_s" min="2" max="30" initial="8"/>
        <port name="v"      min="0.5" max="6" initial="2"/>
      </tunable_ports>
      <rationale>Wait out the gust, then retry at reduced speed.</rationale>
    </patch>

Operators:
  WrapFallback(a, s)   a -> Fallback(a, s)         recovery after a fails
  GuardWith(a, s)      a -> Sequence(s, a)         establish precondition first
  WrapRetry(a, N)      a -> Retry(N)(a)
  InsertBefore(a, s)   parent(a): [.., s, a, ..]
  InsertAfter(a, s)    parent(a): [.., a, s, ..]
  ReplaceSubtree(a, s) a -> s                      only with rationale; penalized

Tunable ports are referenced inside the subtree as ``{name}`` placeholders;
``Patch.instantiate(theta)`` substitutes values. Placeholders without a
``<port>`` declaration are a static-check error.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

OPERATORS = ("WrapFallback", "GuardWith", "WrapRetry", "InsertBefore", "InsertAfter", "ReplaceSubtree")
_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")
# {@port}: value of the anchor leaf's port, resolved when the patch is applied.
# Keeps patches mission-agnostic (no literal coordinates copied from the event).
_ANCHOR_REF = re.compile(r"^\{@([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass
class TunablePort:
    name: str
    min: float
    max: float
    initial: float

    def clip(self, v: float) -> float:
        return max(self.min, min(self.max, v))


@dataclass
class Patch:
    operator: str
    anchor: str
    subtree: Optional[ET.Element]            # None for WrapRetry
    tunable: List[TunablePort] = field(default_factory=list)
    rationale: str = ""
    num_attempts: int = 3                     # WrapRetry only
    patch_id: str = ""

    # --- parameters -------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.tunable)

    def theta0(self) -> List[float]:
        return [p.initial for p in self.tunable]

    def bounds(self) -> List[tuple]:
        return [(p.min, p.max) for p in self.tunable]

    def placeholders(self) -> List[str]:
        names: List[str] = []
        if self.subtree is None:
            return names
        for el in self.subtree.iter():
            for v in el.attrib.values():
                m = _PLACEHOLDER.match(v)
                if m and m.group(1) not in names:
                    names.append(m.group(1))
        return names

    def anchor_refs(self) -> List[str]:
        names: List[str] = []
        if self.subtree is None:
            return names
        for el in self.subtree.iter():
            for v in el.attrib.values():
                m = _ANCHOR_REF.match(v)
                if m and m.group(1) not in names:
                    names.append(m.group(1))
        return names

    def instantiate(self, theta: Optional[List[float]] = None,
                    anchor_ports: Optional[Dict[str, object]] = None) -> Optional[ET.Element]:
        """Return a deep copy of the subtree with tunable placeholders and
        anchor references substituted."""
        if self.subtree is None:
            return None
        theta = list(theta) if theta is not None else self.theta0()
        values = {p.name: p.clip(t) for p, t in zip(self.tunable, theta)}
        anchor_ports = anchor_ports or {}
        out = _deepcopy(self.subtree)
        for el in out.iter():
            for k, v in list(el.attrib.items()):
                m = _PLACEHOLDER.match(v)
                if m:
                    el.set(k, f"{values[m.group(1)]:.6g}")
                    continue
                a = _ANCHOR_REF.match(v)
                if a:
                    if a.group(1) not in anchor_ports:
                        raise KeyError(f"anchor has no port '{a.group(1)}' for reference '{v}'")
                    val = anchor_ports[a.group(1)]
                    el.set(k, f"{val:.6g}" if isinstance(val, float) else str(val))
        return out

    # --- serialization ----------------------------------------------------
    def to_xml_string(self) -> str:
        root = ET.Element("patch", {"operator": self.operator, "anchor": self.anchor})
        if self.operator == "WrapRetry":
            root.set("num_attempts", str(self.num_attempts))
        if self.subtree is not None:
            st = ET.SubElement(root, "subtree")
            st.append(_deepcopy(self.subtree))
        if self.tunable:
            tp = ET.SubElement(root, "tunable_ports")
            for p in self.tunable:
                ET.SubElement(tp, "port", {"name": p.name, "min": f"{p.min:g}", "max": f"{p.max:g}", "initial": f"{p.initial:g}"})
        r = ET.SubElement(root, "rationale")
        r.text = self.rationale
        return ET.tostring(root, encoding="unicode")


def parse_patch_xml(xml: str) -> Patch:
    root = ET.fromstring(xml)
    if root.tag != "patch":
        raise ValueError("root element must be <patch>")
    op = root.get("operator", "")
    if op not in OPERATORS:
        raise ValueError(f"unknown operator '{op}'")
    anchor = root.get("anchor", "")
    if not anchor:
        raise ValueError("patch requires anchor")
    st_el = root.find("subtree")
    subtree = None
    if st_el is not None:
        kids = list(st_el)
        if len(kids) != 1:
            raise ValueError("<subtree> must contain exactly one root node")
        subtree = kids[0]
    if op != "WrapRetry" and subtree is None:
        raise ValueError(f"{op} requires a <subtree>")
    tunable = []
    tp = root.find("tunable_ports")
    if tp is not None:
        for p in tp.findall("port"):
            tunable.append(TunablePort(p.get("name"), float(p.get("min")), float(p.get("max")), float(p.get("initial"))))
    rationale = (root.findtext("rationale") or "").strip()
    return Patch(
        operator=op, anchor=anchor, subtree=subtree, tunable=tunable, rationale=rationale,
        num_attempts=int(root.get("num_attempts", 3)),
    )


def _deepcopy(el: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(el))
