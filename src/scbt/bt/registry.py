"""Node registry and XML <-> py_trees conversion.

XML dialect (BehaviorTree.CPP-like so the paper's patch grammar maps 1:1):

    <Sequence name="mission">
      <Takeoff name="takeoff" timeout_s="20"/>
      <Fallback name="go_or_wait">
        <GoToNED name="wp1" north="30" east="0" down="-8"/>
        <WaitSimTime duration_s="5"/>
      </Fallback>
    </Sequence>

Control nodes: Sequence, ReactiveSequence, Fallback, ReactiveFallback,
Parallel (policy="all"|"one"). Decorators: Inverter, ForceSuccess,
ForceFailure, Retry (num_attempts), Repeat (num_cycles).
Leaves: every class in ``scbt.bt.leaves`` listed in ``LEAF_REGISTRY``.

Every node gets a stable ``name`` attribute; if absent one is generated
(``<Tag>_<index>``) so patches can anchor on it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Iterable, Optional, Type

import py_trees
from py_trees.common import ParallelPolicy

from . import leaves as L

LEAF_REGISTRY: Dict[str, Type[L.VehicleLeaf]] = {
    "Takeoff": L.Takeoff,
    "Land": L.Land,
    "ReturnToLaunch": L.ReturnToLaunch,
    "GoToNED": L.GoToNED,
    "GoToRequiredGoal": L.GoToRequiredGoal,
    "MoveRelative": L.MoveRelative,
    "Hover": L.Hover,
    "WaitSimTime": L.WaitSimTime,
    "SetSpeed": L.SetSpeed,
    "InspectRegion": L.InspectRegion,
    "RequestScout": L.RequestScout,
    "IsAirborne": L.IsAirborne,
    "IsBatteryOk": L.IsBatteryOk,
    "IsWindOk": L.IsWindOk,
    "IsGpsOk": L.IsGpsOk,
    "IsLinkOk": L.IsLinkOk,
    "IsInsideGeofence": L.IsInsideGeofence,
}

CONTROL_TAGS = {"Sequence", "ReactiveSequence", "Fallback", "ReactiveFallback", "Parallel"}
DECORATOR_TAGS = {"Inverter", "ForceSuccess", "ForceFailure", "Retry", "Repeat", "MissionCritical"}


def make_leaf(tag: str, name: str, **ports) -> L.VehicleLeaf:
    cls = LEAF_REGISTRY[tag]
    return cls(name=name, **ports)


class _Counter:
    def __init__(self) -> None:
        self.n = 0

    def next(self, tag: str) -> str:
        self.n += 1
        return f"{tag}_{self.n}"


def from_xml(elem: ET.Element, counter: Optional[_Counter] = None) -> py_trees.behaviour.Behaviour:
    counter = counter or _Counter()
    tag = elem.tag
    attrs = dict(elem.attrib)
    name = attrs.pop("name", None) or counter.next(tag)
    children = list(elem)

    if tag in CONTROL_TAGS:
        kids = [from_xml(c, counter) for c in children]
        if tag == "Sequence":
            node = py_trees.composites.Sequence(name=name, memory=True, children=kids)
        elif tag == "ReactiveSequence":
            node = py_trees.composites.Sequence(name=name, memory=False, children=kids)
        elif tag == "Fallback":
            node = py_trees.composites.Selector(name=name, memory=True, children=kids)
        elif tag == "ReactiveFallback":
            node = py_trees.composites.Selector(name=name, memory=False, children=kids)
        else:  # Parallel
            policy = attrs.get("policy", "all")
            pol = ParallelPolicy.SuccessOnAll(synchronise=False) if policy == "all" else ParallelPolicy.SuccessOnOne()
            node = py_trees.composites.Parallel(name=name, policy=pol, children=kids)
        node.xml_tag = tag  # type: ignore[attr-defined]
        node.xml_attrs = attrs  # type: ignore[attr-defined]
        return node

    if tag in DECORATOR_TAGS:
        if len(children) != 1:
            raise ValueError(f"{tag} '{name}' must have exactly one child")
        child = from_xml(children[0], counter)
        if tag == "MissionCritical":
            from ..observer.interceptor import MissionCritical
            node = MissionCritical(name=name, child=child, patching_enabled=attrs.get("patching", "true") == "true")
        elif tag == "Inverter":
            node = py_trees.decorators.Inverter(name=name, child=child)
        elif tag == "ForceSuccess":
            node = py_trees.decorators.FailureIsSuccess(name=name, child=child)
        elif tag == "ForceFailure":
            node = py_trees.decorators.SuccessIsFailure(name=name, child=child)
        elif tag == "Retry":
            node = py_trees.decorators.Retry(name=name, child=child, num_failures=int(attrs.get("num_attempts", 3)))
        else:  # Repeat
            node = py_trees.decorators.Repeat(name=name, child=child, num_success=int(attrs.get("num_cycles", 1)))
        node.xml_tag = tag  # type: ignore[attr-defined]
        node.xml_attrs = attrs  # type: ignore[attr-defined]
        return node

    if tag in LEAF_REGISTRY:
        if children:
            raise ValueError(f"leaf {tag} '{name}' cannot have children")
        node = make_leaf(tag, name, **attrs)
        node.xml_tag = tag  # type: ignore[attr-defined]
        return node

    raise KeyError(f"unknown node tag '{tag}'")


def tree_from_xml_string(xml: str) -> py_trees.behaviour.Behaviour:
    return from_xml(ET.fromstring(xml))


def to_xml(node: py_trees.behaviour.Behaviour) -> ET.Element:
    """Inverse of ``from_xml`` (port values are emitted, so tuned parameters
    are preserved; this is what crystallization writes out)."""
    tag = getattr(node, "xml_tag", None) or type(node).__name__
    elem = ET.Element(tag, {"name": node.name})
    if isinstance(node, L.VehicleLeaf):
        for k, v in node.port_values().items():
            elem.set(k, _fmt(v))
    else:
        for k, v in getattr(node, "xml_attrs", {}).items():
            elem.set(k, str(v))
        for child in node.children:
            elem.append(to_xml(child))
    return elem


def to_xml_string(node: py_trees.behaviour.Behaviour) -> str:
    elem = to_xml(node)
    _indent(elem)
    return ET.tostring(elem, encoding="unicode")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore[possibly-undefined]
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def iter_nodes(root: py_trees.behaviour.Behaviour) -> Iterable[py_trees.behaviour.Behaviour]:
    yield root
    for c in root.children:
        yield from iter_nodes(c)


def find_node(root: py_trees.behaviour.Behaviour, name: str) -> Optional[py_trees.behaviour.Behaviour]:
    for n in iter_nodes(root):
        if n.name == name:
            return n
    return None
