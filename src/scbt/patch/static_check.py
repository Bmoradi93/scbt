"""Admission stage 1: static check (paper Sec. IV-F, item 1).

Checks, in order:
  1. operator in O; anchor exists in the tree and is not the root
  2. every leaf tag in the subtree is in the class manifest (or has a
     declared substitution, which is applied in place)
  3. every port name exists on that leaf; numeric ports parse; tunable
     placeholders are declared; declared tunables are used; bounds are sane
  4. control/decorator arity is valid; subtree depth <= max_depth
  5. ReplaceSubtree requires a rationale and may not delete MissionCritical
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional

import py_trees

from ..bt.registry import CONTROL_TAGS, DECORATOR_TAGS, LEAF_REGISTRY, find_node
from ..sim.config import ClassSpec
from .grammar import OPERATORS, Patch, _ANCHOR_REF, _PLACEHOLDER


class StaticCheckError(ValueError):
    pass


def static_check(patch: Patch, tree_root: py_trees.behaviour.Behaviour, cls: ClassSpec, max_depth: int = 6) -> List[str]:
    """Raise StaticCheckError on violation; return a list of warnings."""
    warnings: List[str] = []

    if patch.operator not in OPERATORS:
        raise StaticCheckError(f"operator '{patch.operator}' not in grammar")
    anchor = find_node(tree_root, patch.anchor)
    if anchor is None:
        raise StaticCheckError(f"anchor '{patch.anchor}' not found")
    if anchor is tree_root:
        raise StaticCheckError("anchor may not be the root")
    if anchor.parent is None:
        raise StaticCheckError("anchor has no parent")
    if patch.operator == "ReplaceSubtree":
        if not patch.rationale:
            raise StaticCheckError("ReplaceSubtree requires a rationale")
        if type(anchor).__name__ == "MissionCritical":
            raise StaticCheckError("ReplaceSubtree may not remove MissionCritical")
        warnings.append("ReplaceSubtree deletes existing logic (penalized)")

    if patch.operator == "WrapRetry":
        if patch.num_attempts < 1 or patch.num_attempts > 10:
            raise StaticCheckError("WrapRetry num_attempts must be in [1,10]")
        return warnings

    assert patch.subtree is not None
    anchor_ports = getattr(anchor, "port_values", lambda: {})()
    for ref in patch.anchor_refs():
        if ref not in anchor_ports:
            raise StaticCheckError(f"anchor '{patch.anchor}' has no port '{ref}' for reference '{{@{ref}}}'")
    declared = {p.name for p in patch.tunable}
    for p in patch.tunable:
        if not (p.min < p.max) or not (p.min <= p.initial <= p.max):
            raise StaticCheckError(f"tunable '{p.name}' has invalid bounds/initial")
    used = set()
    _check_elem(patch.subtree, cls, declared, used, depth=1, max_depth=max_depth, warnings=warnings)
    unused = declared - used
    if unused:
        raise StaticCheckError(f"declared tunables never used: {sorted(unused)}")
    return warnings


def _check_elem(el: ET.Element, cls: ClassSpec, declared: set, used: set, depth: int, max_depth: int, warnings: List[str]) -> None:
    if depth > max_depth:
        raise StaticCheckError(f"subtree depth exceeds {max_depth}")
    tag = el.tag
    kids = list(el)
    if tag in CONTROL_TAGS:
        if not kids:
            raise StaticCheckError(f"{tag} has no children")
        for k in kids:
            _check_elem(k, cls, declared, used, depth + 1, max_depth, warnings)
        return
    if tag in DECORATOR_TAGS:
        if tag == "MissionCritical":
            raise StaticCheckError("patches may not introduce MissionCritical")
        if len(kids) != 1:
            raise StaticCheckError(f"{tag} must have exactly one child")
        _check_elem(kids[0], cls, declared, used, depth + 1, max_depth, warnings)
        return
    if tag not in LEAF_REGISTRY:
        raise StaticCheckError(f"unknown node '{tag}'")
    if kids:
        raise StaticCheckError(f"leaf {tag} cannot have children")

    # Manifest / substitution
    if tag not in cls.leaves:
        sub = cls.substitutions.get(tag)
        if sub is None or sub not in cls.leaves:
            raise StaticCheckError(f"leaf '{tag}' not in manifest of class '{cls.name}' and no substitution")
        warnings.append(f"substituted {tag} -> {sub} for class {cls.name}")
        el.tag = sub
        tag = sub

    # Ports
    leaf_cls = LEAF_REGISTRY[tag]
    for k, v in el.attrib.items():
        if k == "name":
            continue
        if k not in leaf_cls.PORTS:
            raise StaticCheckError(f"{tag}: unknown port '{k}'")
        ptype, _default, bounds = leaf_cls.PORTS[k]
        if _ANCHOR_REF.match(v):
            continue  # validated against the anchor above
        m = _PLACEHOLDER.match(v)
        if m:
            if m.group(1) not in declared:
                raise StaticCheckError(f"{tag}.{k}: placeholder '{v}' not declared as tunable")
            if ptype is not float:
                raise StaticCheckError(f"{tag}.{k}: only numeric ports may be tunable")
            used.add(m.group(1))
            continue
        try:
            val = ptype(v)
        except (TypeError, ValueError) as exc:
            raise StaticCheckError(f"{tag}.{k}: cannot parse '{v}' as {ptype.__name__}") from exc
        if bounds is not None and not (bounds[0] <= float(val) <= bounds[1]):
            raise StaticCheckError(f"{tag}.{k}={val} outside allowed range {bounds}")
