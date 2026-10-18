"""Apply a patch to a live py_trees tree (hot-patching, no restart).

The anchor's parent composite (or decorator) has the anchor replaced by
the new structure. Names of generated wrapper nodes are derived from the
patch id so that a crystallized tree is readable and anchors stay unique.
"""

from __future__ import annotations

from typing import List, Optional

import py_trees

from ..bt.registry import find_node, from_xml, iter_nodes
from .grammar import Patch, parse_patch_xml


def _unique_names(root: py_trees.behaviour.Behaviour, new: py_trees.behaviour.Behaviour, prefix: str) -> None:
    existing = {n.name for n in iter_nodes(root)}
    for n in iter_nodes(new):
        if n.name in existing:
            base = f"{prefix}_{n.name}"
            cand, i = base, 1
            while cand in existing:
                i += 1
                cand = f"{base}_{i}"
            n.name = cand
        existing.add(n.name)


def _replace(parent: py_trees.behaviour.Behaviour, old: py_trees.behaviour.Behaviour, new: py_trees.behaviour.Behaviour) -> None:
    if isinstance(parent, py_trees.composites.Composite):
        parent.replace_child(old, new)
    elif isinstance(parent, py_trees.decorators.Decorator):
        parent.decorated = new
        parent.children = [new]
        new.parent = parent
        old.parent = None
    else:
        raise TypeError(f"cannot replace child of {type(parent).__name__}")


def apply_patch(root: py_trees.behaviour.Behaviour, patch: Patch, theta: Optional[List[float]] = None,
                setup_fn=None) -> py_trees.behaviour.Behaviour:
    """Mutate ``root`` in place; return the node that now occupies the
    anchor's slot (the wrapper for Wrap*/Guard*, the subtree for Replace,
    or the anchor itself for Insert*). ``setup_fn(node)`` is called on
    newly created nodes (used to run ``setup()`` with the tree's blackboard)."""
    anchor = find_node(root, patch.anchor)
    if anchor is None:
        raise KeyError(f"anchor '{patch.anchor}' not found")
    parent = anchor.parent
    if parent is None:
        raise ValueError("anchor has no parent")
    pid = patch.patch_id or "patch"

    anchor_ports = getattr(anchor, "port_values", lambda: {})()
    sub_el = patch.instantiate(theta, anchor_ports)
    sub = from_xml(sub_el) if sub_el is not None else None
    if sub is not None:
        _unique_names(root, sub, pid)

    # Preempt the anchor if it is mid-flight; the restart is explicit.
    if anchor.status == py_trees.common.Status.RUNNING:
        anchor.stop(py_trees.common.Status.INVALID)

    op = patch.operator
    if op == "WrapFallback":
        wrapper = py_trees.composites.Selector(name=f"{pid}_fallback", memory=True)
        wrapper.xml_tag = "Fallback"; wrapper.xml_attrs = {}  # type: ignore[attr-defined]
        _replace(parent, anchor, wrapper)
        wrapper.add_children([anchor, sub])
        new = wrapper
    elif op == "GuardWith":
        wrapper = py_trees.composites.Sequence(name=f"{pid}_guard", memory=True)
        wrapper.xml_tag = "Sequence"; wrapper.xml_attrs = {}  # type: ignore[attr-defined]
        _replace(parent, anchor, wrapper)
        wrapper.add_children([sub, anchor])
        new = wrapper
    elif op == "WrapRetry":
        wrapper = py_trees.decorators.Retry(name=f"{pid}_retry", child=anchor, num_failures=patch.num_attempts)
        wrapper.xml_tag = "Retry"; wrapper.xml_attrs = {"num_attempts": str(patch.num_attempts)}  # type: ignore[attr-defined]
        # Retry.__init__ set anchor.parent = wrapper; fix parent's child slot.
        _replace(parent, anchor, wrapper)
        anchor.parent = wrapper
        new = wrapper
    elif op in ("InsertBefore", "InsertAfter"):
        if not isinstance(parent, py_trees.composites.Composite):
            raise ValueError(f"{op} requires the anchor's parent to be a composite")
        idx = parent.children.index(anchor) + (1 if op == "InsertAfter" else 0)
        parent.insert_child(sub, idx)
        new = anchor
    elif op == "ReplaceSubtree":
        _replace(parent, anchor, sub)
        new = sub
    else:
        raise ValueError(op)

    if sub is not None and setup_fn is not None:
        for n in iter_nodes(sub):
            setup_fn(n)
    _assert_tree(root)
    return new


def _assert_tree(root: py_trees.behaviour.Behaviour) -> None:
    """Structural invariant: every node reachable once, parent links consistent."""
    seen = set()
    stack = [(root, None)]
    while stack:
        n, parent = stack.pop()
        if id(n) in seen:
            raise RuntimeError(f"tree invariant violated: node '{n.name}' reachable twice")
        seen.add(id(n))
        if n.parent is not parent:
            raise RuntimeError(f"tree invariant violated: '{n.name}'.parent is "
                               f"'{getattr(n.parent, 'name', None)}', expected '{getattr(parent, 'name', None)}'")
        for c in n.children:
            stack.append((c, n))


def apply_patch_xml(root: py_trees.behaviour.Behaviour, xml: str, theta: Optional[List[float]] = None, setup_fn=None):
    return apply_patch(root, parse_patch_xml(xml), theta, setup_fn)
