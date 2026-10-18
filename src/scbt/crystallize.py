"""Crystallization (paper Sec. IV-G): compile the library into frozen trees.

    python -m scbt.crystallize --library runs/lib.json --out-dir runs/frozen \\
        [--state runs/lib.state.json] [--classes scout carrier relay] [--missions M1 M2]

For each (class, mission): start from the base tree T_m, apply every
admitted patch adopted by the class in *priority order*, with the class's
tuned theta baked in, and disable patching on MissionCritical. Output:
``T_star_<class>_<mission>.xml`` plus ``crystallize_<class>_<mission>.json``
listing the patches applied, skipped (anchor absent in this mission) and
their scores.

Priority order: patches are grouped by anchor; within a group they are
scored by the bandit's posterior-mean reward for the patch's own
originating context (falls back to empirical success rate). Because a
later WrapFallback on the same anchor wraps the *inner* anchor, the last
applied patch is tried first; patches are therefore applied in ascending
score so the result is Fallback(Fallback(a, best), second): the best
recovery is tried first. Across groups, admission order is kept so that patches anchored on
nodes introduced by earlier patches find their anchors.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np

from .bt.base_trees import build_base_tree
from .bt.leaves import NAMESPACE
from .bt.registry import find_node, from_xml, iter_nodes, to_xml_string
from .manager.bandit import LinearTS
from .manager.library import LibraryEntry, PatchLibrary
from .patch.apply import apply_patch
from .sim.config import DEFAULT_CLASSES, default_fleet


def score_entry(e: LibraryEntry, bandit: Optional[LinearTS]) -> float:
    if bandit is not None and e.patch_id in bandit.arms:
        return bandit.posterior_mean_reward(e.patch_id, np.asarray(e.phi, dtype=np.float32))
    return e.successes / e.uses if e.uses else 0.0


def ordered_patches(lib: PatchLibrary, cls_name: str, bandit: Optional[LinearTS]) -> List[tuple]:
    admitted = [e for e in lib.entries.values() if e.status == "admitted" and cls_name in e.admitted_for]
    admitted.sort(key=lambda e: e.created)                       # admission order
    groups: Dict[str, List[LibraryEntry]] = {}
    order: List[str] = []
    for e in admitted:
        a = e.patch().anchor
        if a not in groups:
            groups[a] = []; order.append(a)
        groups[a].append(e)
    out = []
    for a in order:
        g = sorted(groups[a], key=lambda e: score_entry(e, bandit))   # ascending: best applied last = innermost
        out += [(e, score_entry(e, bandit)) for e in g]
    return out


def crystallize(lib: PatchLibrary, cls_name: str, mission: str, bandit: Optional[LinearTS],
                home=(0.0, 0.0, 0.0)) -> tuple:
    cls = DEFAULT_CLASSES[cls_name]
    plan = build_base_tree(mission, cls, home)
    tok = NAMESPACE.set(f"/crystal/{cls_name}/{mission}")
    applied, skipped = [], []
    try:
        root = from_xml(plan.xml)
        for e, sc in ordered_patches(lib, cls_name, bandit):
            patch = e.patch()
            if find_node(root, patch.anchor) is None:
                skipped.append({"patch_id": e.patch_id, "anchor": patch.anchor, "reason": "anchor absent"})
                continue
            theta = e.theta.get(cls_name) or e.theta.get(e.origin_class) or patch.theta0()
            try:
                apply_patch(root, patch, theta)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"patch_id": e.patch_id, "anchor": patch.anchor, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            applied.append({"patch_id": e.patch_id, "anchor": patch.anchor, "operator": patch.operator,
                            "theta": [round(float(t), 4) for t in theta], "score": round(float(sc), 4),
                            "uses": e.uses, "successes": e.successes})
        xml = to_xml_string(root)
    finally:
        NAMESPACE.reset(tok)
    el = ET.fromstring(xml)
    for mc in el.iter("MissionCritical"):
        mc.set("patching", "false")
    n_nodes = sum(1 for _ in el.iter())
    depth = _depth(el)
    return el, {"class": cls_name, "mission": mission, "applied": applied, "skipped": skipped,
                "nodes": n_nodes, "depth": depth, "base_nodes": sum(1 for _ in plan.xml.iter())}


def _depth(el: ET.Element) -> int:
    return 1 + max((_depth(c) for c in el), default=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes", nargs="+", default=["scout"])
    ap.add_argument("--missions", nargs="+", default=["M1", "M2"])
    args = ap.parse_args(argv)

    lib = PatchLibrary(args.library)
    state = args.state or args.library.replace(".json", "") + ".state.json"
    bandit = None
    if os.path.exists(state):
        with open(state) as f:
            bandit = LinearTS.from_dict(json.load(f)["bandit"])
    os.makedirs(args.out_dir, exist_ok=True)
    for c in args.classes:
        for m in args.missions:
            el, info = crystallize(lib, c, m, bandit)
            tree_path = os.path.join(args.out_dir, f"T_star_{c}_{m}.xml")
            ET.ElementTree(el).write(tree_path, encoding="unicode")
            with open(os.path.join(args.out_dir, f"crystallize_{c}_{m}.json"), "w") as f:
                json.dump(info, f, indent=1)
            print(f"{c}/{m}: {len(info['applied'])} patches applied, {len(info['skipped'])} skipped, "
                  f"{info['nodes']} nodes (base {info['base_nodes']}), depth {info['depth']} -> {tree_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
