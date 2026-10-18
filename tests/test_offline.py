"""Offline tests: no simulator needed."""

import xml.etree.ElementTree as ET

import py_trees
import pytest
from py_trees.common import Status

from scbt.bt.base_trees import build_base_tree
from scbt.bt.leaves import NAMESPACE
from scbt.bt.registry import LEAF_REGISTRY, find_node, from_xml, iter_nodes, to_xml_string, tree_from_xml_string
from scbt.observer.interceptor import MissionCritical
from scbt.observer.trace import TickTrace, trace_hash
from scbt.patch.apply import apply_patch
from scbt.patch.grammar import parse_patch_xml
from scbt.patch.static_check import StaticCheckError, static_check
from scbt.sim.config import COMMON_LEAVES, DEFAULT_CLASSES, build_scene_config, default_fleet


class FakeVehicle:
    """Minimal stand-in: telemetry dict + spec; no commands issued."""

    def __init__(self, spec):
        self.spec = spec
        self.telemetry = {"t": 0.0, "alt": 5.0, "landed": False, "battery_pct": 80.0,
                          "wind_speed": 0.0, "gps_ok": True, "link_quality": 1.0, "pos": (0, 0, -5)}
        self.home_ned = (0, 0, 0)
        self.name = "Fake"

    @property
    def ned(self):
        return self.telemetry["pos"]

    def dist_to(self, n, e, d):
        return 0.0

    def cancel(self):
        pass


class FakeSession:
    geofence = {"n_min": -200, "n_max": 200, "e_min": -200, "e_max": 200, "alt_min": 0, "alt_max": 60}


def _bind(ns, vehicle):
    bb = py_trees.blackboard.Client(name="t", namespace=ns)
    for k in ("vehicle", "session", "speed", "failure_pending", "give_up"):
        bb.register_key(k, access=py_trees.common.Access.WRITE)
    bb.vehicle = vehicle
    bb.session = FakeSession()
    bb.speed = 3.0
    bb.failure_pending = False
    bb.give_up = False
    return bb


def test_manifest_matches_registry():
    for cls in DEFAULT_CLASSES.values():
        for leaf in cls.leaves:
            assert leaf in LEAF_REGISTRY, leaf
        for sub in cls.substitutions.values():
            assert sub in cls.leaves
    for leaf in COMMON_LEAVES:
        assert leaf in LEAF_REGISTRY


def test_scene_config_builds_for_fleet():
    cfg = build_scene_config(default_fleet(4))
    assert [a["name"] for a in cfg["actors"]] == ["Drone1", "Drone2", "Drone3", "Drone4"]
    for a in cfg["actors"]:
        ids = [s["id"] for s in a["robot-config"]["sensors"]]
        assert "Battery" in ids and "GPS" in ids
        assert not any(s["type"] == "camera" for s in a["robot-config"]["sensors"])
    assert cfg["actors"][2]["robot-config"]["links"][0]["inertial"]["mass"] == DEFAULT_CLASSES["carrier"].frame_mass_kg


def test_base_tree_roundtrip():
    tok = NAMESPACE.set("/t1")
    try:
        el = build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml
        root = from_xml(el)
        xml = to_xml_string(root)
        root2 = tree_from_xml_string(xml)
    finally:
        NAMESPACE.reset(tok)
    names = [n.name for n in iter_nodes(root)]
    assert names == [n.name for n in iter_nodes(root2)]
    assert isinstance(find_node(root, "mission"), MissionCritical)
    assert find_node(root, "wp1").port_values()["north"] == 40.0
    assert find_node(root, "wp1").port_values()["goal_id"] == "deliver"


def test_mission_plans_and_value():
    m1 = build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0))
    assert m1.required_goals == ["deliver"] and m1.value({"deliver"}) == 1.0 and m1.value(set()) == 0.0
    m2 = build_base_tree("M2", DEFAULT_CLASSES["scout"], (0, 0, 0))
    assert m2.required_goals == ["lane1"] and m2.optional_goals == ["lane2", "lane3"]
    assert m2.value({"lane1"}) == 0.5 and m2.value({"lane1", "lane2", "lane3"}) == 1.0


def test_patch_parse_and_static_check():
    tok = NAMESPACE.set("/t2")
    try:
        root = from_xml(build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml)
    finally:
        NAMESPACE.reset(tok)
    xml = """<patch operator="WrapFallback" anchor="wp1">
      <subtree><Sequence><WaitSimTime duration_s="{w}"/><GoToNED north="40" east="0" down="-8"/></Sequence></subtree>
      <tunable_ports><port name="w" min="1" max="30" initial="5"/></tunable_ports>
      <rationale>wait</rationale></patch>"""
    p = parse_patch_xml(xml)
    assert p.dim == 1 and p.placeholders() == ["w"]
    assert static_check(p, root, DEFAULT_CLASSES["scout"]) == []
    inst = p.instantiate([100.0])  # clipped to max
    assert inst.find("WaitSimTime").get("duration_s") == "30"

    # Unknown port
    bad = parse_patch_xml(xml.replace('duration_s="{w}"', 'durationz="{w}"'))
    with pytest.raises(StaticCheckError):
        static_check(bad, root, DEFAULT_CLASSES["scout"])
    # Undeclared placeholder
    bad = parse_patch_xml(xml.replace('<port name="w" min="1" max="30" initial="5"/>', ""))
    with pytest.raises(StaticCheckError):
        static_check(bad, root, DEFAULT_CLASSES["scout"])
    # Out-of-range literal
    bad = parse_patch_xml(xml.replace('down="-8"', 'down="-8" velocity="50"'))
    with pytest.raises(StaticCheckError):
        static_check(bad, root, DEFAULT_CLASSES["scout"])
    # Missing anchor
    bad = parse_patch_xml(xml.replace('anchor="wp1"', 'anchor="nope"'))
    with pytest.raises(StaticCheckError):
        static_check(bad, root, DEFAULT_CLASSES["scout"])


def test_substitution_for_carrier():
    tok = NAMESPACE.set("/t3")
    try:
        root = from_xml(build_base_tree("M1", DEFAULT_CLASSES["carrier"], (0, 0, 0)).xml)
    finally:
        NAMESPACE.reset(tok)
    xml = """<patch operator="GuardWith" anchor="wp1">
      <subtree><InspectRegion north="40" east="0" duration_s="3"/></subtree><rationale>look first</rationale></patch>"""
    p = parse_patch_xml(xml)
    w = static_check(p, root, DEFAULT_CLASSES["carrier"])
    assert any("substituted InspectRegion -> RequestScout" in x for x in w)
    assert p.subtree.tag == "RequestScout"
    # Scout executes InspectRegion natively: no substitution warning
    p_scout = parse_patch_xml(xml)
    assert static_check(p_scout, root, DEFAULT_CLASSES["scout"]) == []
    # Unknown node type is rejected
    nosub = """<patch operator="GuardWith" anchor="wp1"><subtree><Nope/></subtree></patch>"""
    with pytest.raises(StaticCheckError):
        static_check(parse_patch_xml(nosub), root, DEFAULT_CLASSES["relay"])


def test_apply_operators_on_live_tree():
    tok = NAMESPACE.set("/t4")
    try:
        root = from_xml(build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml)
        wp1 = find_node(root, "wp1")
        parent = wp1.parent
        idx = parent.children.index(wp1)

        p = parse_patch_xml("""<patch operator="WrapFallback" anchor="wp1">
            <subtree><WaitSimTime duration_s="5"/></subtree></patch>""")
        p.patch_id = "p1"
        new = apply_patch(root, p)
        assert parent.children[idx] is new and isinstance(new, py_trees.composites.Selector)
        assert new.children[0] is wp1 and wp1.parent is new
        assert "WaitSimTime" in to_xml_string(root)

        p2 = parse_patch_xml("""<patch operator="WrapRetry" anchor="wp2" num_attempts="4"/>""")
        p2.patch_id = "p2"
        new2 = apply_patch(root, p2)
        assert isinstance(new2, py_trees.decorators.Retry) and new2.decorated.name == "wp2"
        assert find_node(root, "waypoints").children[1] is new2

        p3 = parse_patch_xml("""<patch operator="InsertBefore" anchor="land">
            <subtree><Hover duration_s="2"/></subtree></patch>""")
        p3.patch_id = "p3"
        apply_patch(root, p3)
        seq = find_node(root, "mission_seq")
        assert [getattr(c, "xml_tag", "") for c in seq.children][-2:] == ["Hover", "Land"]

        # Name collision resolution: second patch using the same leaf names
        p4 = parse_patch_xml("""<patch operator="InsertAfter" anchor="land">
            <subtree><Hover name="Hover_1" duration_s="2"/></subtree></patch>""")
        p4.patch_id = "p4"
        apply_patch(root, p4)
        names = [n.name for n in iter_nodes(root)]
        assert len(names) == len(set(names))

        # Round-trip still parses
        tree_from_xml_string(to_xml_string(root))
    finally:
        NAMESPACE.reset(tok)


def test_mission_critical_intercepts_and_gives_up():
    ns = "/t5"
    tok = NAMESPACE.set(ns)
    try:
        xml = """<Fallback name="root">
          <MissionCritical name="mission">
            <Sequence name="s"><IsBatteryOk name="guard" min_pct="25"/></Sequence>
          </MissionCritical>
          <IsAirborne name="abort_marker"/>
        </Fallback>"""
        root = tree_from_xml_string(xml)
    finally:
        NAMESPACE.reset(tok)
    v = FakeVehicle(DEFAULT_CLASSES["scout"])
    bb = _bind(ns, v)
    tree = py_trees.trees.BehaviourTree(root)
    tr = TickTrace()
    tree.add_visitor(tr)
    tree.setup()

    tr.tick = 1; tree.tick()
    assert root.status == Status.SUCCESS and not bb.failure_pending

    v.telemetry["battery_pct"] = 10.0
    tr.tick = 2; tree.tick()
    assert bb.failure_pending is True
    assert tr.culprit_leaf().name == "guard"
    assert root.status == Status.RUNNING          # abort branch NOT started
    bb.failure_pending = False

    bb.give_up = True
    tr.tick = 3; tree.tick()
    assert find_node(root, "mission").status == Status.FAILURE
    assert find_node(root, "abort_marker").status == Status.SUCCESS   # abort branch ran
    h1 = trace_hash(tr.records)
    assert len(h1) == 64


def test_frozen_mission_critical_passes_failure_through():
    ns = "/t6"
    tok = NAMESPACE.set(ns)
    try:
        root = tree_from_xml_string("""<MissionCritical name="m" patching="false">
            <IsBatteryOk name="g" min_pct="50"/></MissionCritical>""")
    finally:
        NAMESPACE.reset(tok)
    v = FakeVehicle(DEFAULT_CLASSES["scout"])
    v.telemetry["battery_pct"] = 10.0
    bb = _bind(ns, v)
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup()
    tree.tick()
    assert root.status == Status.FAILURE and bb.failure_pending is False


def test_anchor_port_references_and_required_goal_leaf():
    from scbt.patch.apply import apply_patch
    from scbt.patch.grammar import parse_patch_xml
    from scbt.patch.static_check import StaticCheckError, static_check
    tok = NAMESPACE.set("/t7")
    try:
        root = from_xml(build_base_tree("M2", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml)
        p = parse_patch_xml("""<patch operator="WrapFallback" anchor="wp2">
          <subtree><Sequence><WaitSimTime duration_s="3"/>
            <GoToNED north="{@north}" east="{@east}" down="{@down}" goal_id="{@goal_id}" velocity="2"/></Sequence></subtree>
          <rationale>retry same leg</rationale></patch>""")
        assert p.anchor_refs() == ["north", "east", "down", "goal_id"]
        assert static_check(p, root, DEFAULT_CLASSES["scout"]) == []
        p.patch_id = "pa"
        new = apply_patch(root, p)
        wp2 = find_node(root, "wp2").port_values()
        retry = [n for n in iter_nodes(new) if getattr(n, "xml_tag", "") == "GoToNED" and n.name != "wp2"][0]
        assert retry.port_values()["north"] == wp2["north"] and retry.port_values()["goal_id"] == "lane1"
        # Reference to a port the anchor does not have is rejected
        bad = parse_patch_xml("""<patch operator="GuardWith" anchor="wp1">
          <subtree><WaitSimTime duration_s="{@nope}"/></subtree></patch>""")
        with pytest.raises(StaticCheckError):
            static_check(bad, root, DEFAULT_CLASSES["scout"])
        # GoToRequiredGoal is in the manifest and has no coordinate ports
        g = parse_patch_xml("""<patch operator="WrapFallback" anchor="guarded_flight">
          <subtree><GoToRequiredGoal velocity="3"/></subtree></patch>""")
        assert static_check(g, root, DEFAULT_CLASSES["carrier"]) == []
    finally:
        NAMESPACE.reset(tok)


def test_patch_anchored_on_node_from_earlier_patch_roundtrips():
    """Admission replays the live tree (base + earlier patches); a candidate
    may anchor on a node an earlier patch introduced (and that was renamed
    by the uniquifier)."""
    from scbt.patch.apply import apply_patch
    from scbt.patch.grammar import parse_patch_xml
    from scbt.bt.registry import to_xml
    tok = NAMESPACE.set("/t8")
    try:
        root = from_xml(build_base_tree("M2", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml)
        p1 = parse_patch_xml("""<patch operator="WrapFallback" anchor="guarded_flight">
          <subtree><ReactiveSequence name="guarded_flight"><IsBatteryOk min_pct="12"/>
            <GoToRequiredGoal velocity="3"/></ReactiveSequence></subtree></patch>""")
        p1.patch_id = "p0001"
        apply_patch(root, p1)
        assert find_node(root, "p0001_guarded_flight") is not None     # renamed copy
        live = to_xml(root)
        root2 = from_xml(live)                                            # what admission compiles from
        p2 = parse_patch_xml("""<patch operator="GuardWith" anchor="p0001_guarded_flight">
          <subtree><WaitSimTime duration_s="2"/></subtree></patch>""")
        p2.patch_id = "p0002"
        apply_patch(root2, p2)
        assert find_node(root2, "p0002_guard") is not None
        assert isinstance(find_node(root2, "mission"), MissionCritical)
    finally:
        NAMESPACE.reset(tok)


def test_expert_tree_builds_and_parses():
    from scbt.bt.expert_tree import expert_tree
    for m in ("M1", "M2"):
        plan = expert_tree(m, DEFAULT_CLASSES["scout"], (0, 0, 0))
        tok = NAMESPACE.set(f"/b0/{m}")
        try:
            root = from_xml(plan.xml)
        finally:
            NAMESPACE.reset(tok)
        names = [n.name for n in iter_nodes(root)]
        assert len(names) == len(set(names))
        assert find_node(root, "finish_required") is not None and find_node(root, "wp1_windfb") is not None
        mc = find_node(root, "mission")
        assert isinstance(mc, MissionCritical) and mc.patching_enabled is False


def test_crystallize_orders_same_anchor_by_score_and_skips_missing_anchors(tmp_path):
    from scbt.crystallize import crystallize
    from scbt.manager.library import LibraryEntry, PatchLibrary
    lib = PatchLibrary(str(tmp_path / "l.json"))
    def entry(pid, xml, uses, succ, created):
        return LibraryEntry(patch_id=pid, patch_xml=xml, phi=[0.0] * 4, phi_retrieval=[0.0] * 4, origin_class="scout",
                            theta={"scout": [7.0]}, admitted_for=["scout"], status="admitted", uses=uses, successes=succ,
                            created=created)
    wrap = lambda pid, dur: (f'<patch operator="WrapFallback" anchor="wp1"><subtree><WaitSimTime name="{pid}_w" duration_s="{{d}}"/>'
                             f'</subtree><tunable_ports><port name="d" min="1" max="30" initial="{dur}"/></tunable_ports><rationale>r</rationale></patch>')
    lib.add(entry("pA", wrap("pA", 3), uses=4, succ=1, created=1.0))      # score 0.25
    lib.add(entry("pB", wrap("pB", 5), uses=4, succ=4, created=2.0))      # score 1.0 -> tried first
    lib.add(entry("pC", '<patch operator="WrapRetry" anchor="nope" num_attempts="2"><rationale>r</rationale></patch>', 1, 1, 3.0))
    el, info = crystallize(lib, "scout", "M1", bandit=None)
    assert [a["patch_id"] for a in info["applied"]] == ["pA", "pB"]   # ascending score; pB ends innermost
    assert info["skipped"][0]["patch_id"] == "pC"
    assert el.find(".//MissionCritical").get("patching") == "false"
    # Nesting: Fallback(Fallback(wp1, pB), pA): pB is tried before pA
    outer = el.find(".//Fallback[@name='pA_fallback']")
    inner = outer.find("./Fallback[@name='pB_fallback']")
    assert inner is not None and inner[0].get("name") == "wp1"
    assert inner[1].get("name") == "pB_w" and outer[1].get("name") == "pA_w"
    assert inner[1].get("duration_s") == "7"   # tuned theta baked in
    # Round-trips through the registry
    tok = NAMESPACE.set("/t9")
    try:
        root = from_xml(el)
        assert find_node(root, "mission").patching_enabled is False
    finally:
        NAMESPACE.reset(tok)


def test_wrap_anchor_that_is_direct_child_of_decorator():
    from scbt.patch.apply import apply_patch
    from scbt.patch.grammar import parse_patch_xml
    tok = NAMESPACE.set("/t10")
    try:
        root = from_xml(build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml)
        p = parse_patch_xml("""<patch operator="WrapFallback" anchor="mission_seq">
          <subtree><ReturnToLaunch velocity="3"/></subtree></patch>""")
        p.patch_id = "pd"
        new = apply_patch(root, p)
        mc = find_node(root, "mission")
        assert mc.decorated is new and new.parent is mc and find_node(root, "mission_seq").parent is new
    finally:
        NAMESPACE.reset(tok)
