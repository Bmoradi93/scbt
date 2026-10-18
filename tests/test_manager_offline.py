"""Offline tests for the Patch Manager (no simulator, no API key)."""

import numpy as np
import pytest

from scbt.manager.bandit import CALL_LLM, LinearTS
from scbt.manager.embedding import ContextEmbedder, HashTextEmbedder, cosine
from scbt.manager.library import LibraryEntry, PatchLibrary
from scbt.manager.schemas import PatchProposal
from scbt.observer.events import FailureEvent
from scbt.patch.static_check import static_check
from scbt.sim.config import DEFAULT_CLASSES


def _event(tag="GoToNED", node="wp1", wind=22.0, batt=80.0, cls="scout"):
    return FailureEvent(vehicle="Drone1", vehicle_class=cls, failed_node=node, failed_tag=tag,
                        failed_ports={"north": 40.0, "east": 0.0, "down": -8.0, "arrive_tol_m": 1.5},
                        feedback="arrived 2.6 m from target", anchor_path=["root", "mission", node], tick=100, t=17.0,
                        blackboard={"alt": 8.0, "speed": 1.0, "wind_speed": wind, "battery_pct": batt,
                                    "gps_ok": True, "link_quality": 1.0}, tick_trace=[])


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashTextEmbedder(128)
    a, b = e("wind gust 22 m/s"), e("wind gust 22 m/s")
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert cosine(e("wind gust"), e("wind gust strong")) > cosine(e("wind gust"), e("battery low"))


def test_context_embedding_similarity_structure():
    emb = ContextEmbedder(HashTextEmbedder(128))
    wind_a = emb.retrieval_part(emb(_event(wind=22.0)))
    wind_b = emb.retrieval_part(emb(_event(wind=25.0, batt=70.0)))
    batt = emb.retrieval_part(emb(_event(tag="IsBatteryOk", node="battery_guard", wind=0.0, batt=18.0)))
    assert cosine(wind_a, wind_b) > cosine(wind_a, batt)
    # Class is not part of retrieval similarity (cross-class candidates visible)
    wind_c = emb.retrieval_part(emb(_event(wind=22.0, cls="carrier")))
    assert cosine(wind_a, wind_c) > 0.99


def test_bandit_learns_to_stop_calling_llm():
    rng = np.random.default_rng(0)
    b = LinearTS(dim=4, seed=0)
    ctx = np.array([0.5, 0.1, 0.0, 1.0])
    b.ensure_arm("p1")
    # Stored patch recovers (r=1), LLM call costs kappa (r=0.7) for the same context.
    for _ in range(40):
        arm, _ = b.select(ctx, ["p1", CALL_LLM])
        b.update(arm, ctx, 1.0 if arm == "p1" else 0.7)
    picks = [b.select(ctx, ["p1", CALL_LLM])[0] for _ in range(50)]
    assert picks.count("p1") > 40
    b.frozen = True
    assert all(b.select(ctx, ["p1", CALL_LLM])[0] == "p1" for _ in range(5))
    d = b.to_dict(); b2 = LinearTS.from_dict(d)
    assert b2.arms["p1"].n == b.arms["p1"].n


def test_library_roundtrip_and_retrieval(tmp_path):
    emb = ContextEmbedder(HashTextEmbedder(64))
    lib = PatchLibrary(str(tmp_path / "lib.json"))
    ev = _event()
    phi = emb(ev); pr = emb.retrieval_part(phi)
    xml = '<patch operator="WrapRetry" anchor="wp1" num_attempts="3"><rationale>r</rationale></patch>'
    lib.add(LibraryEntry(patch_id=lib.new_id(), patch_xml=xml, phi=phi.tolist(), phi_retrieval=pr.tolist(),
                         origin_class="scout", status="admitted", admitted_for=["scout"]))
    lib2 = PatchLibrary(str(tmp_path / "lib.json"))
    hits = lib2.retrieve(emb.retrieval_part(emb(_event(wind=24.0))), rho=0.5)
    assert len(hits) == 1 and hits[0][0].patch().operator == "WrapRetry"
    assert lib2.retrieve(emb.retrieval_part(emb(_event(tag="IsBatteryOk", node="g", wind=0, batt=15))), rho=0.95) == []


def test_proposal_schema_to_patch_and_static_check():
    from scbt.bt.base_trees import build_base_tree
    from scbt.bt.leaves import NAMESPACE
    from scbt.bt.registry import from_xml
    tok = NAMESPACE.set("/tm1")
    try:
        root = from_xml(build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0)).xml)
    finally:
        NAMESPACE.reset(tok)
    prop = PatchProposal(operator="WrapFallback", anchor="wp1",
                         subtree_xml='<Sequence><WaitSimTime duration_s="{w}"/><GoToNED north="40" east="0" down="-8"/></Sequence>',
                         tunable_ports=[{"name": "w", "min": 1, "max": 30, "initial": 5}], rationale="wait & retry")
    patch = prop.to_patch("p0001")
    assert patch.dim == 1 and static_check(patch, root, DEFAULT_CLASSES["scout"]) == []
    with pytest.raises(Exception):
        PatchProposal(operator="Teleport", anchor="wp1", rationale="x")  # not in grammar


def test_manager_pipeline_with_mock_proposer_offline(tmp_path):
    """Drive PatchManager with a fake runner: first event calls the LLM
    (mock), second similar event should be served from the library."""
    from scbt.bt.base_trees import build_base_tree
    from scbt.bt.leaves import NAMESPACE
    from scbt.bt.registry import from_xml
    from scbt.manager import ManagerConfig, PatchLibrary, PatchManager
    import py_trees

    ns = "/tm2"
    tok = NAMESPACE.set(ns)
    try:
        the_plan = build_base_tree("M1", DEFAULT_CLASSES["scout"], (0, 0, 0))
        the_root = from_xml(the_plan.xml)
    finally:
        NAMESPACE.reset(tok)

    class V:  # fake vehicle
        spec = DEFAULT_CLASSES["scout"]; name = "Drone1"

    class BB:
        goals_done = set()

    class R:  # fake runner facade
        vehicle = V(); plan = the_plan; root = the_root; bb = BB(); events = [None]; energy_used_pct = 0.0; t_start = 0.0

    class Res:
        success = True; sim_time_s = 60.0; energy_pct = 5.0

    mgr = PatchManager(proposer="mock", library=PatchLibrary(str(tmp_path / "l.json")),
                       embedder=ContextEmbedder(HashTextEmbedder(64)),
                       config=ManagerConfig(rho=0.7, seed=1))
    p1 = mgr(_event(), R)
    assert p1 is not None and mgr.stats.llm_calls == 1 and mgr.library.stats()["admitted"] == 1
    mgr.on_episode_end(Res)
    # Teach the bandit that the stored patch works
    for _ in range(15):
        mgr.bandit.update(p1.patch_id, mgr.embed(_event()), 1.0)
        mgr.bandit.update(CALL_LLM, mgr.embed(_event()), 0.6)
    calls = mgr.stats.llm_calls
    R.events.append(None)
    mgr.bandit.frozen = True           # argmax of posterior means -> deterministic
    p2 = mgr(_event(wind=24.0), R)
    assert p2 is not None and p2.patch_id == p1.patch_id
    assert mgr.stats.llm_calls == calls and mgr.stats.library_hits_same == 1
