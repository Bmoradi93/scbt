"""Structure proposers: the LLM side of the Patch Manager.

``Proposer.propose(event, cls, examples, feedback)`` returns a
``PatchProposal`` or raises ``ProposalError``. The repair loop (re-prompt
with the static checker's message) lives in ``PatchManager``, not here, so
every proposer gets it for free.

* ``ADKProposer``  -- Google ADK ``SequentialAgent(ContextAnalyst, PatchProposer)``
                      with Pydantic ``output_schema`` on both; Gemini by
                      default, any LiteLLM model via ``model=``.
* ``MockProposer`` -- offline stand-in built on the hand-written patches;
                      lets the full System-2 pipeline run without a key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Iterable, List, Optional, Protocol

from ..observer.events import FailureEvent
from ..sim.config import ClassSpec
from . import prompts
from .schemas import FailureDiagnosis, PatchProposal

LOG = logging.getLogger("scbt.manager")


class ProposalError(RuntimeError):
    pass


class Proposer(Protocol):
    name: str

    def propose(self, event: FailureEvent, cls: ClassSpec, examples: Iterable[tuple],
                feedback: Optional[str], plan_hint: str) -> PatchProposal: ...


# ----------------------------------------------------------------------------

class ADKProposer:
    """Google ADK pipeline. Requires GOOGLE_API_KEY (Gemini) or a LiteLLM
    model string such as ``openai/gpt-4o`` with the provider's key."""

    name = "adk"

    def __init__(self, model: str = "gemini-3.1-pro-preview", temperature: float = 0.2, timeout_s: float = 180.0,
                 regenerate: bool = False):
        self.timeout_s = timeout_s
        self.regenerate = regenerate
        from google.adk.agents import LlmAgent, SequentialAgent
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        self.model_name = model
        self._types = types
        model_obj = model
        if "/" in model and not model.startswith("gemini"):
            from google.adk.models.lite_llm import LiteLlm  # needs google-adk[extensions]
            model_obj = LiteLlm(model=model)
        gen_cfg = types.GenerateContentConfig(temperature=temperature)

        self.analyst = LlmAgent(
            name="ContextAnalyst", model=model_obj, instruction=prompts.ANALYST_INSTRUCTION,
            output_schema=FailureDiagnosis, output_key="diagnosis", generate_content_config=gen_cfg,
        )
        self.proposer = LlmAgent(
            name="PatchProposer", model=model_obj,
            instruction=(prompts.REGENERATE_INSTRUCTION if regenerate else prompts.PROPOSER_INSTRUCTION)
            + "\n\nDiagnosis from the analyst: {diagnosis}\n",
            output_schema=PatchProposal, output_key="proposal", generate_content_config=gen_cfg,
        )
        self.pipeline = SequentialAgent(name="PatchManagerLLM", sub_agents=[self.analyst, self.proposer])
        self.runner = InMemoryRunner(agent=self.pipeline, app_name="scbt")
        self.calls = 0
        self.total_latency_s = 0.0
        self.last_trace: List[dict] = []

    def _build_message(self, event, cls, examples, feedback, plan_hint) -> str:
        from ..bt.registry import to_xml_string
        runner = getattr(event, "_runner", None)
        extra = ""
        if self.regenerate and runner is not None:
            from ..bt.registry import find_node
            ms = find_node(runner.root, "mission_seq")
            extra = "Current mission subtree (to be replaced entirely):\n" + (to_xml_string(ms) if ms is not None else "")
        return "\n\n".join([
            prompts.render_event(event),
            extra,
            f"Mission goals: {plan_hint}",
            prompts.OBJECTIVE_TEXT,
            prompts.render_manifest(cls),
            prompts.GRAMMAR_TEXT,
            "Related library entries (admitted or rejected):\n" + prompts.render_examples(examples),
            f"VERIFIER FEEDBACK ON PREVIOUS ATTEMPT: {feedback}" if feedback else "",
        ])

    def propose(self, event, cls, examples, feedback, plan_hint) -> PatchProposal:
        msg = self._build_message(event, cls, examples, feedback, plan_hint)
        t0 = time.time()
        # The episode runs inside an asyncio loop; ADK/genai need their own
        # long-lived loop (closing one per call leaks aclose() tasks).
        fut = asyncio.run_coroutine_threadsafe(self._run(msg), self._loop())
        try:
            proposal = fut.result(timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001  (timeout, API error, schema error)
            fut.cancel()
            self.calls += 1
            self.total_latency_s += time.time() - t0
            raise ProposalError(f"{type(exc).__name__}: {exc}") from exc
        self.calls += 1
        self.total_latency_s += time.time() - t0
        if self.regenerate and (proposal.operator != "ReplaceSubtree" or proposal.anchor != "mission_seq"):
            raise ProposalError(f"regenerate baseline requires ReplaceSubtree on mission_seq, got "
                                f"{proposal.operator} on {proposal.anchor}")
        return proposal

    _bg_loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def _loop(cls) -> asyncio.AbstractEventLoop:
        if cls._bg_loop is None:
            import threading
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="scbt-adk-loop", daemon=True).start()
            cls._bg_loop = loop
        return cls._bg_loop

    async def _run(self, msg: str) -> PatchProposal:
        sid = f"s{int(time.time() * 1e6)}"
        await self.runner.session_service.create_session(app_name="scbt", user_id="fleet", session_id=sid)
        content = self._types.Content(role="user", parts=[self._types.Part(text=msg)])
        final_text = None
        self.last_trace = []
        async for ev in self.runner.run_async(user_id="fleet", session_id=sid, new_message=content):
            self.last_trace.append({"author": ev.author, "final": ev.is_final_response()})
            if ev.author == "PatchProposer" and ev.content and ev.content.parts:
                txt = "".join(p.text or "" for p in ev.content.parts)
                if txt.strip():
                    final_text = txt
        if not final_text:
            raise ProposalError("no proposal produced")
        try:
            return PatchProposal.model_validate_json(final_text)
        except Exception as exc:  # noqa: BLE001
            raise ProposalError(f"proposal did not match schema: {exc}") from exc


# ----------------------------------------------------------------------------

class MockProposer:
    """Offline proposer: reuses the hand-written recoveries as if an LLM had
    produced them. Honors ``feedback`` by giving up on the second attempt
    (so the repair loop is exercised but cannot loop forever)."""

    name = "mock"

    def __init__(self):
        self.calls = 0
        self.total_latency_s = 0.0

    def propose(self, event, cls, examples, feedback, plan_hint) -> PatchProposal:
        from ..bench import policies
        self.calls += 1
        if feedback:
            raise ProposalError(f"mock cannot repair: {feedback}")

        class _R:  # minimal runner facade for policies.handwritten
            pass

        runner = getattr(event, "_runner", None)
        patch = policies.handwritten(event, runner) if runner is not None else None
        if patch is None:
            raise ProposalError("mock has no recovery for this failure")
        from xml.etree import ElementTree as ET
        return PatchProposal(
            operator=patch.operator, anchor=patch.anchor,
            subtree_xml=ET.tostring(patch.subtree, encoding="unicode") if patch.subtree is not None else "",
            num_attempts=patch.num_attempts,
            tunable_ports=[{"name": p.name, "min": p.min, "max": p.max, "initial": p.initial} for p in patch.tunable],
            rationale=patch.rationale or "mock",
        )


def make_proposer(spec: str, regenerate: bool = False) -> Proposer:
    """``"mock"`` or an ADK model name (``gemini-3.1-pro-preview``, ``openai/gpt-4o``...)."""
    if spec == "mock":
        return MockProposer()
    if spec.startswith("gemini") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise ProposalError("GOOGLE_API_KEY is not set")
    return ADKProposer(model=spec, regenerate=regenerate)
