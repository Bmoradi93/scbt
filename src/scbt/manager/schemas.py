"""Schemas for the LLM side of the Patch Manager (ADK ``output_schema``).

Keeping these as Pydantic models means malformed model output is rejected
by ADK before our static checker runs (paper Sec. IV-B, "checkable by
construction"). ``PatchProposal.to_patch()`` converts to the grammar object.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from ..patch.grammar import OPERATORS, Patch, TunablePort, parse_patch_xml

Operator = Literal["WrapFallback", "GuardWith", "WrapRetry", "InsertBefore", "InsertAfter", "ReplaceSubtree"]


class FailureDiagnosis(BaseModel):
    """Output of the ContextAnalyst agent."""

    summary: str = Field(description="One sentence: what failed and under which conditions.")
    likely_cause: str = Field(description="Most plausible physical cause, from the telemetry.")
    recovery_strategy: str = Field(description="One sentence: the recovery idea, in terms of the available leaves.")
    anchor: str = Field(description="Name of the node the patch should anchor on (usually the failed node or its parent).")


class TunablePortSpec(BaseModel):
    name: str = Field(description="Placeholder name used as {name} inside subtree_xml.")
    min: float
    max: float
    initial: float


class PatchProposal(BaseModel):
    """Output of the PatchProposer agent: exactly one patch in grammar O."""

    operator: Operator
    anchor: str = Field(description="Existing node name to anchor the edit on.")
    subtree_xml: str = Field(
        default="",
        description="XML of the inserted subtree (one root element). Empty only for WrapRetry.",
    )
    num_attempts: int = Field(default=3, description="WrapRetry only.")
    tunable_ports: List[TunablePortSpec] = Field(default_factory=list)
    rationale: str

    def to_patch(self, patch_id: str = "") -> Patch:
        parts = [f'<patch operator="{self.operator}" anchor="{self.anchor}" num_attempts="{self.num_attempts}">']
        if self.subtree_xml.strip():
            parts.append(f"<subtree>{self.subtree_xml.strip()}</subtree>")
        if self.tunable_ports:
            parts.append("<tunable_ports>")
            for p in self.tunable_ports:
                parts.append(f'<port name="{p.name}" min="{p.min}" max="{p.max}" initial="{p.initial}"/>')
            parts.append("</tunable_ports>")
        parts.append(f"<rationale>{_xml_escape(self.rationale)}</rationale></patch>")
        patch = parse_patch_xml("".join(parts))
        patch.patch_id = patch_id
        return patch


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


assert set(Operator.__args__) == set(OPERATORS)  # keep schema and grammar in sync
