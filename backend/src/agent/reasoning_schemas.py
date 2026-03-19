"""Pydantic schemas used for structured LLM outputs in the reasoning graph."""

from typing import List

from pydantic import BaseModel, Field


class ClaimsList(BaseModel):
    """Structured output for the claim-extraction step."""

    claims: List[str] = Field(
        description=(
            "A list of core propositions or factual statements extracted from the text. "
            "Each entry should be a single, self-contained claim."
        )
    )
    rationale: str = Field(
        description="Brief explanation of how you identified these claims."
    )


class ConstraintsList(BaseModel):
    """Structured output for the constraint-formulation step."""

    constraints: List[str] = Field(
        description=(
            "A list of logical dependencies, rules, or conditions derived from the text "
            "that any consistent interpretation must satisfy."
        )
    )
    contradictions: List[str] = Field(
        description=(
            "A list of apparent contradictions or tensions detected in the text. "
            "Use an empty list if none are found."
        )
    )
    rationale: str = Field(
        description="Brief explanation of how the constraints were derived."
    )


class SolverIteration(BaseModel):
    """Structured output for a single solver iteration."""

    notes: str = Field(
        description=(
            "Reasoning notes for this iteration: which claims are consistent with the "
            "constraints, which are in tension, and what adjustments were made."
        )
    )
    remaining_contradictions: List[str] = Field(
        description=(
            "Contradictions or tensions that remain unresolved after this iteration. "
            "Use an empty list if the interpretation is now consistent."
        )
    )
    is_consistent: bool = Field(
        description=(
            "True if all claims can be reconciled with all constraints after this "
            "iteration (i.e., a coherent interpretation has been reached)."
        )
    )
