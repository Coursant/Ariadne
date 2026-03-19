"""State definitions for the Ariadne Agent reasoning graph."""

from __future__ import annotations

import operator
from typing import TypedDict

from typing_extensions import Annotated


class ReasoningState(TypedDict):
    """Overall state for the iterative reasoning graph."""

    input_text: str
    """The raw input text to be analyzed (news, philosophy, history, story, …)."""

    claims: Annotated[list[str], operator.add]
    """Core propositions / facts extracted from the text."""

    constraints: Annotated[list[str], operator.add]
    """Logical dependencies, contradictions or rules found in the text."""

    solver_notes: Annotated[list[str], operator.add]
    """Per-iteration notes produced by the solver node."""

    contradictions_found: Annotated[list[str], operator.add]
    """Any contradictions / unresolved tensions identified during solving."""

    is_consistent: bool
    """True once the solver reaches a coherent, non-contradictory interpretation."""

    iteration_count: int
    """How many solver iterations have been performed so far."""

    max_iterations: int
    """Upper bound on solver iterations before forcing synthesis."""

    reasoning_path: Annotated[list[str], operator.add]
    """Ordered list of reasoning steps accumulated across iterations."""

    final_conclusion: str
    """The final synthesised conclusion produced at the end of the graph."""

    reasoning_model: str
    """Optional override for the Gemini model used during reasoning."""


class ClaimExtractionOutput(TypedDict):
    """Partial state returned by the extract_claims node."""

    claims: list[str]
    reasoning_path: list[str]


class ConstraintFormulationOutput(TypedDict):
    """Partial state returned by the formulate_constraints node."""

    constraints: list[str]
    contradictions_found: list[str]
    reasoning_path: list[str]


class SolverOutput(TypedDict):
    """Partial state returned by the iterative_solver node."""

    solver_notes: list[str]
    contradictions_found: list[str]
    is_consistent: bool
    iteration_count: int
    reasoning_path: list[str]


class SynthesisOutput(TypedDict):
    """Partial state returned by the synthesize node."""

    final_conclusion: str
    reasoning_path: list[str]
