"""Ariadne Agent – iterative reasoning graph.

This module implements a LangGraph-based agent that performs Z3-like
constraint-solving reasoning over complex texts (philosophy, history,
news, stories).

Graph flow
----------
START → extract_claims → formulate_constraints → iterative_solver
          ↑ (loop)                                       |
          └───────────────────────────────── check_convergence
                                                         │ (converged / max iter)
                                                     synthesize → END
"""

import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from agent.reasoning_prompts import (
    extract_claims_instructions,
    formulate_constraints_instructions,
    solver_instructions,
    synthesis_instructions,
)
from agent.reasoning_schemas import ClaimsList, ConstraintsList, SolverIteration
from agent.reasoning_state import ReasoningState

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_MAX_ITERATIONS = 5


def _get_llm(state: ReasoningState, config: RunnableConfig) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini model, respecting any per-run override."""
    model = state.get("reasoning_model") or _DEFAULT_MODEL
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Node: extract_claims
# ---------------------------------------------------------------------------
def extract_claims(state: ReasoningState, config: RunnableConfig) -> dict:
    """LangGraph node that extracts core claims / facts from the input text.

    Analyses the raw input text and produces a deduplicated list of
    propositions that will be used in subsequent reasoning steps.

    Args:
        state: Current graph state containing ``input_text``.
        config: LangGraph runnable configuration.

    Returns:
        Partial state update with ``claims`` and initial ``reasoning_path`` entry.
    """
    llm = _get_llm(state, config)
    structured_llm = llm.with_structured_output(ClaimsList)

    prompt = extract_claims_instructions.format(input_text=state["input_text"])
    result: ClaimsList = structured_llm.invoke(prompt)

    step = (
        f"**Claim Extraction** – identified {len(result.claims)} claim(s).\n"
        f"Rationale: {result.rationale}"
    )
    return {
        "claims": result.claims,
        "reasoning_path": [step],
    }


# ---------------------------------------------------------------------------
# Node: formulate_constraints
# ---------------------------------------------------------------------------
def formulate_constraints(state: ReasoningState, config: RunnableConfig) -> dict:
    """LangGraph node that derives logical constraints from the extracted claims.

    Identifies logical dependencies, rules, and initial contradictions that
    constrain a valid interpretation of the text.

    Args:
        state: Current graph state containing ``claims`` and ``input_text``.
        config: LangGraph runnable configuration.

    Returns:
        Partial state update with ``constraints``, ``contradictions_found``,
        and a ``reasoning_path`` entry.
    """
    llm = _get_llm(state, config)
    structured_llm = llm.with_structured_output(ConstraintsList)

    claims_text = "\n".join(f"- {c}" for c in state["claims"])
    prompt = formulate_constraints_instructions.format(
        claims=claims_text,
        input_text=state["input_text"],
    )
    result: ConstraintsList = structured_llm.invoke(prompt)

    step = (
        f"**Constraint Formulation** – derived {len(result.constraints)} constraint(s), "
        f"found {len(result.contradictions)} initial contradiction(s).\n"
        f"Rationale: {result.rationale}"
    )
    return {
        "constraints": result.constraints,
        "contradictions_found": result.contradictions,
        "reasoning_path": [step],
        "iteration_count": 0,
        "is_consistent": False,
    }


# ---------------------------------------------------------------------------
# Node: iterative_solver
# ---------------------------------------------------------------------------
def iterative_solver(state: ReasoningState, config: RunnableConfig) -> dict:
    """LangGraph node that performs one iteration of constraint-solving reasoning.

    Attempts to find a consistent interpretation of all claims under the
    derived constraints.  Records its findings and flags whether the
    interpretation is now fully consistent.

    Args:
        state: Current graph state containing ``claims``, ``constraints``,
            ``contradictions_found``, ``solver_notes``, and ``iteration_count``.
        config: LangGraph runnable configuration.

    Returns:
        Partial state update with new ``solver_notes`` entry, updated
        ``contradictions_found``, ``is_consistent`` flag, incremented
        ``iteration_count``, and a ``reasoning_path`` entry.
    """
    llm = _get_llm(state, config)
    structured_llm = llm.with_structured_output(SolverIteration)

    iteration = state.get("iteration_count", 0) + 1
    claims_text = "\n".join(f"- {c}" for c in state["claims"])
    constraints_text = "\n".join(f"- {c}" for c in state.get("constraints", []))
    contradictions_text = (
        "\n".join(f"- {c}" for c in state.get("contradictions_found", []))
        or "(none)"
    )
    solver_notes_text = (
        "\n\n".join(
            f"Iteration {i + 1}:\n{n}"
            for i, n in enumerate(state.get("solver_notes", []))
        )
        or "(first iteration)"
    )

    prompt = solver_instructions.format(
        iteration_count=iteration,
        claims=claims_text,
        constraints=constraints_text,
        contradictions_found=contradictions_text,
        solver_notes=solver_notes_text,
    )
    result: SolverIteration = structured_llm.invoke(prompt)

    step = (
        f"**Solver Iteration {iteration}** – "
        + ("consistent ✓" if result.is_consistent else "not yet consistent ✗")
        + f", {len(result.remaining_contradictions)} contradiction(s) remaining."
    )
    return {
        "solver_notes": [result.notes],
        "contradictions_found": result.remaining_contradictions,
        "is_consistent": result.is_consistent,
        "iteration_count": iteration,
        "reasoning_path": [step],
    }


# ---------------------------------------------------------------------------
# Node: synthesize
# ---------------------------------------------------------------------------
def synthesize(state: ReasoningState, config: RunnableConfig) -> dict:
    """LangGraph node that produces the final reasoning report.

    Summarises all reasoning steps and outputs a structured conclusion
    (or an acknowledgement of unresolvable paradoxes).

    Args:
        state: Full graph state at the end of the solving loop.
        config: LangGraph runnable configuration.

    Returns:
        Partial state update with ``final_conclusion`` and a closing
        ``reasoning_path`` entry.
    """
    llm = _get_llm(state, config)

    claims_text = "\n".join(f"- {c}" for c in state["claims"])
    constraints_text = "\n".join(f"- {c}" for c in state.get("constraints", []))
    contradictions_text = (
        "\n".join(f"- {c}" for c in state.get("contradictions_found", []))
        or "(none)"
    )
    solver_notes_text = "\n\n".join(
        f"Iteration {i + 1}:\n{n}"
        for i, n in enumerate(state.get("solver_notes", []))
    )
    reasoning_path_text = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(state.get("reasoning_path", []))
    )

    prompt = synthesis_instructions.format(
        input_text=state["input_text"],
        claims=claims_text,
        constraints=constraints_text,
        solver_notes=solver_notes_text,
        contradictions_found=contradictions_text,
        is_consistent=state.get("is_consistent", False),
        reasoning_path=reasoning_path_text,
    )

    result = llm.invoke(prompt)
    conclusion = result.content if hasattr(result, "content") else str(result)

    return {
        "final_conclusion": conclusion,
        "reasoning_path": ["**Synthesis** – final conclusion produced."],
    }


# ---------------------------------------------------------------------------
# Routing function: check_convergence
# ---------------------------------------------------------------------------
def check_convergence(state: ReasoningState, config: RunnableConfig) -> str:
    """Routing function that decides whether to loop or proceed to synthesis.

    Returns ``"synthesize"`` when the solver has reached a consistent
    interpretation or has exhausted the allowed iterations.  Otherwise
    returns ``"iterative_solver"`` to continue the loop.

    Args:
        state: Current graph state.
        config: LangGraph runnable configuration.

    Returns:
        Name of the next node to visit.
    """
    max_iter = state.get("max_iterations") or _DEFAULT_MAX_ITERATIONS
    if state.get("is_consistent") or state.get("iteration_count", 0) >= max_iter:
        return "synthesize"
    return "iterative_solver"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
builder = StateGraph(ReasoningState)

builder.add_node("extract_claims", extract_claims)
builder.add_node("formulate_constraints", formulate_constraints)
builder.add_node("iterative_solver", iterative_solver)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "extract_claims")
builder.add_edge("extract_claims", "formulate_constraints")
builder.add_edge("formulate_constraints", "iterative_solver")
builder.add_conditional_edges(
    "iterative_solver",
    check_convergence,
    ["iterative_solver", "synthesize"],
)
builder.add_edge("synthesize", END)

reasoning_graph = builder.compile(name="ariadne-reasoning-agent")
