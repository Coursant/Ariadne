import os
from typing import Any

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from agent.reasoning_graph import ClaimsList, ConstraintsList, SolverIteration
from agent import reasoning_graph as reasoning_graph_module


class _FakeLLM:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self._schema = None
        self.last_prompt = None

    def with_structured_output(self, schema: Any) -> "_FakeLLM":
        self._schema = schema
        return self

    def invoke(self, prompt: str) -> Any:
        self.last_prompt = prompt
        if self._schema is ClaimsList:
            return ClaimsList(
                claims=[
                    "The treaty was signed in 1648.",
                    "The same text also claims the war ended in 1649.",
                    "The narrator says both claims come from the same archive.",
                ],
                rationale="Split the historical text into distinct factual statements.",
            )
        if self._schema is ConstraintsList:
            return ConstraintsList(
                constraints=[
                    "If the war ended in 1649, a treaty signed in 1648 requires explanation."
                ],
                contradictions=[
                    "The timeline appears internally inconsistent unless the treaty was preliminary."
                ],
                rationale="Derived temporal dependency from the extracted claims.",
            )
        if self._schema is SolverIteration:
            return SolverIteration(
                notes="Interpreted the 1648 treaty as preliminary and 1649 as final settlement.",
                remaining_contradictions=[],
                is_consistent=True,
            )

        class _Response:
            content = "## Conclusion\nA consistent interpretation was produced from a complex text."

        return _Response()


def test_extract_claims_supports_complex_text(monkeypatch) -> None:
    state = {
        "input_text": (
            "The chronicle says the treaty was signed in 1648, yet another chapter says "
            "the war ended in 1649, and both claims are attributed to one archive."
        )
    }
    fake_llm = _FakeLLM(state)

    monkeypatch.setattr(reasoning_graph_module, "_get_llm", lambda *_: fake_llm)

    result = reasoning_graph_module.extract_claims(state, {})

    assert len(result["claims"]) == 3
    assert "Claim Extraction" in result["reasoning_path"][0]
    assert "identified 3 claim(s)" in result["reasoning_path"][0]
    assert "Text to" in fake_llm.last_prompt


def test_reasoning_graph_analyzes_complex_text_end_to_end(monkeypatch) -> None:
    initial_state = {
        "input_text": (
            "A philosophical essay claims free will is real, then says every decision is "
            "fully determined by prior causes, and finally argues moral responsibility still holds."
        ),
        "claims": [],
        "constraints": [],
        "solver_notes": [],
        "contradictions_found": [],
        "is_consistent": False,
        "iteration_count": 0,
        "max_iterations": 3,
        "reasoning_path": [],
        "final_conclusion": "",
        "reasoning_model": "gemini-2.5-flash",
    }
    monkeypatch.setattr(
        reasoning_graph_module, "_get_llm", lambda state, config: _FakeLLM(state)
    )

    result = reasoning_graph_module.reasoning_graph.invoke(initial_state)

    assert len(result["claims"]) == 3
    assert result["is_consistent"] is True
    assert result["iteration_count"] == 1
    assert "consistent interpretation was produced" in result["final_conclusion"].lower()
    assert any("Claim Extraction" in step for step in result["reasoning_path"])
    assert any("Synthesis" in step for step in result["reasoning_path"])
