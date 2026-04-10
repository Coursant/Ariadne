import json
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from agent.reasoning_graph import ClaimsList, ConstraintsList, SolverIteration
from agent import reasoning_graph as reasoning_graph_module

DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "test_data" / "public_reasoning_dataset.json"
)
PUBLIC_REASONING_DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))


class _FakeLLM:
    def __init__(self, expected_claim_count: int = 3) -> None:
        self.expected_claim_count = expected_claim_count
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
                    f"Extracted claim {idx + 1}"
                    for idx in range(self.expected_claim_count)
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


@pytest.mark.parametrize("dataset_case", PUBLIC_REASONING_DATASET)
def test_extract_claims_supports_complex_text(monkeypatch, dataset_case) -> None:
    state = {"input_text": dataset_case["text"]}
    fake_llm = _FakeLLM(expected_claim_count=dataset_case["expected_claim_count"])

    monkeypatch.setattr(reasoning_graph_module, "_get_llm", lambda *_: fake_llm)

    result = reasoning_graph_module.extract_claims(state, {})

    assert len(result["claims"]) == dataset_case["expected_claim_count"]
    assert "Claim Extraction" in result["reasoning_path"][0]
    assert (
        f"identified {dataset_case['expected_claim_count']} claim(s)"
        in result["reasoning_path"][0]
    )
    assert "Text to" in fake_llm.last_prompt
    assert dataset_case["text"] in fake_llm.last_prompt


def test_reasoning_graph_analyzes_complex_text_end_to_end(monkeypatch) -> None:
    dataset_case = PUBLIC_REASONING_DATASET[2]
    initial_state = {
        "input_text": dataset_case["text"],
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
    fake_llm = _FakeLLM(expected_claim_count=dataset_case["expected_claim_count"])
    monkeypatch.setattr(
        reasoning_graph_module, "_get_llm", lambda state, config: fake_llm
    )

    result = reasoning_graph_module.reasoning_graph.invoke(initial_state)

    assert len(result["claims"]) == dataset_case["expected_claim_count"]
    assert result["is_consistent"] is True
    assert result["iteration_count"] == 1
    assert "consistent interpretation was produced" in result["final_conclusion"].lower()
    assert any("Claim Extraction" in step for step in result["reasoning_path"])
    assert any("Synthesis" in step for step in result["reasoning_path"])
