"""LLM-judge evaluation pipeline for checking whether Ariadne agents finish tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agent.graph import graph
from agent.reasoning_graph import reasoning_graph

load_dotenv()

AgentType = Literal["reasoning", "research"]


@dataclass(frozen=True)
class EvaluationCase:
    """One test case used to evaluate an agent."""

    case_id: str
    agent: AgentType
    task: str
    input_text: str
    expected_outcome: str


class JudgeVerdict(BaseModel):
    """Structured output returned by the LLM judge."""

    passed: bool = Field(description="Whether the agent completed the task.")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Task completion score between 0 and 1.",
    )
    rationale: str = Field(description="Short explanation for the verdict.")
    missing_requirements: list[str] = Field(
        default_factory=list,
        description="Missing requirements that prevented full task completion.",
    )


@dataclass(frozen=True)
class CaseResult:
    """Evaluation result for one case."""

    case_id: str
    agent: AgentType
    passed: bool
    score: float
    rationale: str
    missing_requirements: list[str]
    agent_output: str


class CasesFileEntry(TypedDict):
    """Supported JSON shape for one case file entry."""

    case_id: str
    agent: AgentType
    task: str
    input_text: str
    expected_outcome: str


DEFAULT_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        case_id="reasoning_liar_paradox",
        agent="reasoning",
        task="Analyze the text and provide a coherent conclusion.",
        input_text=(
            "This statement is false. If it is true, then it must be false; "
            "if false, then it becomes true."
        ),
        expected_outcome=(
            "Identify the paradox, explain why it is self-referential, and provide "
            "a careful final conclusion about unresolved contradiction."
        ),
    ),
    EvaluationCase(
        case_id="reasoning_socratic_claim",
        agent="reasoning",
        task="Extract claims and synthesize a logically consistent interpretation.",
        input_text=(
            "Socrates said he knew nothing, but he also claimed that recognizing "
            "one's ignorance is a form of wisdom."
        ),
        expected_outcome=(
            "Extract key claims and conclude that Socratic wisdom can be interpreted "
            "as meta-knowledge rather than factual omniscience."
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-judge evaluation to measure whether Ariadne agents complete tasks."
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        help=(
            "Optional JSON file with cases. If omitted, built-in reasoning cases are used."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-flash",
        help="Model used by the LLM judge.",
    )
    parser.add_argument(
        "--agent-model",
        default="gemini-2.5-flash",
        help="Model override passed to the evaluated agent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_report.json"),
        help="Path to write the JSON evaluation report.",
    )
    return parser.parse_args()


def _load_cases(cases_file: Path | None) -> list[EvaluationCase]:
    if cases_file is None:
        return list(DEFAULT_CASES)

    raw_content = cases_file.read_text(encoding="utf-8")
    raw_cases = cast(list[CasesFileEntry], json.loads(raw_content))
    return [
        EvaluationCase(
            case_id=entry["case_id"],
            agent=entry["agent"],
            task=entry["task"],
            input_text=entry["input_text"],
            expected_outcome=entry["expected_outcome"],
        )
        for entry in raw_cases
    ]


def _run_reasoning_case(case: EvaluationCase, agent_model: str) -> str:
    result = reasoning_graph.invoke(
        {
            "input_text": case.input_text,
            "claims": [],
            "constraints": [],
            "solver_notes": [],
            "contradictions_found": [],
            "is_consistent": False,
            "iteration_count": 0,
            "max_iterations": 5,
            "reasoning_path": [],
            "final_conclusion": "",
            "reasoning_model": agent_model,
        }
    )
    output = result.get("final_conclusion")
    return output if isinstance(output, str) else ""


def _run_research_case(case: EvaluationCase, agent_model: str) -> str:
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=case.input_text)],
            "initial_search_query_count": 3,
            "max_research_loops": 2,
            "reasoning_model": agent_model,
        }
    )
    messages = result.get("messages")
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        return content if isinstance(content, str) else str(content)
    return ""


def _run_agent(case: EvaluationCase, agent_model: str) -> str:
    if case.agent == "reasoning":
        return _run_reasoning_case(case, agent_model)
    return _run_research_case(case, agent_model)


def _judge_case(
    case: EvaluationCase,
    agent_output: str,
    judge_model: str,
) -> JudgeVerdict:
    judge = ChatGoogleGenerativeAI(model=judge_model, temperature=0.0, max_retries=2)
    structured_judge = judge.with_structured_output(JudgeVerdict)
    prompt = f"""
You are an evaluation judge for an AI agent.
Decide whether the agent completed the task.

Task:
{case.task}

Input:
{case.input_text}

Expected outcome:
{case.expected_outcome}

Agent output:
{agent_output}

Scoring rubric:
- 1.0: fully completed all requirements from expected outcome.
- 0.7~0.9: mostly complete, minor missing detail.
- 0.4~0.6: partially complete.
- 0.0~0.3: failed to complete core task.
"""
    return cast(JudgeVerdict, structured_judge.invoke(prompt))


def _write_report(path: Path, case_results: list[CaseResult]) -> None:
    passed_count = sum(1 for item in case_results if item.passed)
    total = len(case_results)
    avg_score = sum(item.score for item in case_results) / total if total else 0.0
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_cases": total,
        "passed_cases": passed_count,
        "completion_rate": passed_count / total if total else 0.0,
        "average_score": avg_score,
        "results": [asdict(item) for item in case_results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the full evaluation pipeline."""
    args = _parse_args()
    cases = _load_cases(args.cases_file)
    results: list[CaseResult] = []

    for case in cases:
        agent_output = _run_agent(case, args.agent_model)
        verdict = _judge_case(case, agent_output, args.judge_model)
        case_result = CaseResult(
            case_id=case.case_id,
            agent=case.agent,
            passed=verdict.passed,
            score=verdict.score,
            rationale=verdict.rationale,
            missing_requirements=verdict.missing_requirements,
            agent_output=agent_output,
        )
        results.append(case_result)
        status = "PASS" if case_result.passed else "FAIL"
        print(f"[{status}] {case_result.case_id} score={case_result.score:.2f}")

    _write_report(args.output, results)
    print(f"\nEvaluation report written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
