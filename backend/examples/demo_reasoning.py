r"""Demo / CLI runner for the Ariadne Agent reasoning graph.

Usage
-----
From the ``backend/`` directory (with the virtual environment activated and
a valid ``GEMINI_API_KEY`` in the environment or a ``.env`` file):

    python examples/demo_reasoning.py "Your text here"

Or with optional flags:

    python examples/demo_reasoning.py "Your text here" \\
        --max-iterations 3 \\
        --model gemini-2.5-flash

The script prints the complete reasoning path followed by the final
conclusion to stdout.
"""

import argparse
import os
import sys

# Ensure the src directory is on the path when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.reasoning_graph import reasoning_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Built-in sample texts for quick demos
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "liar_paradox": (
        "This statement is false. If the statement is true, then what it says must "
        "hold: it is false. But if it is false, then what it says does not hold, so "
        "it must be true. Therefore the statement is both true and false."
    ),
    "trolley": (
        "It is always wrong to use a person merely as a means to an end. "
        "Redirecting a runaway trolley to kill one person in order to save five "
        "uses that one person as a means. Therefore it is wrong to redirect the "
        "trolley. Yet allowing five people to die when you could prevent it seems "
        "equally wrong. The driver faces an inescapable moral dilemma."
    ),
    "socrates": (
        "Socrates claimed to know nothing. Yet he spent his life questioning "
        "politicians, poets, and craftsmen and demonstrating that they did not "
        "know what they claimed to know. He knew that he was wiser than others "
        "because he, unlike them, did not mistake ignorance for knowledge. "
        "Therefore Socrates did know at least one thing: the limits of his own "
        "knowledge."
    ),
}


def main() -> None:
    """Run the Ariadne reasoning agent from the command line."""
    parser = argparse.ArgumentParser(
        description="Ariadne Agent – iterative constraint-solving text reasoning"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "text",
        nargs="?",
        help="The text to reason about.",
    )
    input_group.add_argument(
        "--sample",
        choices=list(SAMPLE_TEXTS.keys()),
        help="Use one of the built-in sample texts.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of solver iterations (default: 5).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate reasoning steps.",
    )

    args = parser.parse_args()

    input_text = args.text if args.text else SAMPLE_TEXTS[args.sample]

    print("\n" + "=" * 70)
    print("Ariadne Agent – Iterative Reasoning")
    print("=" * 70)
    print(f"\nInput text:\n{input_text}\n")
    print("=" * 70)
    print(f"Model: {args.model}  |  Max iterations: {args.max_iterations}")
    print("=" * 70 + "\n")

    initial_state = {
        "input_text": input_text,
        "claims": [],
        "constraints": [],
        "solver_notes": [],
        "contradictions_found": [],
        "is_consistent": False,
        "iteration_count": 0,
        "max_iterations": args.max_iterations,
        "reasoning_path": [],
        "final_conclusion": "",
        "reasoning_model": args.model,
    }

    result = reasoning_graph.invoke(initial_state)

    if args.verbose:
        print("── Reasoning Path ──────────────────────────────────────────────────")
        for step in result.get("reasoning_path", []):
            print(f"  • {step}")
        print()

    print("── Final Conclusion ────────────────────────────────────────────────")
    print(result.get("final_conclusion", "(no conclusion generated)"))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
