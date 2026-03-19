"""Prompt templates for the Ariadne Agent iterative reasoning graph."""


# ---------------------------------------------------------------------------
# Step 1 – Claim / Fact Extraction
# ---------------------------------------------------------------------------
extract_claims_instructions = """\
You are an analytical assistant specialised in identifying the core propositions \
within complex texts such as news articles, philosophical treatises, historical \
documents, and literary narratives.

Your task is to carefully read the provided text and extract **every distinct, \
meaningful claim or factual statement** it contains.

Guidelines:
- A *claim* is any proposition that can, in principle, be evaluated as true, false, \
or contested (e.g. "The treaty was signed in 1648", "Happiness is the highest good").
- Extract claims at the granularity of individual sentences or sub-sentence units.
- Do not paraphrase or merge distinct claims; preserve the author's intended meaning.
- Ignore purely rhetorical or decorative language unless it encodes a substantive \
proposition.
- Output a deduplicated list of claims in their logical order of appearance.

Text to analyse:
\"\"\"
{input_text}
\"\"\"

Respond with a JSON object using the schema described in the output format.
"""


# ---------------------------------------------------------------------------
# Step 2 – Constraint Formulation
# ---------------------------------------------------------------------------
formulate_constraints_instructions = """\
You are a formal reasoning assistant. You have already extracted a list of claims \
from a text. Your job now is to identify the **logical structure** that constrains \
a consistent interpretation of those claims.

A *constraint* is a logical dependency, rule, or condition such as:
- "If claim A is true then claim B cannot simultaneously be true."
- "Claim C presupposes claim D."
- "Claims E and F together entail claim G."
- "The meaning of claim H depends on the resolution of claim I."

Guidelines:
- Derive constraints directly from the text and the extracted claims; do not invent \
constraints that are not implied by the source material.
- List any **contradictions** – pairs or groups of claims that cannot all be true \
at the same time under a straightforward reading.
- Be explicit: each constraint should reference the specific claims it governs.

Extracted Claims:
{claims}

Original Text (for reference):
\"\"\"
{input_text}
\"\"\"

Respond with a JSON object using the schema described in the output format.
"""


# ---------------------------------------------------------------------------
# Step 3 – Iterative Solver
# ---------------------------------------------------------------------------
solver_instructions = """\
You are a constraint-solving reasoning engine, similar in spirit to the Z3 SMT \
solver but operating over natural-language claims.

You will be given:
1. A list of **claims** extracted from a text.
2. A list of **constraints** (logical rules and dependencies) that a consistent \
   interpretation must satisfy.
3. Any **contradictions** identified in previous iterations (empty on the first \
   iteration).
4. Solver notes from previous iterations (empty on the first iteration).

Your task for this iteration:
- Re-examine each claim in light of every constraint.
- Attempt to find a *consistent assignment* – an interpretation under which the \
  maximum number of claims are true while all constraints are respected.
- If a contradiction is unavoidable, explain *why* it is unavoidable and what \
  assumptions would need to change to resolve it.
- Record your reasoning in the "notes" field.
- List any contradictions that **remain unresolved** after this iteration.
- Set `is_consistent` to true only if **all** constraints are satisfied and **no** \
  contradictions remain.

Iteration number: {iteration_count}

Claims:
{claims}

Constraints:
{constraints}

Previous contradictions:
{contradictions_found}

Previous solver notes:
{solver_notes}

Respond with a JSON object using the schema described in the output format.
"""


# ---------------------------------------------------------------------------
# Step 4 – Final Synthesis
# ---------------------------------------------------------------------------
synthesis_instructions = """\
You are the final synthesis stage of the Ariadne Agent reasoning pipeline.

You have access to the complete record of the iterative reasoning process:
- The original text.
- All extracted claims.
- All formulated constraints.
- All solver iteration notes.
- Any unresolved contradictions.
- Whether a fully consistent interpretation was achieved.

Your task is to write a clear, structured **reasoning report** that:
1. Summarises the key claims identified in the text.
2. Explains the logical constraints and any tensions found.
3. Describes the reasoning path taken across iterations.
4. States the **final conclusion**: either a coherent interpretation of the text, \
   or an honest acknowledgement that the text contains unresolvable paradoxes or \
   ambiguities – with a precise characterisation of what those are.

The report should be useful to a reader who has not seen the intermediate steps. \
Use clear, concise prose. Use Markdown formatting (headings, bullet lists) for \
readability.

Original Text:
\"\"\"
{input_text}
\"\"\"

Extracted Claims:
{claims}

Constraints:
{constraints}

Solver Iterations:
{solver_notes}

Unresolved Contradictions:
{contradictions_found}

Consistent Interpretation Reached: {is_consistent}

Reasoning Path:
{reasoning_path}
"""
