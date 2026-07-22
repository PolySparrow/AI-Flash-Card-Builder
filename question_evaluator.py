import json
from typing import Any, Dict, List, Optional, Tuple

import ollama

# -----------------------------------------------------------------------------
# Evaluator Configuration
# -----------------------------------------------------------------------------

EVALUATOR_MODEL = "qwen2.5:14b"
EVALUATOR_MAX_RETRIES = 2
MIN_EVALUATION_SCORE = 0.75

VALID_OPTION_KEYS = {"A", "B", "C", "D"}


# -----------------------------------------------------------------------------
# Local Structural Validation
# -----------------------------------------------------------------------------

def clean_llm_json(raw: str) -> str:
    """Remove an accidental Markdown code fence from an LLM JSON response."""
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    return raw


def validate_question_structure(
    question: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Perform cheap deterministic validation before spending an LLM call
    on the evaluator.

    Returns:
        (True, "") for a valid structure.
        (False, error_message) otherwise.
    """
    if "error" in question:
        return False, "Question generation failed."

    required_fields = [
        "task_id",
        "scenario",
        "question",
        "options",
        "option_explanations",
        "explanation",
    ]

    for field in required_fields:
        if not question.get(field):
            return False, f"Missing or empty required field: {field}"

    options = question.get("options")

    if not isinstance(options, dict):
        return False, "`options` must be a JSON object."

    if set(options.keys()) != VALID_OPTION_KEYS:
        return False, "Options must contain exactly A, B, C, and D."

    for option_key, option_text in options.items():
        if not isinstance(option_text, str) or not option_text.strip():
            return False, f"Option {option_key} is empty or invalid."

    raw_correct = question.get("correct_answers") or question.get(
        "correct_answer"
    )

    if isinstance(raw_correct, str):
        correct_answers = {raw_correct.strip().upper()}
    elif isinstance(raw_correct, list):
        correct_answers = {
            answer.strip().upper()
            for answer in raw_correct
            if isinstance(answer, str)
        }
    else:
        return False, "Missing or invalid correct answer field."

    if not correct_answers:
        return False, "No correct answer was provided."

    if not correct_answers.issubset(VALID_OPTION_KEYS):
        return False, "Correct answer includes an invalid option key."

    is_multi_answer = len(correct_answers) > 1

    if is_multi_answer:
        if len(correct_answers) != 2:
            return (
                False,
                "Multiple-answer questions must have exactly two correct answers.",
            )

        if not question["question"].strip().startswith("[SELECT TWO]"):
            return (
                False,
                "Multiple-answer questions must start with [SELECT TWO].",
            )
    elif len(correct_answers) != 1:
        return False, "Single-answer questions must have exactly one answer."

    option_explanations = question.get("option_explanations")

    if not isinstance(option_explanations, dict):
        return False, "`option_explanations` must be a JSON object."

    if set(option_explanations.keys()) != VALID_OPTION_KEYS:
        return (
            False,
            "Option explanations must contain exactly A, B, C, and D.",
        )

    for option_key, explanation in option_explanations.items():
        if not isinstance(explanation, str) or not explanation.strip():
            return False, f"Explanation for option {option_key} is invalid."

    return True, ""


# -----------------------------------------------------------------------------
# Prompt Construction
# -----------------------------------------------------------------------------

def build_evaluator_prompt(
    question: Dict[str, Any],
    task_statement: str,
    task_desc: str,
    rag_context: str,
) -> str:
    """
    Build a deterministic quality-review prompt.

    The evaluator is intentionally constrained to the supplied documentation.
    That helps prevent it from approving questions based on hallucinated
    product behavior or outside assumptions.
    """
    question_json = json.dumps(question, indent=2, ensure_ascii=False)

    return f"""You are a rigorous technical certification exam evaluator.

Your job is to assess whether a generated multiple-choice question is accurate,
unambiguous, scenario-driven, technically correct, and supported by the
provided documentation.

Evaluate only from:
1. The stated exam objective.
2. The supplied documentation context.
3. The generated question itself.

Do not assume undocumented product behavior.
Do not approve a question merely because it sounds plausible.
If the intended correct answer cannot be supported by the documentation,
the question must be rejected.

[EXAM OBJECTIVE]
Standard Objective: {task_statement}
Expected Core Knowledge: {task_desc}

[DOCUMENTATION REFERENCE]
{rag_context}

[GENERATED QUESTION]
{question_json}

[EVALUATION CRITERIA]

1. Documentation grounding
   - Is the declared correct answer explicitly or strongly supported by the
     documentation reference?
   - Does the explanation make unsupported claims?
   - Do any options rely on invented APIs, CLI flags, configuration syntax,
     SDK behavior, or product capabilities?

2. Answer validity
   - For a single-answer question, is exactly one answer clearly best?
   - For a [SELECT TWO] question, are exactly two answers clearly correct?
   - Is the declared answer key logically correct?
   - Are any distractors arguably correct based on the supplied context?

3. Certification quality
   - Is the prompt scenario-driven and concrete?
   - Does the question assess the specified objective rather than trivia?
   - Does it present a realistic engineering decision?
   - Does it avoid obvious answer giveaways?

4. Distractor quality
   - Are incorrect answers plausible to a junior or intermediate developer?
   - Are they technically wrong for a specific reason?
   - Are they not absurd, irrelevant, or clearly weaker solely because they
     are much longer or more detailed?

5. Technical accuracy
   - Check SDK names, command-line flags, YAML/frontmatter syntax, APIs,
     tool behavior, and configuration details against the documentation.
   - Flag invented or unsupported details as critical issues.

6. Clarity
   - Is the question wording precise?
   - Is there enough information to select the intended answer?
   - Does it avoid ambiguity, double negatives, or conflicting assumptions?

[DECISION RULES]

Use "accept" only when:
- The declared answer is supported.
- The question has no material ambiguity.
- The score is at least 0.75.
- There are no critical technical accuracy or grounding issues.

Use "revise" when:
- The item is salvageable.
- Wording, distractors, specificity, or explanation needs improvement.
- The core question can be repaired without changing its learning objective.

Use "reject" when:
- The correct answer is unsupported or incorrect.
- More than one answer is valid in a single-answer question.
- There are invented technical facts or APIs.
- The question cannot be safely repaired from the supplied context.
- The question is structurally invalid.

[OUTPUT REQUIREMENTS]

Return ONLY one raw JSON object. Do not use Markdown fences.
Use exactly this structure:

{{
  "decision": "accept",
  "overall_score": 0.0,
  "scores": {{
    "grounding": 0.0,
    "answer_validity": 0.0,
    "scenario_quality": 0.0,
    "distractor_quality": 0.0,
    "technical_accuracy": 0.0,
    "clarity": 0.0
  }},
  "correct_answer_is_supported": true,
  "declared_answer_key_is_correct": true,
  "ambiguous_options": [],
  "unsupported_claims": [],
  "issues": [
    {{
      "severity": "critical",
      "field": "question",
      "description": "Specific, actionable issue."
    }}
  ],
  "revision_instructions": [
    "Concrete repair instruction."
  ],
  "summary": "Brief evaluation summary."
}}

The allowed decision values are exactly:
- "accept"
- "revise"
- "reject"

The allowed severity values are exactly:
- "critical"
- "major"
- "minor"
"""


# -----------------------------------------------------------------------------
# Evaluation Response Normalization
# -----------------------------------------------------------------------------

def build_rejection_evaluation(
    summary: str,
    description: str,
    field: str = "evaluation",
) -> Dict[str, Any]:
    """Return a consistently shaped rejection response."""
    return {
        "decision": "reject",
        "overall_score": 0.0,
        "scores": {
            "grounding": 0.0,
            "answer_validity": 0.0,
            "scenario_quality": 0.0,
            "distractor_quality": 0.0,
            "technical_accuracy": 0.0,
            "clarity": 0.0,
        },
        "correct_answer_is_supported": False,
        "declared_answer_key_is_correct": False,
        "ambiguous_options": [],
        "unsupported_claims": [],
        "issues": [
            {
                "severity": "critical",
                "field": field,
                "description": description,
            }
        ],
        "revision_instructions": [],
        "summary": summary,
    }


def normalize_evaluation(
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ensure the evaluator response has predictable keys and safe values.

    An invalid or incomplete evaluator response becomes a rejection instead
    of accidentally allowing a question into the exported exam bank.
    """
    if not isinstance(evaluation, dict):
        return build_rejection_evaluation(
            "Evaluator returned a non-object response.",
            "The evaluator output was not a JSON object.",
        )

    decision = str(evaluation.get("decision", "")).strip().lower()

    if decision not in {"accept", "revise", "reject"}:
        return build_rejection_evaluation(
            "Evaluator returned an invalid decision.",
            f"Unsupported decision value: {decision!r}",
        )

    try:
        overall_score = float(evaluation.get("overall_score", 0.0))
    except (TypeError, ValueError):
        overall_score = 0.0

    overall_score = max(0.0, min(1.0, overall_score))

    raw_scores = evaluation.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    score_keys = [
        "grounding",
        "answer_validity",
        "scenario_quality",
        "distractor_quality",
        "technical_accuracy",
        "clarity",
    ]

    scores: Dict[str, float] = {}

    for key in score_keys:
        try:
            value = float(raw_scores.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0

        scores[key] = max(0.0, min(1.0, value))

    issues = evaluation.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    revision_instructions = evaluation.get("revision_instructions", [])
    if not isinstance(revision_instructions, list):
        revision_instructions = []

    ambiguous_options = evaluation.get("ambiguous_options", [])
    if not isinstance(ambiguous_options, list):
        ambiguous_options = []

    unsupported_claims = evaluation.get("unsupported_claims", [])
    if not isinstance(unsupported_claims, list):
        unsupported_claims = []

    normalized = {
        "decision": decision,
        "overall_score": overall_score,
        "scores": scores,
        "correct_answer_is_supported": bool(
            evaluation.get("correct_answer_is_supported", False)
        ),
        "declared_answer_key_is_correct": bool(
            evaluation.get("declared_answer_key_is_correct", False)
        ),
        "ambiguous_options": ambiguous_options,
        "unsupported_claims": unsupported_claims,
        "issues": issues,
        "revision_instructions": revision_instructions,
        "summary": str(evaluation.get("summary", "")).strip(),
    }

    return normalized


# -----------------------------------------------------------------------------
# Main Evaluator Entry Point
# -----------------------------------------------------------------------------

def evaluate_question_with_retry(
    question: Dict[str, Any],
    task_statement: str,
    task_desc: str,
    rag_context: str,
    model: str = EVALUATOR_MODEL,
    max_retries: int = EVALUATOR_MAX_RETRIES,
    min_score: float = MIN_EVALUATION_SCORE,
) -> Dict[str, Any]:
    """
    Evaluate one generated question.

    Args:
        question:
            Generated exam question dictionary.

        task_statement:
            Taxonomy objective statement for this task.

        task_desc:
            Taxonomy description / expected knowledge for this task.

        rag_context:
            Documentation context used to generate and evaluate the question.

        model:
            Ollama model name used as evaluator.

        max_retries:
            Number of evaluator calls if parsing or transport fails.

        min_score:
            Minimum score required for an `accept` decision.

    Returns:
        A normalized evaluation dictionary containing `decision`,
        `overall_score`, detailed scores, issues, and revision guidance.
    """
    valid, validation_error = validate_question_structure(question)

    if not valid:
        return build_rejection_evaluation(
            summary="Question failed deterministic structural validation.",
            description=validation_error,
            field="structure",
        )

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        prompt = build_evaluator_prompt(
            question=question,
            task_statement=task_statement,
            task_desc=task_desc,
            rag_context=rag_context,
        )

        if last_error:
            prompt += f"""

[PREVIOUS RESPONSE FAILURE]
The previous evaluator response was invalid.

Failure:
{last_error}

Return only a valid JSON object matching the required output structure.
"""

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "format": "json",
                },
            )

            raw_response = response.get("response", "")
            parsed = json.loads(clean_llm_json(raw_response))
            evaluation = normalize_evaluation(parsed)

            # Never accept an unsupported answer, even if the evaluator
            # accidentally labels it "accept".
            # Never accept an unsupported answer, even if the evaluator
            # accidentally labels it "accept".
            if not evaluation["correct_answer_is_supported"]:
                evaluation["decision"] = "reject"
                evaluation["revision_instructions"].append(
                    "Ground the correct answer explicitly in the supplied "
                    "documentation before resubmitting."
                )

            # Never accept an item if the evaluator itself says the declared
            # answer key is wrong.
            if not evaluation["declared_answer_key_is_correct"]:
                evaluation["decision"] = "reject"
                evaluation["revision_instructions"].append(
                    "Correct the answer key and ensure every distractor is "
                    "unambiguously incorrect."
                )

            # Reject items containing claims that are not supported by the
            # documentation, such as invented flags, APIs, or config keys.
            if evaluation["unsupported_claims"]:
                evaluation["decision"] = "reject"
                evaluation["revision_instructions"].append(
                    "Regenerate the item using only technical claims "
                    "explicitly present in the supplied documentation."
                )

            # A critical issue means the item cannot safely be exported.
            has_critical_issue = any(
                isinstance(issue, dict)
                and str(issue.get("severity", "")).lower() == "critical"
                for issue in evaluation["issues"]
            )

            if has_critical_issue:
                evaluation["decision"] = "reject"
                evaluation["revision_instructions"].append(
                    "Resolve all critical issues before this item can be "
                    "accepted."
                )

            # Enforce your local acceptance threshold independently of the LLM.
            if (
                evaluation["decision"] == "accept"
                and evaluation["overall_score"] < min_score
            ):
                evaluation["decision"] = "revise"
                evaluation["revision_instructions"].append(
                    f"Improve the question to reach the local acceptance "
                    f"threshold of {min_score:.2f}."
                )

            # Critical issues cannot be exported as accepted questions.
            for issue in evaluation["issues"]:
                if not isinstance(issue, dict):
                    continue

                severity = str(issue.get("severity", "")).lower()

                if severity == "critical" and evaluation["decision"] == "accept":
                    evaluation["decision"] = "revise"
                    evaluation["revision_instructions"].append(
                        "Resolve all critical issues before accepting the item."
                    )
                    break

            return evaluation

        except json.JSONDecodeError as exc:
            last_error = f"Invalid JSON from evaluator: {exc}"

        except Exception as exc:
            last_error = f"Evaluator request failed: {exc}"

        print(
            f"  ✗ Evaluator attempt {attempt}/{max_retries} failed: "
            f"{last_error}"
        )

    return build_rejection_evaluation(
        summary="Evaluator could not complete a reliable review.",
        description=(
            f"Evaluation failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        ),
        field="evaluation",
    )


# -----------------------------------------------------------------------------
# Revision Agent
# -----------------------------------------------------------------------------

def build_revision_prompt(
    question: Dict[str, Any],
    evaluation: Dict[str, Any],
    task_statement: str,
    task_desc: str,
    rag_context: str,
) -> str:
    """
    Build a question-repair prompt using the evaluator's structured feedback.
    """
    question_json = json.dumps(question, indent=2, ensure_ascii=False)
    evaluation_json = json.dumps(evaluation, indent=2, ensure_ascii=False)

    return f"""You are repairing a technical certification exam question.

Return a complete replacement question that preserves the target exam objective
but resolves every valid issue reported by the evaluator.

Use only the supplied documentation as the factual source of truth.
Do not invent APIs, command-line flags, configuration options, SDK behavior,
tool behavior, syntax, or product capabilities.

[EXAM OBJECTIVE]
Standard Objective: {task_statement}
Expected Core Knowledge: {task_desc}

[DOCUMENTATION REFERENCE]
{rag_context}

[ORIGINAL QUESTION]
{question_json}

[EVALUATOR FEEDBACK]
{evaluation_json}

[REPAIR REQUIREMENTS]
1. Return a complete replacement question, not a list of edits.
2. Keep the original task_id and scenario.
3. Keep exactly four options: A, B, C, and D.
4. Retain a single correct answer unless the original question begins with
   [SELECT TWO].
5. If it is [SELECT TWO], provide exactly two correct answers.
6. Make the declared answer key fully supported by the documentation.
7. Ensure distractors are plausible but wrong for a documented reason.
8. Remove unsupported technical claims.
9. Ensure the question is scenario-driven, concrete, and unambiguous.
10. Include option-level explanations for all four options.
11. Return only raw, valid JSON. Do not use Markdown fences.

[REQUIRED JSON FORMAT]
{{
  "task_id": "{question.get("task_id", "")}",
  "scenario": "{question.get("scenario", "")}",
  "question": "Scenario-driven question text",
  "options": {{
    "A": "Option text",
    "B": "Option text",
    "C": "Option text",
    "D": "Option text"
  }},
  "correct_answer": "A",
  "option_explanations": {{
    "A": "Why this option is correct or incorrect.",
    "B": "Why this option is correct or incorrect.",
    "C": "Why this option is correct or incorrect.",
    "D": "Why this option is correct or incorrect."
  }},
  "explanation": "Complete explanation grounded in the documentation."
}}

If the original question is a [SELECT TWO] question, replace
"correct_answer" with:

"correct_answers": ["A", "C"]
"""


def revise_question(
    question: Dict[str, Any],
    evaluation: Dict[str, Any],
    task_statement: str,
    task_desc: str,
    rag_context: str,
    model: str,
) -> Dict[str, Any]:
    """
    Ask an LLM to repair a question based on evaluator feedback.

    This function preserves source/domain metadata used by the original script.
    It raises JSONDecodeError or Ollama exceptions to let the caller decide
    whether to retry or reject the item.
    """
    prompt = build_revision_prompt(
        question=question,
        evaluation=evaluation,
        task_statement=task_statement,
        task_desc=task_desc,
        rag_context=rag_context,
    )

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.1,
            "format": "json",
        },
    )

    revised_question = json.loads(clean_llm_json(response["response"]))

    # Preserve metadata used by your JSON/CSV exports.
    revised_question["_primary_source"] = question.get("_primary_source", "")
    revised_question["_primary_domain_id"] = question.get(
        "_primary_domain_id",
        "",
    )

    return revised_question


# -----------------------------------------------------------------------------
# Batch-Friendly Convenience Function
# -----------------------------------------------------------------------------

def evaluate_and_optionally_revise(
    question: Dict[str, Any],
    task_statement: str,
    task_desc: str,
    rag_context: str,
    evaluator_model: str = EVALUATOR_MODEL,
    revision_model: Optional[str] = None,
    auto_revise: bool = True,
    max_revision_attempts: int = 1,
    min_score: float = MIN_EVALUATION_SCORE,
) -> Dict[str, Any]:
    """
    Evaluate a question and optionally make one or more repair attempts.

    The returned question contains an `_evaluation` key. A repaired question
    additionally contains `_revision_history`.

    This is the easiest entry point to call from your existing generator.
    """
    evaluation = evaluate_question_with_retry(
        question=question,
        task_statement=task_statement,
        task_desc=task_desc,
        rag_context=rag_context,
        model=evaluator_model,
        min_score=min_score,
    )

    question["_evaluation"] = evaluation

    if not auto_revise or evaluation["decision"] != "revise":
        return question

    revision_model = revision_model or evaluator_model
    revision_history: List[Dict[str, Any]] = []

    for attempt in range(1, max_revision_attempts + 1):
        previous_evaluation = question.get("_evaluation", {})

        try:
            revised_question = revise_question(
                question=question,
                evaluation=previous_evaluation,
                task_statement=task_statement,
                task_desc=task_desc,
                rag_context=rag_context,
                model=revision_model,
            )

            revised_evaluation = evaluate_question_with_retry(
                question=revised_question,
                task_statement=task_statement,
                task_desc=task_desc,
                rag_context=rag_context,
                model=evaluator_model,
                min_score=min_score,
            )

            revision_history.append(
                {
                    "attempt": attempt,
                    "previous_decision": previous_evaluation.get("decision"),
                    "previous_score": previous_evaluation.get("overall_score"),
                    "new_decision": revised_evaluation.get("decision"),
                    "new_score": revised_evaluation.get("overall_score"),
                }
            )

            revised_question["_evaluation"] = revised_evaluation
            revised_question["_revision_history"] = revision_history

            question = revised_question

            if revised_evaluation["decision"] != "revise":
                break

        except Exception as exc:
            question.setdefault("_revision_history", []).append(
                {
                    "attempt": attempt,
                    "error": str(exc),
                }
            )
            break

    return question