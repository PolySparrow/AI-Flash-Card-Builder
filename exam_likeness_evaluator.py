import json
from typing import Any, Dict, List, Optional

import ollama

from exam_reference import EXAM_REFERENCE_PROFILE


EXAM_LIKENESS_MODEL = "qwen2.5:14b"
EXAM_LIKENESS_MAX_RETRIES = 2
MIN_EXAM_LIKENESS_SCORE = 0.75


def clean_llm_json(raw: str) -> str:
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    return raw


def build_exam_likeness_prompt(
    question: Dict[str, Any],
    domain_name: str,
    task_statement: str,
    task_desc: str,
    official_style_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    question_json = json.dumps(question, indent=2, ensure_ascii=False)

    profile_json = json.dumps(
        EXAM_REFERENCE_PROFILE,
        indent=2,
        ensure_ascii=False,
    )

    examples_json = json.dumps(
        official_style_examples or [],
        indent=2,
        ensure_ascii=False,
    )

    return f"""You are a psychometric and exam-content reviewer.

Your role is NOT to validate product documentation facts. A separate evaluator
already does that.

Your role is to determine whether a generated question resembles a high-quality
question that belongs in the target certification exam.

Evaluate the question based on the exam blueprint, the supplied exam-style
reference profile, and any authorized official example patterns.

[TARGET DOMAIN]
Domain: {domain_name}
Task Objective: {task_statement}
Expected Knowledge: {task_desc}

[EXAM REFERENCE PROFILE]
{profile_json}

[AUTHORIZED STYLE EXAMPLES OR STYLE NOTES]
{examples_json}

[QUESTION TO REVIEW]
{question_json}

[EVALUATE THESE DIMENSIONS]

1. Objective alignment
   - Does the item actually test the stated task objective?
   - Does it assess an important capability rather than an obscure detail?

2. Scenario realism
   - Would a practicing developer plausibly encounter this situation?
   - Are the constraints, symptoms, filenames, tools, and trade-offs realistic?

3. Exam-likeness
   - Does the item resemble the expected scenario-driven certification style?
   - Does it require judgment or application rather than recall alone?
   - Is its complexity appropriate for an intermediate/advanced developer exam?

4. Difficulty calibration
   - Is it too easy because one option is obvious?
   - Is it too difficult because it depends on hidden assumptions?
   - Does it require a meaningful but fair decision?

5. Distractor quality
   - Are distractors credible implementation choices?
   - Do they model realistic engineering mistakes?
   - Are distractors comparable in specificity and plausibility?

6. Item-writing quality
   - Does the stem contain sufficient information?
   - Is there one best answer for a single-answer item?
   - Is it concise enough for an exam but detailed enough to establish context?

7. Novelty
   - Does it appear generic, repetitive, templated, or overly similar to common
     generated questions?
   - Flag repeated phrasing or recurring artificial patterns.

[DECISION RULES]
- "accept": suitable for a high-quality certification question bank.
- "revise": technically usable but not sufficiently exam-like yet.
- "reject": fundamentally unrepresentative, trivial, vague, artificial, or
  misaligned with the objective.

Return ONLY valid JSON:

{{
  "decision": "accept | revise | reject",
  "overall_exam_likeness_score": 0.0,
  "scores": {{
    "objective_alignment": 0.0,
    "scenario_realism": 0.0,
    "exam_likeness": 0.0,
    "difficulty_calibration": 0.0,
    "distractor_quality": 0.0,
    "item_writing_quality": 0.0,
    "novelty": 0.0
  }},
  "estimated_difficulty": "too_easy | appropriate | too_hard",
  "is_representative_of_target_exam": true,
  "issues": [
    {{
      "severity": "critical | major | minor",
      "category": "objective_alignment | realism | difficulty | distractors | wording | novelty",
      "description": "Specific issue."
    }}
  ],
  "revision_instructions": [
    "Concrete guidance for improving exam resemblance."
  ],
  "summary": "Brief conclusion."
}}
"""


def evaluate_exam_likeness(
    question: Dict[str, Any],
    domain_name: str,
    task_statement: str,
    task_desc: str,
    official_style_examples: Optional[List[Dict[str, Any]]] = None,
    model: str = EXAM_LIKENESS_MODEL,
    max_retries: int = EXAM_LIKENESS_MAX_RETRIES,
) -> Dict[str, Any]:
    last_error: Optional[str] = None

    for _attempt in range(max_retries):
        prompt = build_exam_likeness_prompt(
            question=question,
            domain_name=domain_name,
            task_statement=task_statement,
            task_desc=task_desc,
            official_style_examples=official_style_examples,
        )

        if last_error:
            prompt += (
                "\n\nThe previous response was invalid: "
                f"{last_error}\nReturn only valid JSON."
            )

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "format": "json",
                },
            )

            evaluation = json.loads(
                clean_llm_json(response["response"])
            )

            score = float(
                evaluation.get("overall_exam_likeness_score", 0.0)
            )

            decision = evaluation.get("decision", "reject")

            if decision not in {"accept", "revise", "reject"}:
                decision = "reject"

            # Enforce a local acceptance rule instead of trusting the evaluator.
            if decision == "accept" and score < MIN_EXAM_LIKENESS_SCORE:
                decision = "revise"

            evaluation["decision"] = decision
            evaluation["overall_exam_likeness_score"] = max(
                0.0,
                min(1.0, score),
            )

            return evaluation

        except Exception as exc:
            last_error = str(exc)

    return {
        "decision": "reject",
        "overall_exam_likeness_score": 0.0,
        "scores": {},
        "estimated_difficulty": "too_hard",
        "is_representative_of_target_exam": False,
        "issues": [
            {
                "severity": "critical",
                "category": "wording",
                "description": (
                    "Exam-likeness evaluator failed: "
                    f"{last_error}"
                ),
            }
        ],
        "revision_instructions": [],
        "summary": "Could not reliably assess exam resemblance.",
    }