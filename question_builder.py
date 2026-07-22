import csv
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from question_evaluator import evaluate_and_optionally_revise
from taxonomy import TAXONOMY, TASK_TO_CUSTOM_CATEGORY

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------

FAISS_INDEX_PATH = "./faiss_claude_docs"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_API_BASE = "http://127.0.0.1:11434"

LLM_MODEL = "qwen2.5:14b"

# CPU: 1-2. Single GPU: 2-4. Multi-GPU: 4-8.
MAX_WORKERS = 4

# Retry a failed generation this many times before giving up.
AUTO_REVISE = True
MAX_REVISION_ATTEMPTS = 2
MAX_CANDIDATES_PER_SLOT = 3

# Automatically repair questions marked "revise" by the evaluator.
AUTO_REVISE = True
MAX_REVISION_ATTEMPTS = 1

# Do not generate questions if a task has no documentation context.
REQUIRE_RAG_CONTEXT = True

MAX_OPTIONS = 12
TOP_K_DOCS = 4
MIN_CONFIDENCE = 0.5
EXAM_SIZE = 60
MULTI_ANSWER_RATE = 0.0

DOMAIN_WEIGHTS: Dict[str, float] = {
    "1": 0.27,
    "2": 0.18,
    "3": 0.20,
    "4": 0.20,
    "5": 0.15,
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Scenario 1: Customer Support Resolution Agent": {
        "description": (
            "You are building a customer support resolution agent using the "
            "Claude Agent SDK. The agent handles high-ambiguity requests such "
            "as returns, billing disputes, and account issues. It can access "
            "your own backend systems through developer-provided custom MCP "
            "tools such as get_customer, lookup_order, process_refund, and "
            "escalate_to_human. Your target is 80%+ first-contact resolution "
            "while knowing when to escalate."
        ),
        "primary_domains": ["1", "2", "5"],
    },
    "Scenario 2: Code Generation with Claude Code": {
        "description": (
            "You are using Claude Code to accelerate software development. "
            "Your team uses it for code generation, refactoring, debugging, "
            "and documentation. You need to integrate it into your development "
            "workflow and understand when to use planning versus direct "
            "execution, based only on documented product behavior."
        ),
        "primary_domains": ["3", "5"],
    },
    "Scenario 3: Multi-Agent Research System": {
        "description": (
            "You are building a multi-agent research system using the Claude "
            "Agent SDK. A coordinator delegates work to specialized agents "
            "that search approved sources, analyze supplied documents, "
            "synthesize findings, and generate reports. The system must "
            "produce comprehensive and cited reports."
        ),
        "primary_domains": ["1", "2", "5"],
    },
    "Scenario 4: Developer Productivity with Claude": {
        "description": (
            "You are building developer productivity tools using the Claude "
            "Agent SDK. The system helps engineers explore unfamiliar "
            "codebases, understand legacy systems, generate boilerplate, and "
            "automate repetitive work. It may use documented built-in tools "
            "and integrate with developer-provided MCP servers."
        ),
        "primary_domains": ["1", "2", "3"],
    },
    "Scenario 5: Claude Code for Continuous Integration": {
        "description": (
            "You are integrating Claude Code into a CI/CD pipeline. The system "
            "runs automated code reviews, generates test cases, and provides "
            "feedback on pull requests. You need to design workflows that "
            "produce actionable feedback and minimize false positives."
        ),
        "primary_domains": ["3", "4"],
    },
    "Scenario 6: Structured Data Extraction": {
        "description": (
            "You are building a structured data extraction system using Claude. "
            "The system extracts information from unstructured documents, "
            "validates output using JSON schemas, and integrates results with "
            "downstream systems. It must handle edge cases gracefully and "
            "maintain high accuracy."
        ),
        "primary_domains": ["4", "5"],
    },
}

# Generic identifiers allowed in question text even if they are not wrapped in
# backticks in the RAG source. Keep this list intentionally small.
ALLOWED_GENERIC_IDENTIFIERS: Set[str] = {
    "API",
    "CLI",
    "JSON",
    "MCP",
    "SDK",
    "YAML",
}


# -----------------------------------------------------------------------------
# 2. Exam Blueprint
# -----------------------------------------------------------------------------

def select_exam_scenarios(n: int = 4) -> Dict[str, str]:
    scenario_keys = list(SCENARIOS.keys())

    for _ in range(200):
        chosen = random.sample(scenario_keys, n)
        covered: Set[str] = set()

        for key in chosen:
            covered.update(SCENARIOS[key]["primary_domains"])

        if covered >= set(DOMAIN_WEIGHTS.keys()):
            return {
                key: SCENARIOS[key]["description"]
                for key in chosen
            }

    chosen = random.sample(scenario_keys, n)
    covered: Set[str] = set()

    for key in chosen:
        covered.update(SCENARIOS[key]["primary_domains"])

    for domain_id in set(DOMAIN_WEIGHTS.keys()) - covered:
        candidates = [
            key
            for key in scenario_keys
            if (
                domain_id in SCENARIOS[key]["primary_domains"]
                and key not in chosen
            )
        ]

        if candidates:
            chosen[random.randrange(len(chosen))] = random.choice(candidates)

    return {
        key: SCENARIOS[key]["description"]
        for key in chosen
    }


def compute_domain_question_counts(
    exam_size: int,
    weights: Dict[str, float],
) -> Dict[str, int]:
    exact = {
        domain_id: weight * exam_size
        for domain_id, weight in weights.items()
    }

    floors = {
        domain_id: math.floor(value)
        for domain_id, value in exact.items()
    }

    remainder = exam_size - sum(floors.values())

    by_fraction = sorted(
        exact,
        key=lambda domain_id: exact[domain_id] - floors[domain_id],
        reverse=True,
    )

    counts = dict(floors)

    for index in range(remainder):
        counts[by_fraction[index]] += 1

    return counts


def build_task_question_plan(
    domain_counts: Dict[str, int],
) -> Dict[str, int]:
    plan: Dict[str, int] = {}

    for domain_id, total_questions in domain_counts.items():
        tasks = list(TAXONOMY[domain_id]["tasks"].keys())

        if not tasks:
            continue

        base, extra = divmod(total_questions, len(tasks))

        for index, task_id in enumerate(tasks):
            plan[task_id] = base + (1 if index < extra else 0)

    return plan


# -----------------------------------------------------------------------------
# 3. FAISS / Metadata Index
# -----------------------------------------------------------------------------

def get_vectorstore() -> FAISS:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'."
        )

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_API_BASE,
    )

    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_task_doc_index(
    vectorstore: FAISS,
) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}

    for _doc_id, doc in vectorstore.docstore._dict.items():
        meta = doc.metadata

        if meta.get("classification_confidence", 1.0) < MIN_CONFIDENCE:
            continue

        task_ids: Set[str] = set()

        primary_task_id = meta.get("primary_task_id")
        if primary_task_id:
            task_ids.add(str(primary_task_id))

        for secondary_task_id in meta.get("secondary_task_ids") or []:
            task_ids.add(str(secondary_task_id))

        entry = {
            "content": doc.page_content,
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "confidence": meta.get("classification_confidence", 1.0),
            "primary_domain_id": meta.get("primary_domain_id", ""),
        }

        for task_id in task_ids:
            index.setdefault(task_id, []).append(entry)

    for task_id in index:
        index[task_id].sort(
            key=lambda document: document["confidence"],
            reverse=True,
        )

    return index


def get_context_for_task(
    task_doc_index: Dict[str, List[Dict[str, Any]]],
    task_id: str,
    k: int = TOP_K_DOCS,
) -> str:
    """
    Return labeled documentation excerpts for one task.

    An empty string means no reliable source material exists for the task.
    The caller should skip generation rather than asking the LLM to guess.
    """
    docs = task_doc_index.get(task_id, [])[:k]

    if not docs:
        return ""

    return "\n\n---\n\n".join(
        (
            f"[DOC_{index}]\n"
            f"Source: {doc['source']}\n"
            f"Document Section: {doc['title']}\n"
            f"Content:\n{doc['content']}"
        )
        for index, doc in enumerate(docs, start=1)
    )


def get_top_doc_meta(
    task_doc_index: Dict[str, List[Dict[str, Any]]],
    task_id: str,
    domain_id: str,
) -> Tuple[str, str]:
    """
    Return the highest-confidence document's source and domain metadata.
    """
    docs = task_doc_index.get(task_id, [])

    if docs:
        return (
            docs[0]["source"],
            docs[0]["primary_domain_id"] or domain_id,
        )

    return "", domain_id


def get_scenario_for_task(
    domain_id: str,
    active_scenarios: Dict[str, str],
) -> Tuple[str, str]:
    eligible = [
        (name, description)
        for name, description in active_scenarios.items()
        if domain_id in SCENARIOS[name]["primary_domains"]
    ]

    if eligible:
        return random.choice(eligible)

    fallback_name = random.choice(list(active_scenarios.keys()))

    return fallback_name, active_scenarios[fallback_name]


# -----------------------------------------------------------------------------
# 4. Validation Helpers
# -----------------------------------------------------------------------------

def clean_llm_json(raw: str) -> str:
    """
    Remove accidental Markdown fences from a model response.
    """
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    return raw


def extract_backtick_identifiers(text: str) -> Set[str]:
    """
    Extract code-formatted identifiers such as `--flag`, `method()`, or
    `ENVIRONMENT_VARIABLE`.
    """
    return set(re.findall(r"`([^`]+)`", text))


def get_question_text(question: Dict[str, Any]) -> str:
    """
    Join fields that might contain technical identifiers.
    """
    parts = [
        str(question.get("question", "")),
        str(question.get("explanation", "")),
    ]

    options = question.get("options", {})
    if isinstance(options, dict):
        parts.extend(str(value) for value in options.values())

    option_explanations = question.get("option_explanations", {})
    if isinstance(option_explanations, dict):
        parts.extend(str(value) for value in option_explanations.values())

    return "\n".join(parts)


def validate_identifiers_against_context(
    question: Dict[str, Any],
    rag_context: str,
) -> Tuple[bool, List[str]]:
    """
    Reject code-formatted identifiers that do not occur in the RAG context.

    This is deliberately strict to catch invented CLI flags, configuration
    keys, tool names, environment variables, and SDK methods.
    """
    known_identifiers = extract_backtick_identifiers(rag_context)
    used_identifiers = extract_backtick_identifiers(
        get_question_text(question)
    )

    unsupported = sorted(
        identifier
        for identifier in used_identifiers
        if (
            identifier not in known_identifiers
            and identifier not in ALLOWED_GENERIC_IDENTIFIERS
        )
    )

    return not unsupported, unsupported


# -----------------------------------------------------------------------------
# 5. LLM Question Generation
# -----------------------------------------------------------------------------

def build_prompt(
    scenario_name: str,
    scenario_desc: str,
    task_id: str,
    task_statement: str,
    task_desc: str,
    rag_context: str,
    is_multi: bool,
) -> str:
    """
    Build a constrained question-generation prompt.

    The supplied RAG content is the sole factual authority.
    """
    focus_areas = [
        (
            "A production symptom, edge case, or operational constraint that "
            "can be addressed using the supplied documentation"
        ),
        (
            "A design decision where the supplied documentation establishes "
            "a supported approach, limitation, or trade-off"
        ),
        (
            "A workflow or integration decision directly supported by the "
            "supplied documentation"
        ),
        (
            "A reliability, safety, debugging, or error-handling decision "
            "explicitly supported by the supplied documentation"
        ),
    ]

    selected_focus = random.choice(focus_areas)

    if is_multi:
        type_instruction = (
            "Generate a MULTIPLE-ANSWER question with exactly two correct "
            "answers out of four. Prefix the question text with "
            "'[SELECT TWO]'."
        )
        answer_field = '"correct_answers": ["A", "C"]'
    else:
        type_instruction = (
            "Generate a SINGLE-ANSWER question with exactly one correct "
            "answer out of four."
        )
        answer_field = '"correct_answer": "A"'

    return f"""You are writing one technical certification exam question.

Your primary requirement is factual accuracy. The supplied documentation is
the only source of truth. Do not use outside knowledge, assumptions, examples,
or remembered product behavior.

[EXAM SCENARIO CONTEXT]
Scenario Name: {scenario_name}
Scenario Baseline: {scenario_desc}

[EXAM BLUEPRINT OBJECTIVE]
Task ID: {task_id}
Standard Objective: {task_statement}
Expected Core Knowledge: {task_desc}

[QUESTION FOCUS]
Write a question about:
{selected_focus}

[DOCUMENTATION REFERENCE]
{rag_context}

[NON-NEGOTIABLE GROUNDING RULES]

1. Every factual product claim must be supported by the documentation reference.
2. Every technical identifier must appear in the documentation reference before
   you use it. This includes CLI flags, command names, SDK classes, SDK methods,
   SDK parameters, tool names, MCP capabilities, environment variables,
   configuration keys, file formats, YAML or JSON syntax, session features,
   and agent capabilities.
3. Never invent a technical identifier because it sounds plausible.
4. Never claim a capability is built in unless the documentation says so.
5. Hypothetical application details such as a repository path, organization
   name, or business requirement are allowed. However, do not imply that a
   hypothetical custom tool, API, or workflow is built into Claude, Claude
   Code, the Agent SDK, or MCP.
6. If the supplied documentation is insufficient to create an accurate item,
   return exactly this JSON object:

{{
  "cannot_generate": true,
  "reason": "Insufficient documentation grounding for this task."
}}

[QUESTION WRITING REQUIREMENTS]

1. Start with a concrete engineering situation, symptom, requirement, or
   implementation decision.
2. Test the assigned task objective rather than a definition or trivia.
3. Include exactly four options: A, B, C, and D.
4. The correct answer must be directly supported by the documentation.
5. Incorrect options must be wrong because they contradict the supplied
   documentation or misuse a documented concept.
6. Do not use invented flags, syntax, environment variables, tool names,
   API methods, configuration keys, or product behavior in distractors.
7. Avoid vague claims such as "best practice" or "deterministic" unless the
   supplied documentation explicitly explains that idea.
8. Distractors must be plausible but clearly inferior based on evidence.
9. Keep the scenario concise enough for a certification exam.
10. Cite source labels such as DOC_1 or DOC_2 in explanations.

[QUESTION TYPE]
{type_instruction}

[REQUIRED JSON OUTPUT]

Return only valid JSON. Do not include Markdown, analysis, or code fences.

{{
  "task_id": "{task_id}",
  "scenario": "{scenario_name}",
  "question": "A concise, scenario-driven question.",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  {answer_field},
  "option_explanations": {{
    "A": "Why this option is correct or incorrect, grounded in DOC_n.",
    "B": "Why this option is correct or incorrect, grounded in DOC_n.",
    "C": "Why this option is correct or incorrect, grounded in DOC_n.",
    "D": "Why this option is correct or incorrect, grounded in DOC_n."
  }},
  "explanation": "Explain why the answer is correct using only the supplied documentation and cite DOC_n labels.",
  "evidence": {{
    "correct_answer_docs": ["DOC_1"]
  }}
}}
"""


def generate_question_with_retry(
    scenario_name: str,
    scenario_desc: str,
    task_id: str,
    task_statement: str,
    task_desc: str,
    rag_context: str,
    question_index: int,
    total_questions: int,
    primary_source: str = "",
    primary_domain_id: str = "",
) -> Dict[str, Any]:
    """
    Generate one question, retrying malformed or ungrounded outputs.
    """
    is_multi = random.random() < MULTI_ANSWER_RATE
    last_error: Optional[str] = None
    raw_response: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = build_prompt(
            scenario_name=scenario_name,
            scenario_desc=scenario_desc,
            task_id=task_id,
            task_statement=task_statement,
            task_desc=task_desc,
            rag_context=rag_context,
            is_multi=is_multi,
        )

        if last_error:
            prompt += f"""

[PREVIOUS ATTEMPT FAILED]

Failure:
{last_error}

Return only valid JSON. Do not include extra text.
"""

        try:
            response = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": 0.15,
                    "format": "json",
                },
            )

            raw_response = response["response"]
            parsed = json.loads(clean_llm_json(raw_response))

            if parsed.get("cannot_generate"):
                return {
                    "task_id": task_id,
                    "scenario": scenario_name,
                    "error": parsed.get(
                        "reason",
                        "Model could not generate a grounded question.",
                    ),
                    "_primary_source": primary_source,
                    "_primary_domain_id": primary_domain_id,
                }

            identifiers_valid, unsupported_identifiers = (
                validate_identifiers_against_context(
                    question=parsed,
                    rag_context=rag_context,
                )
            )

            if not identifiers_valid:
                last_error = (
                    "Generated question used technical identifiers not found "
                    f"in the documentation: {unsupported_identifiers}"
                )

                print(
                    f"  ✗ [{task_id}] Q{question_index}/{total_questions} "
                    f"unsupported identifiers: {unsupported_identifiers}"
                )

                continue

            parsed["_primary_source"] = primary_source
            parsed["_primary_domain_id"] = primary_domain_id

            print(
                f"  ✓ [{task_id}] Q{question_index}/{total_questions}"
                + (f" (attempt {attempt})" if attempt > 1 else "")
            )

            return parsed

        except json.JSONDecodeError as exc:
            last_error = f"Invalid JSON: {exc}"

            print(
                f"  ✗ [{task_id}] Q{question_index}/{total_questions} "
                f"JSON error attempt {attempt}/{MAX_RETRIES}: {exc}"
            )

        except Exception as exc:
            last_error = str(exc)

            print(
                f"  ✗ [{task_id}] Q{question_index}/{total_questions} "
                f"unexpected error attempt {attempt}/{MAX_RETRIES}: {exc}"
            )

    return {
        "task_id": task_id,
        "scenario": scenario_name,
        "error": (
            f"Failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        ),
        "raw_response": raw_response,
        "_primary_source": primary_source,
        "_primary_domain_id": primary_domain_id,
    }


# -----------------------------------------------------------------------------
# 6. CSV Helpers
# -----------------------------------------------------------------------------

def source_to_slug(url: str) -> str:
    """
    Convert a source URL to a compact tag-friendly value.
    """
    if not url:
        return ""

    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")

    return path.replace("/", "-") if path else parsed_url.netloc


def build_csv_row(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert an accepted evaluated question into the desired CSV format.
    """
    if "error" in q:
        return None

    evaluation = q.get("_evaluation", {})

    if evaluation.get("decision") != "accept":
        return None

    options = q.get("options", {})
    option_explanations = q.get("option_explanations", {})

    if not isinstance(options, dict):
        return None

    if not isinstance(option_explanations, dict):
        return None

    raw_correct = q.get("correct_answers") or q.get("correct_answer")

    if isinstance(raw_correct, list):
        correct_keys = {
            key.strip().upper()
            for key in raw_correct
            if isinstance(key, str)
        }
    elif isinstance(raw_correct, str):
        correct_keys = {raw_correct.strip().upper()}
    else:
        return None

    option_keys = list(options.keys())

    if not option_keys:
        return None

    is_multi = len(correct_keys) > 1

    answers_str = " ".join(
        "1" if key.upper() in correct_keys else "0"
        for key in option_keys
    )

    q_cols = {
        f"Q_{index + 1}": ""
        for index in range(MAX_OPTIONS)
    }

    e_cols = {
        f"E_{index + 1}": ""
        for index in range(MAX_OPTIONS)
    }

    for index, key in enumerate(option_keys[:MAX_OPTIONS]):
        q_cols[f"Q_{index + 1}"] = options.get(key, "")
        e_cols[f"E_{index + 1}"] = option_explanations.get(key, "")

    tags = ["claude-cert"]

    if is_multi:
        tags.append("multiple-answer")

    primary_domain_id = q.get("_primary_domain_id", "")
    if primary_domain_id:
        tags.append(f"domain:{primary_domain_id}")

    primary_source = q.get("_primary_source", "")
    if primary_source:
        tags.append(f"source:{source_to_slug(primary_source)}")

    score = evaluation.get("overall_score")
    if isinstance(score, (int, float)):
        tags.append(f"eval:{score:.2f}")

    return {
        "Question": q.get("question", ""),
        "QType": 1 if is_multi else 2,
        "Answers": answers_str,
        **q_cols,
        **e_cols,
        "Explanation": q.get("explanation", ""),
        "Tags": " ".join(tags),
        "Source": primary_source,
        "EvaluationScore": score if score is not None else "",
        "EvaluationDecision": evaluation.get("decision", ""),
    }


def get_csv_fieldnames() -> List[str]:
    return (
        ["Question", "QType", "Answers"]
        + [f"Q_{index + 1}" for index in range(MAX_OPTIONS)]
        + [f"E_{index + 1}" for index in range(MAX_OPTIONS)]
        + [
            "Explanation",
            "Tags",
            "Source",
            "EvaluationScore",
            "EvaluationDecision",
        ]
    )


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------

def main() -> None:
    print(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")

    try:
        vectorstore = get_vectorstore()
    except Exception as exc:
        print(f"Index loading failed: {exc}")
        sys.exit(1)

    print("Building task→document index (one-time scan)...")

    task_doc_index = build_task_doc_index(vectorstore)

    print(
        f"  Indexed {sum(len(value) for value in task_doc_index.values())} "
        f"doc-task mappings across {len(task_doc_index)} task buckets.\n"
    )

    # ── Build blueprint ───────────────────────────────────────────────────────
    active_scenarios = select_exam_scenarios(n=4)

    domain_counts = compute_domain_question_counts(
        EXAM_SIZE,
        DOMAIN_WEIGHTS,
    )

    task_plan = build_task_question_plan(domain_counts)

    intended_total = sum(task_plan.values())

    print("=" * 60)
    print(
        f"  EXAM BLUEPRINT  ({intended_total} intended questions, "
        f"{MAX_WORKERS} workers)"
    )
    print("=" * 60)

    print("\nActive Scenarios:")
    for name in active_scenarios:
        print(f"  • {name}")

    print("\nDomain Question Counts:")
    for domain_id, count in domain_counts.items():
        print(
            f"  {TAXONOMY[domain_id]['domain']}: "
            f"{count} ({count / EXAM_SIZE * 100:.1f}%)"
        )

    print("=" * 60)
    print()

    # ── Build work items ──────────────────────────────────────────────────────
    work_items: List[Dict[str, Any]] = []
    question_index = 0

    for domain_id, domain_info in TAXONOMY.items():
        for task_id, task_details in domain_info["tasks"].items():
            question_count = task_plan.get(task_id, 0)

            rag_context = get_context_for_task(
                task_doc_index=task_doc_index,
                task_id=task_id,
            )

            if REQUIRE_RAG_CONTEXT and not rag_context.strip():
                print(
                    f"  [SKIP] {task_id}: no indexed documentation context "
                    f"available for grounded generation."
                )
                continue

            primary_source, primary_domain_id = get_top_doc_meta(
                task_doc_index=task_doc_index,
                task_id=task_id,
                domain_id=domain_id,
            )

            for _ in range(question_count):
                question_index += 1

                scenario_name, scenario_desc = get_scenario_for_task(
                    domain_id=domain_id,
                    active_scenarios=active_scenarios,
                )

                work_items.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_desc": scenario_desc,
                        "task_id": task_id,
                        "task_statement": task_details["statement"],
                        "task_desc": task_details["description"],
                        "rag_context": rag_context,
                        "question_index": question_index,
                        # Updated after all skipped tasks are known.
                        "total_questions": 0,
                        "primary_source": primary_source,
                        "primary_domain_id": primary_domain_id,
                    }
                )

    total_planned = len(work_items)

    if total_planned == 0:
        print(
            "No questions can be generated because no usable RAG context "
            "was found for any task."
        )
        return

    for item in work_items:
        item["total_questions"] = total_planned

    print(
        f"Generating {total_planned} question candidates after "
        "documentation-coverage filtering...\n"
    )

    # ── Parallel generation ───────────────────────────────────────────────────
    question_bank: List[Optional[Dict[str, Any]]] = [
        None
    ] * len(work_items)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(generate_question_with_retry, **item): index
            for index, item in enumerate(work_items)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]

            try:
                question_bank[index] = future.result()

            except Exception as exc:
                item = work_items[index]

                print(f"  [FATAL] Task {item['task_id']}: {exc}")

                question_bank[index] = {
                    "task_id": item["task_id"],
                    "scenario": item["scenario_name"],
                    "error": str(exc),
                    "_primary_source": item.get("primary_source", ""),
                    "_primary_domain_id": item.get(
                        "primary_domain_id",
                        "",
                    ),
                }

    # ── Evaluation / optional revision ────────────────────────────────────────
    print("\nEvaluating generated questions...")

    for index, question in enumerate(question_bank):
        if question is None or "error" in question:
            continue

        item = work_items[index]

        # Preserve the exact evidence and objective used for audit/review.
        question["_task_statement"] = item["task_statement"]
        question["_task_desc"] = item["task_desc"]
        question["_rag_context"] = item["rag_context"]

        evaluated_question = evaluate_and_optionally_revise(
            question=question,
            task_statement=item["task_statement"],
            task_desc=item["task_desc"],
            rag_context=item["rag_context"],
            evaluator_model=LLM_MODEL,
            revision_model=LLM_MODEL,
            auto_revise=AUTO_REVISE,
            max_revision_attempts=MAX_REVISION_ATTEMPTS,
        )

        question_bank[index] = evaluated_question

        evaluation = evaluated_question.get("_evaluation", {})

        print(
            f"  [{evaluated_question.get('task_id', 'unknown')}] "
            f"{evaluation.get('decision', 'reject').upper()} "
            f"({evaluation.get('overall_score', 0.0):.2f})"
        )

    results: List[Dict[str, Any]] = [
        question
        for question in question_bank
        if question is not None
    ]

    # ── Output ────────────────────────────────────────────────────────────────
    json_output = "claude_exam_questions.json"
    csv_output = "claude_exam_questions.csv"

    with open(json_output, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved JSON → '{json_output}'")

    rows_written = 0

    with open(
        csv_output,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=get_csv_fieldnames(),
            quoting=csv.QUOTE_ALL,
        )

        writer.writeheader()

        for question in results:
            row = build_csv_row(question)

            if row:
                writer.writerow(row)
                rows_written += 1

    print(f"Saved CSV  → '{csv_output}' ({rows_written} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    errors = sum(
        1
        for question in results
        if "error" in question
    )

    accepted = sum(
        1
        for question in results
        if question.get("_evaluation", {}).get("decision") == "accept"
    )

    rejected = sum(
        1
        for question in results
        if question.get("_evaluation", {}).get("decision") == "reject"
    )

    revise_remaining = sum(
        1
        for question in results
        if question.get("_evaluation", {}).get("decision") == "revise"
    )

    print(
        f"\nTotal: {len(results)} generated | "
        f"{accepted} accepted | "
        f"{revise_remaining} needs revision | "
        f"{rejected} rejected | "
        f"{errors} generation errors | "
        f"{rows_written} exported"
    )


if __name__ == "__main__":
    main()