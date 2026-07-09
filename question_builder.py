import csv
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import ollama

from taxonomy import TAXONOMY, TASK_TO_CUSTOM_CATEGORY

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
FAISS_INDEX_PATH = "./faiss_claude_docs"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_API_BASE = "http://127.0.0.1:11434"

# Recommended: drop to 14b for ~2x speedup with minimal quality loss.
# If you have a GPU that fits 32b comfortably, keep it.
LLM_MODEL = "qwen2.5:14b"

# How many questions to generate concurrently.
# - CPU-only (Ollama running on CPU): keep at 1-2
# - Single GPU: 2-4 (Ollama queues requests internally)
# - Multiple GPUs / large VRAM: 4-8
MAX_WORKERS = 4

# Retry a failed generation this many times before giving up
MAX_RETRIES = 2

MAX_OPTIONS = 12
TOP_K_DOCS = 4
MIN_CONFIDENCE = 0.5
EXAM_SIZE = 60
MULTI_ANSWER_RATE = 0

DOMAIN_WEIGHTS: Dict[str, float] = {
    "1": 0.27,
    "2": 0.18,
    "3": 0.20,
    "4": 0.20,
    "5": 0.15,
}

SCENARIOS: Dict[str, Dict] = {
    "Scenario 1: Customer Support Resolution Agent": {
        "description": (
            "You are building a customer support resolution agent using the Claude Agent SDK. "
            "The agent handles high-ambiguity requests like returns, billing disputes, and "
            "account issues. It has access to your backend systems through custom Model Context "
            "Protocol (MCP) tools (get_customer, lookup_order, process_refund, escalate_to_human). "
            "Your target is 80%+ first-contact resolution while knowing when to escalate."
        ),
        "primary_domains": ["1", "2", "5"],
    },
    "Scenario 2: Code Generation with Claude Code": {
        "description": (
            "You are using Claude Code to accelerate software development. Your team uses it for "
            "code generation, refactoring, debugging, and documentation. You need to integrate "
            "it into your development workflow with custom slash commands, CLAUDE.md "
            "configurations, and understand when to use plan mode vs direct execution."
        ),
        "primary_domains": ["3", "5"],
    },
    "Scenario 3: Multi-Agent Research System": {
        "description": (
            "You are building a multi-agent research system using the Claude Agent SDK. "
            "A coordinator agent delegates to specialized subagents: one searches the web, "
            "one analyzes documents, one synthesizes findings, and one generates reports. "
            "The system researches topics and produces comprehensive, cited reports."
        ),
        "primary_domains": ["1", "2", "5"],
    },
    "Scenario 4: Developer Productivity with Claude": {
        "description": (
            "You are building developer productivity tools using the Claude Agent SDK. "
            "The agent helps engineers explore unfamiliar codebases, understand legacy systems, "
            "generate boilerplate code, and automate repetitive tasks. It uses the built-in tools "
            "(Read, Write, Bash, Grep, Glob) and integrates with Model Context Protocol (MCP) servers."
        ),
        "primary_domains": ["1", "2", "3"],
    },
    "Scenario 5: Claude Code for Continuous Integration": {
        "description": (
            "You are integrating Claude Code into your Continuous Integration/Continuous "
            "Deployment (CI/CD) pipeline. The system runs automated code reviews, generates "
            "test cases, and provides feedback on pull requests. You need to design prompts "
            "that provide actionable feedback and minimize false positives."
        ),
        "primary_domains": ["3", "4"],
    },
    "Scenario 6: Structured Data Extraction": {
        "description": (
            "You are building a structured data extraction system using Claude. The system "
            "extracts information from unstructured documents, validates the output using "
            "JavaScript Object Notation (JSON) schemas, and maintains high accuracy. It "
            "must handle edge cases gracefully and integrate with downstream systems."
        ),
        "primary_domains": ["4", "5"],
    },
}


# -----------------------------------------------------------------------------
# 2. Exam Blueprint
# -----------------------------------------------------------------------------
def select_exam_scenarios(n: int = 4) -> Dict[str, str]:
    scenario_keys = list(SCENARIOS.keys())
    for _ in range(200):
        chosen = random.sample(scenario_keys, n)
        covered: set = set()
        for k in chosen:
            covered.update(SCENARIOS[k]["primary_domains"])
        if covered >= set(DOMAIN_WEIGHTS.keys()):
            return {k: SCENARIOS[k]["description"] for k in chosen}

    # Guaranteed fallback
    chosen = random.sample(scenario_keys, n)
    covered = set()
    for k in chosen:
        covered.update(SCENARIOS[k]["primary_domains"])
    for domain_id in set(DOMAIN_WEIGHTS.keys()) - covered:
        candidates = [
            k
            for k in scenario_keys
            if domain_id in SCENARIOS[k]["primary_domains"] and k not in chosen
        ]
        if candidates:
            chosen[random.randrange(len(chosen))] = random.choice(candidates)
    return {k: SCENARIOS[k]["description"] for k in chosen}


def compute_domain_question_counts(
    exam_size: int,
    weights: Dict[str, float],
) -> Dict[str, int]:
    exact = {d: w * exam_size for d, w in weights.items()}
    floors = {d: math.floor(v) for d, v in exact.items()}
    remainder = exam_size - sum(floors.values())
    by_fraction = sorted(exact, key=lambda d: exact[d] - floors[d], reverse=True)
    counts = dict(floors)
    for i in range(remainder):
        counts[by_fraction[i]] += 1
    return counts


def build_task_question_plan(domain_counts: Dict[str, int]) -> Dict[str, int]:
    plan: Dict[str, int] = {}
    for domain_id, total_qs in domain_counts.items():
        tasks = list(TAXONOMY[domain_id]["tasks"].keys())
        base, extra = divmod(total_qs, len(tasks))
        for i, task_id in enumerate(tasks):
            plan[task_id] = base + (1 if i < extra else 0)
    return plan


# -----------------------------------------------------------------------------
# 3. FAISS / Metadata Index
# -----------------------------------------------------------------------------
def get_vectorstore() -> FAISS:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'."
        )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_API_BASE)
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_task_doc_index(vectorstore: FAISS) -> Dict[str, List[Dict]]:
    index: Dict[str, List[Dict]] = {}
    for _doc_id, doc in vectorstore.docstore._dict.items():
        meta = doc.metadata
        if meta.get("classification_confidence", 1.0) < MIN_CONFIDENCE:
            continue
        task_ids: set = set()
        if primary := meta.get("primary_task_id"):
            task_ids.add(str(primary))
        for sec in meta.get("secondary_task_ids") or []:
            task_ids.add(str(sec))
        entry = {
            "content": doc.page_content,
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "confidence": meta.get("classification_confidence", 1.0),
            "primary_domain_id": meta.get("primary_domain_id", ""),  # NEW
        }
        for tid in task_ids:
            index.setdefault(tid, []).append(entry)
    for tid in index:
        index[tid].sort(key=lambda d: d["confidence"], reverse=True)
    return index


def get_context_for_task(
    task_doc_index: Dict[str, List[Dict]],
    task_id: str,
    k: int = TOP_K_DOCS,
) -> str:
    docs = task_doc_index.get(task_id, [])[:k]
    if not docs:
        return "(No indexed documentation found for this task.)"
    return "\n\n---\n\n".join(
        f"Source: {d['source']}\nDocument Section: {d['title']}\nContent:\n{d['content']}"
        for d in docs
    )


def get_top_doc_meta(
    task_doc_index: Dict[str, List[Dict]],
    task_id: str,
    domain_id: str,
) -> Tuple[str, str]:
    """Return (primary_source, primary_domain_id) from the highest-confidence doc."""
    docs = task_doc_index.get(task_id, [])
    if docs:
        return docs[0]["source"], docs[0]["primary_domain_id"] or domain_id
    return "", domain_id


def get_scenario_for_task(
    domain_id: str,
    active_scenarios: Dict[str, str],
) -> Tuple[str, str]:
    eligible = [
        (name, desc)
        for name, desc in active_scenarios.items()
        if domain_id in SCENARIOS[name]["primary_domains"]
    ]
    if eligible:
        return random.choice(eligible)
    fallback = random.choice(list(active_scenarios.keys()))
    return fallback, active_scenarios[fallback]


# -----------------------------------------------------------------------------
# 4. LLM Generation  (with retry + structured JSON mode)
# -----------------------------------------------------------------------------
def clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def build_prompt(
    scenario_name: str,
    scenario_desc: str,
    task_id: str,
    task_statement: str,
    task_desc: str,
    rag_context: str,
    is_multi: bool,
) -> str:
    focus_areas = [
        "A production failure, edge case, or performance bug discovered in logs",
        "An architectural tradeoff (e.g., programmatic enforcement vs. system prompts)",
        "Pipeline integration syntax, flags (e.g., -p, --print), or config files (e.g., .claude/rules/)",
        "Unexpected developer behaviors and how to mitigate them deterministically",
    ]
    selected_focus = random.choice(focus_areas)

    if is_multi:
        type_instruction = (
            "Generate a MULTIPLE-ANSWER question (exactly 2 correct answers out of 4). "
            "The question text MUST be prefixed with '[SELECT TWO] ...' and ask for exactly 2 correct choices. "
            'Provide a JSON list of exactly 2 keys in "correct_answers" (e.g., ["A", "C"]).'
        )
        answer_field = '"correct_answers": ["A", "C"]'
    else:
        type_instruction = (
            "Generate a SINGLE-ANSWER question (exactly 1 correct answer out of 4). "
            'Set "correct_answer" to exactly one option key string (e.g., "B").'
        )
        answer_field = '"correct_answer": "B"'

    return f"""You are an elite exam developer writing high-fidelity questions for the Anthropic Claude Developer Credentials certification.
Your goal is to write a highly specific, situational question modeled on actual engineering dilemmas. Refer to the style in the official study guide.

[EXAM SCENARIO CONTEXT]
Scenario Name: {scenario_name}
Environment Overview: {scenario_desc}

[EXAM BLUEPRINT OBJECTIVE]
Task ID: {task_id}
Standard Objective: {task_statement}
Expected Core Knowledge: {task_desc}

[SPECIFIC QUESTION FOCUS]
To prevent duplicate questions in this domain, dedicate this particular question to testing:
-> {selected_focus}

[DOCUMENTATION REFERENCE (RAG)]
{rag_context}

[QUESTION FORMAT CRITERIA]
{type_instruction}

[STRICT QUESTION WRITING RULES]
1. Avoid high-level, generic definitions (e.g., do NOT ask 'What does stop_reason: tool_use indicate?').
2. Instead, craft a concrete production scenario. Start with a realistic engineering problem (e.g., 'Your pipeline script hangs indefinitely in CI...', 'Production logs show that in 12% of cases, your agent skips tools...', or 'Your codebase has distinct subprojects in separate directories...').
3. Distractors (incorrect options) must reflect highly plausible anti-patterns, probabilistic prompt hacks (that fail to offer deterministic guarantees), or obsolete/non-existent SDK parameters.
4. Ensure that the correct answer is directly verifiable using the provided Documentation Context.

[JSON SCHEMAS]
Output ONLY a raw, parser-compliant JSON object matching this structure:
{{
  "task_id": "{task_id}",
  "scenario": "{scenario_name}",
  "question": "A situational question starting with a system setup or problem... Use code snippets or exact syntax where possible.",
  "options": {{
    "A": "Plausible but incorrect option / anti-pattern",
    "B": "Correct engineering solution based strictly on docs",
    "C": "Alternative incorrect option / over-engineered approach",
    "D": "Incorrect option utilizing non-existent SDK features"
  }},
  {answer_field},
  "option_explanations": {{
    "A": "Explanation of why this option fails, identifying its class of error (e.g., 'This relies on probabilistic LLM compliance instead of deterministic guarantees').",
    "B": "Explanation validating why this option is correct.",
    "C": "Explanation detailing why this option is suboptimal or incorrect.",
    "D": "Explanation identifying non-existent features or syntax errors in this choice."
  }},
  "explanation": "A complete, comprehensive architectural explanation of the solution based on the documentation, explicitly justifying the correct answer(s) over the distractors."
}}"""


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
    Attempts generation up to MAX_RETRIES times.
    On a JSON parse failure the error is fed back into the next attempt
    so the model can self-correct.
    """
    is_multi = random.random() < MULTI_ANSWER_RATE
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = build_prompt(
            scenario_name,
            scenario_desc,
            task_id,
            task_statement,
            task_desc,
            rag_context,
            is_multi,
        )

        if last_error:
            prompt += (
                f"\n\n[PREVIOUS ATTEMPT FAILED]\n"
                f"Your last response could not be parsed as JSON.\n"
                f"Error: {last_error}\n"
                f"Return ONLY valid JSON. No extra text."
            )

        raw_response: Optional[str] = None
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

            # Attach source metadata so it flows through to CSV
            parsed["_primary_source"] = primary_source
            parsed["_primary_domain_id"] = primary_domain_id

            print(
                f"  ✓ [{task_id}] Q{question_index}/{total_questions}"
                + (f" (attempt {attempt})" if attempt > 1 else "")
            )
            return parsed

        except json.JSONDecodeError as exc:
            last_error = str(exc)
            print(
                f"  ✗ [{task_id}] Q{question_index}/{total_questions} "
                f"JSON error attempt {attempt}/{MAX_RETRIES}: {exc}"
            )
        except Exception as exc:
            print(
                f"  ✗ [{task_id}] Q{question_index}/{total_questions} "
                f"Unexpected error attempt {attempt}/{MAX_RETRIES}: {exc}"
            )
            last_error = str(exc)

    return {
        "task_id": task_id,
        "scenario": scenario_name,
        "error": f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}",
        "raw_response": raw_response,
        "_primary_source": primary_source,
        "_primary_domain_id": primary_domain_id,
    }


# -----------------------------------------------------------------------------
# 5. CSV Helpers
# -----------------------------------------------------------------------------
def source_to_slug(url: str) -> str:
    """Convert a URL to a compact tag-friendly slug, e.g. 'docs-claude-code-setup'."""
    if not url:
        return ""
    path = urlparse(url).path.strip("/")
    return path.replace("/", "-") if path else urlparse(url).netloc


def build_csv_row(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "error" in q:
        return None
    options: Dict[str, str] = q.get("options", {})
    option_explanations: Dict[str, str] = q.get("option_explanations", {})
    raw_correct = q.get("correct_answers") or q.get("correct_answer")
    if isinstance(raw_correct, list):
        correct_keys = {k.strip().upper() for k in raw_correct}
    elif isinstance(raw_correct, str):
        correct_keys = {raw_correct.strip().upper()}
    else:
        return None

    option_keys = list(options.keys())
    is_multi = len(correct_keys) > 1
    answers_str = " ".join(
        "1" if option_keys[i].upper() in correct_keys else "0"
        for i in range(len(option_keys))
    )

    q_cols = {f"Q_{i+1}": "" for i in range(MAX_OPTIONS)}
    e_cols = {f"E_{i+1}": "" for i in range(MAX_OPTIONS)}
    for i, key in enumerate(option_keys[:MAX_OPTIONS]):
        q_cols[f"Q_{i+1}"] = options.get(key, "")
        e_cols[f"E_{i+1}"] = option_explanations.get(key, "")

    # ── Build tags ────────────────────────────────────────────────────────────
    tags = ["claude-cert"]
    if is_multi:
        tags.append("multiple-answer")
    if domain_id := q.get("_primary_domain_id"):
        tags.append(f"domain:{domain_id}")
    if source := q.get("_primary_source"):
        tags.append(f"source:{source_to_slug(source)}")

    return {
        "Question": q.get("question", ""),
        "QType": 1 if is_multi else 2,
        "Answers": answers_str,
        **q_cols,
        **e_cols,
        "Explanation": q.get("explanation", ""),
        "Tags": " ".join(tags),
        "Source": q.get("_primary_source", ""),  # NEW: full URL
    }


def get_csv_fieldnames() -> List[str]:
    return (
        ["Question", "QType", "Answers"]
        + [f"Q_{i+1}" for i in range(MAX_OPTIONS)]
        + [f"E_{i+1}" for i in range(MAX_OPTIONS)]
        + ["Explanation", "Tags", "Source"]  # NEW: Source appended
    )


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        print(f"Index loading failed: {e}")
        sys.exit(1)

    print("Building task→document index (one-time scan)...")
    task_doc_index = build_task_doc_index(vectorstore)
    print(
        f"  Indexed {sum(len(v) for v in task_doc_index.values())} "
        f"doc-task mappings across {len(task_doc_index)} task buckets.\n"
    )

    # ── Blueprint ─────────────────────────────────────────────────────────────
    active_scenarios = select_exam_scenarios(n=4)
    domain_counts = compute_domain_question_counts(EXAM_SIZE, DOMAIN_WEIGHTS)
    task_plan = build_task_question_plan(domain_counts)
    total_planned = sum(task_plan.values())

    print("=" * 60)
    print(f"  EXAM BLUEPRINT  ({total_planned} questions, {MAX_WORKERS} workers)")
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
    print("=" * 60, "\n")

    # ── Build work items ──────────────────────────────────────────────────────
    work_items = []
    q_index = 0
    for domain_id, domain_info in TAXONOMY.items():
        for task_id, task_details in domain_info["tasks"].items():
            n_questions = task_plan.get(task_id, 0)
            primary_source, primary_domain_id = get_top_doc_meta(
                task_doc_index, task_id, domain_id
            )
            for _ in range(n_questions):
                q_index += 1
                scenario_name, scenario_desc = get_scenario_for_task(
                    domain_id, active_scenarios
                )
                work_items.append(
                    dict(
                        scenario_name=scenario_name,
                        scenario_desc=scenario_desc,
                        task_id=task_id,
                        task_statement=task_details["statement"],
                        task_desc=task_details["description"],
                        rag_context=get_context_for_task(
                            task_doc_index, task_id
                        ),
                        question_index=q_index,
                        total_questions=total_planned,
                        primary_source=primary_source,
                        primary_domain_id=primary_domain_id,
                    )
                )

    # ── Parallel generation ───────────────────────────────────────────────────
    question_bank: List[Optional[Dict[str, Any]]] = [None] * len(work_items)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(generate_question_with_retry, **item): idx
            for idx, item in enumerate(work_items)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                question_bank[idx] = future.result()
            except Exception as exc:
                item = work_items[idx]
                print(f"  [FATAL] Task {item['task_id']}: {exc}")
                question_bank[idx] = {
                    "task_id": item["task_id"],
                    "scenario": item["scenario_name"],
                    "error": str(exc),
                    "_primary_source": item.get("primary_source", ""),
                    "_primary_domain_id": item.get("primary_domain_id", ""),
                }

    results: List[Dict[str, Any]] = [q for q in question_bank if q is not None]

    # ── Output ────────────────────────────────────────────────────────────────
    json_output = "claude_exam_questions.json"
    csv_output = "claude_exam_questions.csv"

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON → '{json_output}'")

    rows_written = 0
    with open(csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=get_csv_fieldnames(), quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for q in results:
            row = build_csv_row(q)
            if row:
                writer.writerow(row)
                rows_written += 1

    print(f"Saved CSV  → '{csv_output}' ({rows_written} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    errors = sum(1 for q in results if "error" in q)
    print(
        f"\nTotal: {len(results)} generated | "
        f"{rows_written} exported | {errors} errors"
    )


if __name__ == "__main__":
    main()