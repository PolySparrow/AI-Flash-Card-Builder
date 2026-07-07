import json
import os
import time
import csv
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from taxonomy import TASK_TO_CUSTOM_CATEGORY, TAXONOMY


BASE_URL = "https://platform.claude.com/docs/en/"
OUTPUT_DIR = "./faiss_claude_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:32b"

MAX_PAGES = 768
REQUEST_DELAY_SECONDS = 0.5
TIMEOUT_SECONDS = 20

def write_classifications_to_csv(
    documents: list[Document],
    csv_path: str = "page_classifications.csv",
):
    fieldnames = [
        "source",
        "title",
        "url_tags",
        "human_tags",
        "primary_domain_id",
        "primary_domain",
        "primary_task_id",
        "primary_task_statement",
        "secondary_task_ids",
        "classification_confidence",
        "classification_reason",
        "custom_category",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for doc in documents:
            metadata = doc.metadata
            writer.writerow(
                {
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "url_tags": "|".join(metadata.get("url_tags", [])),
                    "human_tags": "|".join(metadata.get("human_tags", [])),
                    "primary_domain_id": metadata.get(
                        "primary_domain_id",
                        "",
                    ),
                    "primary_domain": metadata.get("primary_domain", ""),
                    "primary_task_id": metadata.get("primary_task_id", ""),
                    "primary_task_statement": metadata.get(
                        "primary_task_statement",
                        "",
                    ),
                    "secondary_task_ids": "|".join(
                        metadata.get("secondary_task_ids", [])
                    ),
                    "classification_confidence": metadata.get(
                        "classification_confidence",
                        "",
                    ),
                    "classification_reason": metadata.get(
                        "classification_reason",
                        "",
                    ),
                    "custom_category": metadata.get("custom_category", ""),
                }
            )

    print(f"Classification CSV written to: {csv_path}")

def normalize_url(url: str) -> str:
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"

    normalized = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path,
        params="",
        query="",
        fragment="",
    )
    return normalized.geturl()


def is_allowed_url(url: str, base_url: str) -> bool:
    return normalize_url(url).startswith(normalize_url(base_url))


def extract_links(html: str, current_url: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        absolute = urljoin(current_url, href)
        normalized = normalize_url(absolute)

        if is_allowed_url(normalized, base_url):
            found.add(normalized)

    return sorted(found)


def fetch_html(session: requests.Session, url: str) -> str | None:
    try:
        response = session.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None
        return response.text
    except Exception as exc:
        print(f"Failed to fetch {url}: {exc}")
        return None


def extract_main_text(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_formatting=False,
        url=url,
    )
    if extracted and extracted.strip():
        return extracted.strip()

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text.strip()


def get_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def extract_tags_from_url(url: str) -> list[str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    ignored = {"docs", "en"}
    return [part for part in parts if part not in ignored]


def extract_human_tags_from_url(url: str) -> list[str]:
    return [tag.replace("-", " ").strip() for tag in extract_tags_from_url(url)]


def format_taxonomy_for_prompt() -> str:
    lines = []
    for domain_id, domain_data in TAXONOMY.items():
        lines.append(domain_data["domain"])
        for task_id, task_data in domain_data["tasks"].items():
            lines.append(f"  {task_id}: {task_data['statement']}")
            lines.append(f"    Description: {task_data['description']}")
            lines.append(
                f"    Keywords: {', '.join(task_data['keywords'][:6])}"
            )
    return "\n".join(lines)


def guess_domain_from_text(text: str) -> str | None:
    text = text.lower()

    domain_keywords = {
        "1": [
            "agent",
            "subagent",
            "task tool",
            "orchestration",
            "tool_use",
            "end_turn",
            "fork_session",
            "coordinator",
        ],
        "2": [
            "mcp",
            "tool_choice",
            "grep",
            "glob",
            "read",
            "write",
            "edit",
            "bash",
            "server",
            "tool description",
        ],
        "3": [
            "claude.md",
            "slash command",
            "skill",
            "plan mode",
            "ci",
            "cd",
            "workflow",
            ".claude",
        ],
        "4": [
            "few-shot",
            "json schema",
            "structured output",
            "prompt",
            "batch",
            "tool use",
            "validation",
            "retry",
        ],
        "5": [
            "context",
            "provenance",
            "ambiguity",
            "escalation",
            "confidence",
            "summarization",
            "uncertainty",
            "reliability",
        ],
    }

    scores = {}
    for domain_id, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[domain_id] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


def safe_json_loads(text: str) -> dict:
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")

    text = text[start : end + 1]

    # Try standard parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to fix trailing commas (common llama3.1 failure)
    import re
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return json.loads(text)

def classify_page_forced(url: str, title: str, content: str) -> dict:
    """Simpler prompt that just asks for a task ID number, no JSON."""
    llm = ChatOllama(model=LLM_MODEL, base_url="http://127.0.0.1:11434", temperature=0)

    valid_task_ids = [
        task_id
        for domain_data in TAXONOMY.values()
        for task_id in domain_data["tasks"]
    ]
    options = "\n".join(
        f"{tid}: {TAXONOMY[tid.split('.')[0]]['tasks'][tid]}"
        for tid in valid_task_ids
    )
    excerpt = build_classification_input(url, title, content)

    prompt = f"""Pick the single best task ID for this page. Reply with ONLY the task ID (e.g. "2.4"), nothing else.

OPTIONS:
{options}

PAGE:
{excerpt[:1500]}

TASK ID:"""

    try:
        response = llm.invoke(prompt)
        task_id = response.content.strip().strip('"').strip()
        domain_id = task_id.split(".")[0]

        if domain_id not in TAXONOMY or task_id not in TAXONOMY[domain_id]["tasks"]:
            raise ValueError(f"Invalid task ID: {task_id!r}")

        return {
            "primary_domain_id": domain_id,
            "primary_domain": TAXONOMY[domain_id]["domain"],
            "primary_task_id": task_id,
            "primary_task_statement": TAXONOMY[domain_id]["tasks"][task_id]["statement"],
            "secondary_task_ids": [],
            "confidence": 0.5,
            "reason": "Forced-choice classification.",
            "custom_category": TASK_TO_CUSTOM_CATEGORY.get(task_id, "uncategorized"),
        }
    except Exception as exc:
        return classify_page(url, title, content)  # fall back to original
    
def classify_page(url: str, title: str, content: str) -> dict:
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0,
    )

    excerpt = build_classification_input(url, title, content)
    # Don't filter — show the full taxonomy so the model can always find a valid ID
    taxonomy_text = format_taxonomy_for_prompt()

    # Build a flat list of valid task IDs for the model to reference
    valid_task_ids = []
    for domain_data in TAXONOMY.values():
        valid_task_ids.extend(domain_data["tasks"].keys())
    valid_ids_str = ", ".join(valid_task_ids)

    prompt = f"""You are classifying a documentation page into a taxonomy.

VALID TASK IDs (you MUST use one of these exactly): {valid_ids_str}

TAXONOMY:
{taxonomy_text}

INSTRUCTIONS:
- Read the page content below and pick the single best matching task ID from the list above.
- "primary_domain_id" must be the digit before the dot in your chosen task ID (e.g. task "2.4" → domain_id "2").
- "primary_domain" must be the exact domain name from the taxonomy.
- "primary_task_statement" must be the exact statement from the taxonomy.
- "secondary_task_ids" should be a JSON array of 0-2 other relevant task IDs, or [].
- "confidence" is a float from 0.0 to 1.0.
- "reason" is one sentence explaining your choice.
- Output ONLY valid JSON. No explanation. No markdown fences. No extra text.

EXAMPLE OUTPUT:
{{"primary_domain_id":"2","primary_domain":"Domain2: Tool Design & MCP Integration","primary_task_id":"2.4","primary_task_statement":"Integrate MCP servers into Claude Code and agent workflows","secondary_task_ids":["2.1"],"confidence":0.85,"reason":"Page describes connecting remote MCP servers to agent workflows."}}

PAGE URL: {url}
PAGE TITLE: {title}
PAGE CONTENT:
{excerpt}

JSON:"""

    try:
        response = llm.invoke(prompt)
        parsed = safe_json_loads(response.content)

        primary_task_id = parsed.get("primary_task_id", "").strip()
        primary_domain_id = parsed.get("primary_domain_id", "").strip()

        # Auto-correct domain_id from task_id if the model got it wrong
        if primary_task_id and "." in primary_task_id:
            derived_domain_id = primary_task_id.split(".")[0]
            if derived_domain_id in TAXONOMY:
                primary_domain_id = derived_domain_id
                parsed["primary_domain_id"] = primary_domain_id
                parsed["primary_domain"] = TAXONOMY[primary_domain_id]["domain"]

        if (
            primary_domain_id not in TAXONOMY
            or primary_task_id
            not in TAXONOMY.get(primary_domain_id, {}).get("tasks", {})
        ):
            raise ValueError(
                f"Invalid taxonomy mapping: domain={primary_domain_id!r}, "
                f"task={primary_task_id!r}"
            )

        # Normalize task statement to ground truth
        parsed["primary_task_statement"] = TAXONOMY[primary_domain_id][
            "tasks"
        ][primary_task_id]
        parsed["primary_domain"] = TAXONOMY[primary_domain_id]["domain"]

        parsed["custom_category"] = TASK_TO_CUSTOM_CATEGORY.get(
            primary_task_id, "uncategorized"
        )

        return parsed

    except Exception as exc:
        guessed_domain = guess_domain_from_text(
            f"{url}\n{title}\n{content[:2000]}"
        )
        fallback_domain_id = guessed_domain or "5"
        fallback_domain = TAXONOMY[fallback_domain_id]["domain"]
        fallback_task_id = next(iter(TAXONOMY[fallback_domain_id]["tasks"]))
        fallback_task_statement = TAXONOMY[fallback_domain_id]["tasks"][
            fallback_task_id
        ]

        return {
            "primary_domain_id": fallback_domain_id,
            "primary_domain": fallback_domain,
            "primary_task_id": fallback_task_id,
            "primary_task_statement": fallback_task_statement,
            "secondary_task_ids": [],
            "confidence": 0.2,
            "reason": f"Fallback classification due to error: {exc}",
            "custom_category": TASK_TO_CUSTOM_CATEGORY.get(
                fallback_task_id, "uncategorized"
            ),
        }


def enrich_document_for_embedding(doc: Document) -> Document:
    metadata = dict(doc.metadata)
    prefix_lines = []

    if metadata.get("title"):
        prefix_lines.append(f"Title: {metadata['title']}")

    if metadata.get("url_tags"):
        prefix_lines.append(f"URL Tags: {', '.join(metadata['url_tags'])}")

    if metadata.get("human_tags"):
        prefix_lines.append(f"Human Tags: {', '.join(metadata['human_tags'])}")

    if metadata.get("primary_domain"):
        prefix_lines.append(f"Domain: {metadata['primary_domain']}")

    if metadata.get("primary_task_id") and metadata.get(
        "primary_task_statement"
    ):
        prefix_lines.append(
            "Primary Task: "
            f"{metadata['primary_task_id']} - "
            f"{metadata['primary_task_statement']}"
        )

    if metadata.get("secondary_task_ids"):
        prefix_lines.append(
            f"Secondary Tasks: {', '.join(metadata['secondary_task_ids'])}"
        )

    if metadata.get("custom_category"):
        prefix_lines.append(
            f"Custom Category: {metadata['custom_category']}"
        )

    enriched_text = "\n".join(prefix_lines).strip()
    if enriched_text:
        enriched_text = f"{enriched_text}\n\n{doc.page_content}"
    else:
        enriched_text = doc.page_content

    return Document(
        page_content=enriched_text,
        metadata=metadata,
    )


def test_ollama_embeddings():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://127.0.0.1:11434",
    )
    vector = embeddings.embed_query("test connection")
    print(f"Ollama embedding test passed. Vector length: {len(vector)}")


def crawl_docs(base_url: str, max_pages: int = 200) -> list[Document]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; DocsRAGBot/1.0; +https://example.com)"
            )
        }
    )

    queue = deque([normalize_url(base_url)])
    visited = set()
    documents = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        visited.add(url)
        print(f"Crawling ({len(visited)}/{max_pages}): {url}")

        html = fetch_html(session, url)
        if not html:
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        title = get_title(html)
        text = extract_main_text(html, url)

        if text:
            classification = classify_page(url, title, text)
            if classification["confidence"] <= 0.2:
                classification = classify_page_forced(url, title, text)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": title,
                        "url_tags": extract_tags_from_url(url),
                        "human_tags": extract_human_tags_from_url(url),
                        "primary_domain_id": classification[
                            "primary_domain_id"
                        ],
                        "primary_domain": classification["primary_domain"],
                        "primary_task_id": classification["primary_task_id"],
                        "primary_task_statement": classification[
                            "primary_task_statement"
                        ],
                        "secondary_task_ids": classification[
                            "secondary_task_ids"
                        ],
                        "classification_confidence": classification[
                            "confidence"
                        ],
                        "classification_reason": classification["reason"],
                        "custom_category": classification["custom_category"],
                    },
                )
            )

            print(
                "Classified as: "
                f"{classification['primary_task_id']} | "
                f"{classification['primary_task_statement']}"
            )

        for link in extract_links(html, url, base_url):
            if link not in visited:
                queue.append(link)

        time.sleep(REQUEST_DELAY_SECONDS)

    return documents

def build_classification_input(url: str, title: str, content: str) -> str:
    """Combine URL structure, title, and content into a rich signal string."""
    url_tags = extract_human_tags_from_url(url)
    tag_line = " > ".join(url_tags) if url_tags else ""

    parts = []
    if tag_line:
        parts.append(f"Navigation path: {tag_line}")
    if title:
        parts.append(f"Title: {title}")
    parts.append("")
    parts.append(content[:2500])

    return "\n".join(parts)

def build_faiss(
    documents: list[Document],
    output_dir: str,
    batch_size: int = 4,
    pause_seconds: float = 1.0,
    max_retries: int = 3,
):
    if not documents:
        raise ValueError("No documents were collected from the crawl.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    chunked_docs = []
    for i, doc in enumerate(chunks):
        metadata = dict(doc.metadata)
        metadata["chunk_id"] = i

        enriched_doc = enrich_document_for_embedding(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )
        chunked_docs.append(enriched_doc)

    print(f"Pages crawled: {len(documents)}")
    print(f"Chunks created: {len(chunked_docs)}")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://127.0.0.1:11434",
    )

    first_batch = chunked_docs[:batch_size]
    if not first_batch:
        raise ValueError("No chunks were created from the input documents.")

    vectorstore = None

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"Indexing first batch: 1-{len(first_batch)} "
                f"(attempt {attempt}/{max_retries})"
            )
            vectorstore = FAISS.from_documents(first_batch, embeddings)
            break
        except Exception as exc:
            print(f"First batch failed: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(pause_seconds * 2)

    for start in range(batch_size, len(chunked_docs), batch_size):
        end = min(start + batch_size, len(chunked_docs))
        batch = chunked_docs[start:end]

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"Indexing batch: {start + 1}-{end} "
                    f"(attempt {attempt}/{max_retries})"
                )
                vectorstore.add_documents(batch)
                success = True
                break
            except Exception as exc:
                print(f"Batch {start + 1}-{end} failed: {exc}")
                if attempt == max_retries:
                    raise
                time.sleep(pause_seconds * 2)

        if success:
            time.sleep(pause_seconds)

    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)

    print(f"Saved FAISS index to: {output_dir}")
    print(f"Pages crawled: {len(documents)}")
    print(f"Chunks indexed: {len(chunked_docs)}")


def main():
    test_ollama_embeddings()
    docs = crawl_docs(BASE_URL, max_pages=MAX_PAGES)

    write_classifications_to_csv(
        documents=docs,
        csv_path="page_classifications.csv",
    )

    build_faiss(
        documents=docs,
        output_dir=OUTPUT_DIR,
        batch_size=4,
        pause_seconds=1.0,
        max_retries=3,
    )


if __name__ == "__main__":
    main()