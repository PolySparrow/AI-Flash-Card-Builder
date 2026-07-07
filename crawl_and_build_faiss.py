import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


BASE_URL = "https://platform.claude.com/docs/en/"
OUTPUT_DIR = "./faiss_claude_docs"
EMBED_MODEL = "nomic-embed-text"

MAX_PAGES = 500
REQUEST_DELAY_SECONDS = 0.5
TIMEOUT_SECONDS = 20


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
    normalized = normalize_url(url)
    return normalized.startswith(normalize_url(base_url))


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


def crawl_docs(base_url: str, max_pages: int = 500) -> list[Document]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; DocsRAGBot/1.0; +https://example.com/bot)"
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
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": title,
                    },
                )
            )

        for link in extract_links(html, url, base_url):
            if link not in visited:
                queue.append(link)

        time.sleep(REQUEST_DELAY_SECONDS)

    return documents


def build_faiss(
    documents: list[Document],
    output_dir: str,
    batch_size: int = 8,
    pause_seconds: float = 0.5,
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
        chunked_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    print(f"Pages crawled: {len(documents)}")
    print(f"Chunks created: {len(chunked_docs)}")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://127.0.0.1:11434",
    )

    first_batch = chunked_docs[:batch_size]
    if not first_batch:
        raise ValueError("No chunks were created from the input documents.")

    print(f"Indexing first batch: 1-{len(first_batch)}")
    vectorstore = FAISS.from_documents(first_batch, embeddings)

    for start in range(batch_size, len(chunked_docs), batch_size):
        end = min(start + batch_size, len(chunked_docs))
        batch = chunked_docs[start:end]

        print(f"Indexing batch: {start + 1}-{end}")
        vectorstore.add_documents(batch)
        time.sleep(pause_seconds)

    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)

    print(f"Saved FAISS index to: {output_dir}")
    print(f"Pages crawled: {len(documents)}")
    print(f"Chunks indexed: {len(chunked_docs)}")


def main():
    docs = crawl_docs(BASE_URL, max_pages=MAX_PAGES)
    build_faiss(docs, OUTPUT_DIR)


if __name__ == "__main__":
    main()