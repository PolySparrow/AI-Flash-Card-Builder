from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings


OUTPUT_DIR = "./faiss_claude_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"


def load_vectorstore(output_dir: str):
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://127.0.0.1:11434",
    )
    return FAISS.load_local(
        output_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def rerank_docs_by_metadata(docs: list[Document], query: str) -> list[Document]:
    query_lower = query.lower()

    def score(doc: Document) -> int:
        metadata = doc.metadata
        value = 0

        for field in [
            "primary_domain",
            "primary_task_statement",
            "custom_category",
            "title",
        ]:
            field_value = str(metadata.get(field, "")).lower()
            if field_value and field_value in query_lower:
                value += 3

        for tag in metadata.get("human_tags", []):
            if tag.lower() in query_lower:
                value += 2

        for task_id in metadata.get("secondary_task_ids", []):
            if task_id.lower() in query_lower:
                value += 1

        return value

    return sorted(docs, key=score, reverse=True)


def ask_rag(question: str, output_dir: str = OUTPUT_DIR, k: int = 8):
    vectorstore = load_vectorstore(output_dir)
    docs = vectorstore.similarity_search(question, k=k)
    docs = rerank_docs_by_metadata(docs, question)[:5]

    context = "\n\n---\n\n".join(
        (
            f"Source: {doc.metadata.get('source', '')}\n"
            f"Title: {doc.metadata.get('title', '')}\n"
            f"Domain: {doc.metadata.get('primary_domain', '')}\n"
            f"Primary Task: {doc.metadata.get('primary_task_id', '')} - "
            f"{doc.metadata.get('primary_task_statement', '')}\n"
            f"Custom Category: {doc.metadata.get('custom_category', '')}\n\n"
            f"{doc.page_content}"
        )
        for doc in docs
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0,
    )

    prompt = f"""
You are a helpful assistant. Answer using only the provided context.
If the answer is not supported by the context, say "I don't know."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:\n")
    for doc in docs:
        print(
            f"- {doc.metadata.get('source', '')} | "
            f"{doc.metadata.get('primary_task_id', '')} | "
            f"{doc.metadata.get('custom_category', '')}"
        )


if __name__ == "__main__":
    ask_rag("Which pages relate to MCP servers and tool choice?")