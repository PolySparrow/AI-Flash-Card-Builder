from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama


FAISS_DIR = "./faiss_claude_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def ask(question: str):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', '')}\n"
        f"Title: {doc.metadata.get('title', '')}\n\n"
        f"{doc.page_content}"
        for doc in docs
    )

    prompt = f"""
Answer the user's question using only the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:\n")
    for doc in docs:
        print(doc.metadata.get("source", ""))


if __name__ == "__main__":
    ask("How do I authenticate with the Claude API?")