# AI-Flash-Card-Builder
Uses oLlama and web parsing to scan documentation and then create multiple choice questions based on a sample set of questions

# URL RAG FAISS

## Setup

### 1. Create a virtual environment

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

Using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or using `pyproject.toml`:

```bash
pip install --upgrade pip
pip install .
```

For editable install:

```bash
pip install -e .
```

### 3. Install Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

### 4. Optional: install Playwright browser

Only needed for JavaScript-heavy documentation sites.

```bash
playwright install chromium
```

## Recommended `requirements.txt`

```text
requests>=2.32.0
beautifulsoup4>=4.12.0
lxml>=5.2.0
trafilatura>=1.9.0
faiss-cpu>=1.8.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-ollama>=0.2.0
python-dotenv>=1.0.1
```

## Full optional `requirements.txt`

Use this if you also want browser-rendered crawling support:

```text
requests>=2.32.0
beautifulsoup4>=4.12.0
lxml>=5.2.0
trafilatura>=1.9.0
faiss-cpu>=1.8.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-ollama>=0.2.0
python-dotenv>=1.0.1
playwright>=1.46.0
```

## Example project structure

```text
url-rag-faiss/
  crawl_and_build_faiss.py
  query_faiss.py
  requirements.txt
  pyproject.toml
  README.md
```
```