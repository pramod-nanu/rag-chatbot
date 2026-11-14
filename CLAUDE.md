# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task

Build a RAG chatbot for a website. Use an open-source embedding model and an LLM, both running locally on CPU only (no GPU, no cloud APIs).

## How to work with me

- I'm a beginner and want to learn while building.
- Go one step at a time. Before each command or file change, explain in simple words: what it does, why we need it, what other options exist, and why we chose this one.
- Wait for me to confirm before moving to the next step. Do not implement later steps of the plan unprompted.

## My machine

- macOS, Apple M3, 8 GB RAM.
- conda "base" is active in my terminal. Always use uv, never conda or pip directly.

## CPU-only rule (important)

The M3 has a GPU that libraries use automatically. Always force CPU:

- sentence-transformers: `SentenceTransformer(..., device="cpu")`
- Ollama: pass `options={"num_gpu": 0}`

## Tech stack

- Python 3.12 (uv-managed, `python-preference = "only-managed"`)
- Scraping: requests + BeautifulSoup
- Chunking: LangChain text splitters
- Embeddings: `BAAI/bge-small-en-v1.5` via sentence-transformers (max 512 tokens)
- Vector store: ChromaDB, stored in `data/`
- LLM: Ollama, 3B model in 4-bit (e.g. `llama3.2:3b`), due to 8 GB RAM
- UI: Streamlit first; final step: FastAPI endpoint + website chat widget
- Tests: pytest

## Website

- Site: requests documentation, https://requests.readthedocs.io/en/latest/
- Scope: only pages under `/en/latest/`. Skip `/en/stable/` (duplicate) and old versions (blocked by robots.txt).
- No full sitemap (`sitemap.xml` lists only the version roots), so the scraper follows links within `/en/latest/`.
- Also skip `/_modules/` (the library's Python source, reached via "[source]" links): it was 50% of the text and would crowd out the explained docs. Revisit in step 15.
- Scrape with `uv run python -m rag_chatbot.scrape` → `data/raw/pages.jsonl` (one `{"url", "title", "text"}` per line, 15 pages).
- Clean with `uv run python -m rag_chatbot.clean` → `data/clean/pages.jsonl` (no network; never modifies the raw file). It strips the title suffix and `[source]` labels and straightens curly quotes. Repeated paragraphs and tiny pages are kept on purpose: they're real content (e.g. the same parameter text under several API functions).
- Chunk with `uv run python -m rag_chatbot.chunk` → `data/chunks/chunks.jsonl` (153 chunks; fields `id`, `url`, `title`, `chunk_index`, `n_tokens`, `text`). 400 tokens + 50 overlap, counted with bge-small's own tokenizer (truncation turned off so oversized chunks are visible). Each chunk's text starts with its page title. Overlap only carries whole paragraphs, so about half of neighbouring chunks share no text — expected LangChain behaviour.
- Embed with `uv run python -m rag_chatbot.embed` → `data/embeddings/chunks.npy` (153 × 384, normalized) + `ids.json` (row → chunk id). Takes ~5 s on CPU. Questions must go through `embed_question()`, which adds bge's query prefix; chunks get no prefix.
- Quick check: `uv run python -m rag_chatbot.check_embeddings "<question>"` prints the top 3 chunks from the `.npy` file (no Chroma).
- Store with `uv run python -m rag_chatbot.store` → Chroma collection `requests_docs` in `data/chroma/` (rebuilt from scratch each run; metadata `url`, `title`, `chunk_index`). Uses the saved Step 6 vectors, cosine space, `embedding_function=None` (never Chroma's built-in model) and `anonymized_telemetry=False`. Chroma returns distances: similarity = 1 − distance.
- Retrieve with `search()` in `retrieve.py` (top 3 chunks, each with `id`, `url`, `title`, `text`, `score`) — reuse it for the chatbot. `uv run python -m rag_chatbot.retrieve` runs a 10-question exam (right page in top 3); it scored 10/10, all at rank 1. The exam checks the page, not the exact chunk.
- LLM: Ollama 0.40.16 (installed via the install script), model `llama3.2:3b` (Q4_K_M, 2.0 GB on disk, ~2.9 GB in RAM). `ask()` in `llm.py` sends one question with `num_gpu: 0` and `temperature: 0` (same question → same answer) — reuse it for the chatbot. `uv run python -m rag_chatbot.llm "<question>"` prints the answer and time (~15 s for a long answer); `ollama ps` should show `100% CPU`.
- Without docs, the LLM hallucinates confidently: it explained the non-existent `requests.turbo_get()`, and for timeouts it invented `session.timeout = 5` and `retries=`/`backoff_factor=` arguments. Given a text without the answer plus an "answer only from this text" rule, it correctly said "I don't know". Use these as test questions in step 14.

## Scraping rules (from robots.txt and Read the Docs' automated access policy)

- Check robots.txt before fetching (`urllib.robotparser`).
- Max 1 request per second.
- Identify the bot with an honest User-Agent: `"rag-chatbot-learning-project (contact: https://github.com/pramod-nanu)"`. Never pretend to be a browser.
- Content is copyrighted by the requests project: keep scraped data local in `data/`, use it only for this learning project, and always cite source URLs in answers.

## Plan

1. Setup (DONE)
2. Choose website, check robots.txt and terms (DONE)
3. Scrape pages, save clean text with URL (DONE)
4. Clean text (remove menus, footers, duplicates) (DONE)
5. Chunk (300-500 tokens, small overlap) (DONE)
6. Embed chunks locally (DONE)
7. Store in Chroma with URL metadata (DONE)
8. Retrieval test (no LLM yet) (DONE)
9. Install Ollama, test the LLM on its own (DONE)
10. RAG prompt: answer only from context, otherwise say "I don't know"  <-- NEXT
11. Connect retrieval and LLM
12. Add citations (source URLs)
13. Streamlit chat UI
14. Evaluate: 15-20 test questions incl. unanswerable ones; note speed
15. Improve: chunk size, top-k, prompt, model size
16. README
17. FastAPI endpoint + website chat widget
18. Optional: more unit tests, Docker, hybrid search

## Golden rule

Never add the LLM until retrieval returns good chunks.

## Commands

```bash
uv sync                      # create/update .venv from uv.lock
uv run rag-chatbot           # run the app (entry point: rag_chatbot:main)
uv add <package>             # add a runtime dependency (keeps pyproject.toml + uv.lock in sync)
uv add --dev <package>       # add a dev-only dependency
```

`.venv/` is created by uv and disposable — rebuild with `uv sync`.

Once pytest is added (`uv add --dev pytest`): `uv run pytest`; single test: `uv run pytest tests/test_file.py::test_name`.

## Layout

- `src/rag_chatbot/` — the package (src layout, built with `uv_build`). `__init__.py:main` is the `rag-chatbot` console script. The project is installed editable into `.venv`, so code changes take effect without reinstalling.
- `data/` — scraped text and the Chroma database. Git-ignored; never commit its contents.
- `tests/` — pytest tests (empty so far).

- After each step works, update the Plan section (mark it DONE and move NEXT) and make a git commit with a clear message.
