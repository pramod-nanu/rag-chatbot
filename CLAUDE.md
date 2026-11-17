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
- RAG prompt (`prompt.py`): rules alone did NOT stop the 3B model from answering from memory (8 prompt variants tried, incl. a repeated reminder and a worked example — it always made up the proxy answer). What works is "show your proof": the AI must write `Quote:` (a sentence copied from the notes) then `Answer:`; `quote_is_real()` checks the quote appears word for word in the notes (if `Quote:` is missing, the answer itself is the proof), otherwise `answer()` returns "I don't know.". `llm.chat(messages)` sends system + user messages. `uv run python -m rag_chatbot.prompt` runs 6 tests on a hand-written note: 6/6. The check is strict, so slightly reworded quotes from real chunks will give a false "I don't know" — measure this in step 14.
- Full chatbot: `ask_docs(question, model, collection)` in `rag.py` = `search()` + `answer()`, returns `answer`, `reply` (raw AI text) and `notes`. `uv run python -m rag_chatbot.rag "<question>"` (no question → 3 test questions) shows the notes found, the AI's reply, the answer and the time (~11–18 s on CPU). Correct for timeout, proxy, JSON POST; "I don't know." for turbo_get(), MySQL, capital of France, etc. Weakness: the code example often ends up only in the quote, not in the final answer.
- Citations: `find_sources()` in `prompt.py` returns the URL(s) of the note(s) containing the proof quote (deduplicated); `answer()` returns `(final, reply, sources)` and "I don't know" never gets a source. `format_answer(result)` in `rag.py` prints the answer + `Source:` links — reuse it for the UI. Limit: only the page of the quote is cited, even if the answer also used another note (e.g. timeout cites quickstart, but the `(3.05, 27)` example is from advanced). Revisit in step 15.
- Chat UI: `uv run streamlit run src/rag_chatbot/app.py` → http://localhost:8501 (Ctrl+C to stop). `app.py` loads the search model + collection once (`st.cache_resource`), keeps the chat in `st.session_state.messages`, and shows `format_answer()` output. `.streamlit/config.toml` sets `gatherUsageStats = false` (no telemetry). Test without a browser via `streamlit.testing.v1.AppTest`.
- Evaluate with `uv run python -m rag_chatbot.evaluate` (~5 min): 14 answerable questions (pass = an answer whose source is one of the right pages) + 6 unanswerable (pass = "I don't know."). Saves every answer to `data/eval/results.jsonl`. Baseline (step 14): 20/20, average 15.9 s (8.8–26.1 s). The score checks the page, not correctness — reading the answers found: retry answer slightly wrong (skips `HTTPAdapter`), upload/headers answers vague with no code, proxy answer says "as shown in Note 2" (users can't see notes). Nothing made up.
- Step 15 lessons: adding rules to the prompt ("include the code", "don't mention Note 1") broke it — the 3B model stopped writing `Quote:` and the score fell to 9/20; two other layouts also failed. Keep the AI's rules minimal and fix things in Python instead. Improvement 1 (kept, still 20/20, ~12 s average): `ask_docs()` returns the checked proof `quote`, and `format_answer()` shows it as a "From the docs:" code block (often contains the code example); `answer()` replaces "Note N" with "the docs". Still weak: retry answer (quote stops before the code).
- Web: `uv run uvicorn rag_chatbot.api:app --host 127.0.0.1 --port 8000`. `api.py` has `POST /ask` (`{"question"}` 1–500 chars → `{"answer", "quote", "sources"}`, 422 if invalid; plain `def` so the slow AI call runs in a thread; search model loaded once via `functools.cache`) and `GET /` serving `static/index.html` (demo page + 💬 widget; same origin, so no CORS needed). The widget uses `textContent` only (no innerHTML for AI text) and links only `https://requests.readthedocs.io/` URLs. Test with `fastapi.testclient.TestClient`.

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
10. RAG prompt: answer only from context, otherwise say "I don't know" (DONE)
11. Connect retrieval and LLM (DONE)
12. Add citations (source URLs) (DONE)
13. Streamlit chat UI (DONE)
14. Evaluate: 15-20 test questions incl. unanswerable ones; note speed (DONE)
15. Improve: chunk size, top-k, prompt, model size (DONE: "From the docs" quote; other ideas not needed yet)
16. README (DONE)
17. FastAPI endpoint + website chat widget (DONE)
18. Optional: more unit tests, Docker, hybrid search  <-- NEXT

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
- `tests/` — pytest tests (`uv run pytest`, < 1 s, no AI or network). `test_prompt.py`: 12 checks of the proof check (`split_reply`, `find_sources`, `answer`), using `monkeypatch.setattr(prompt, "chat", ...)` to replace the AI with a hand-written reply.

- After each step works, update the Plan section (mark it DONE and move NEXT) and make a git commit with a clear message.
