# RAG chatbot for the requests documentation

A chatbot that answers questions about the Python
[requests](https://requests.readthedocs.io/en/latest/) library, using only
its documentation. Everything runs locally on CPU: no GPU, no cloud APIs.

```
How do I upload a file?

You can upload a file by passing a dictionary of tuples, where each tuple
contains the filename, file data, and optional content type and headers.

From the docs:
>>> r = requests.post(url, files=files)

Source: https://requests.readthedocs.io/en/latest/user/quickstart/
```

If the answer is not in the docs, it says **"I don't know."** instead of
guessing.

This is a learning project, built step by step.

## How it works

RAG means "Retrieval-Augmented Generation": first find the right part of
the docs, then let an AI write the answer from it.

1. **Scrape**: download the docs pages (15 pages under `/en/latest/`).
2. **Clean**: remove leftovers such as `[source]` labels and curly quotes.
3. **Chunk**: split the pages into 153 small pieces of about 400 tokens.
4. **Embed**: turn each piece into numbers that capture its meaning, with
   [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).
5. **Store**: save the pieces, their numbers and page links in
   [ChromaDB](https://www.trychroma.com/).
6. **Answer**: for each question, search for the 3 closest pieces and give
   them to [`llama3.2:3b`](https://ollama.com/library/llama3.2) (through
   [Ollama](https://ollama.com/)) with strict rules.

### The "show your proof" check

A small 3B model often answers from its own memory, even when told not to.
So the AI must first copy the sentence from the docs that answers the
question (`Quote: ...`), then answer (`Answer: ...`). Python checks that the
quote really appears, word for word, in the docs pieces. If it doesn't, the
reply is "I don't know." The page that holds the quote becomes the source
link, and the quote is shown under the answer as "From the docs".

## Requirements

- macOS or Linux with about 8 GB of RAM (built on an Apple M3 with 8 GB)
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you)
- [Ollama](https://ollama.com/download)
- About 3 GB of disk space (Python packages + the 2 GB model)

## Setup

```bash
uv sync                    # install Python and the packages into .venv
ollama pull llama3.2:3b    # download the AI model (about 2 GB, once)
```

## Build the docs database

Run these once, in this order. Each step reads the output of the one before
it, and everything is saved in `data/`.

```bash
uv run python -m rag_chatbot.scrape   # download the docs → data/raw/
uv run python -m rag_chatbot.clean    # clean the text   → data/clean/
uv run python -m rag_chatbot.chunk    # split into pieces → data/chunks/
uv run python -m rag_chatbot.embed    # meaning numbers  → data/embeddings/
uv run python -m rag_chatbot.store    # save in Chroma   → data/chroma/
```

The scraper is polite: it follows `robots.txt`, waits 1 second between
requests and sends an honest User-Agent.

## Use the chatbot

**In the browser** (Streamlit chat window):

```bash
uv run streamlit run src/rag_chatbot/app.py
```

Then open http://localhost:8501. Press Ctrl+C in the terminal to stop it.

**On a web page** (FastAPI server + chat widget):

```bash
uv run uvicorn rag_chatbot.api:app --host 127.0.0.1 --port 8000
```

Then open http://localhost:8000/ and click the 💬 button in the bottom-right
corner. The page sends each question to `POST /ask`, which returns JSON:
`{"answer": "...", "quote": "...", "sources": ["https://..."]}`. FastAPI's
test page for the endpoint is at http://localhost:8000/docs. With
`--host 127.0.0.1`, only this computer can reach the server.

**In the terminal:**

```bash
uv run python -m rag_chatbot.rag "How do I set a timeout?"
```

An answer takes about 10–20 seconds on CPU.

## Test it

```bash
uv run python -m rag_chatbot.retrieve   # search only: 10 questions, right page in the top 3?
uv run python -m rag_chatbot.prompt     # rules + proof check on a hand-written note
uv run python -m rag_chatbot.evaluate   # whole chatbot: 20 questions (about 5 minutes)
```

Current results:

| Test | Score |
|---|---|
| Search finds the right page (top 3) | 10 / 10 |
| Answers from the right page | 14 / 14 |
| Says "I don't know" to questions the docs don't cover | 6 / 6 |
| Average time per answer | about 12–16 s |

The evaluation checks the source page, not every sentence. Reading the
answers showed they are sometimes short or vague (for example, the retry
answer skips a step), but nothing was made up.

## Project layout

```
src/rag_chatbot/
  scrape.py            download the docs pages
  clean.py             clean the text
  chunk.py             split into pieces
  embed.py             embedding model (CPU only)
  check_embeddings.py  quick search check without Chroma
  store.py             save to / open the Chroma database
  retrieve.py          search() + the search test
  llm.py               talk to the AI through Ollama (CPU only)
  prompt.py            rules for the AI + the "show your proof" check
  rag.py               ask_docs(): the whole chatbot
  app.py               Streamlit chat window
  api.py               FastAPI server: POST /ask + serves the chat page
  static/index.html    demo web page with the chat widget
  evaluate.py          the 20-question exam
data/                  everything downloaded or built (not in git)
```

## Settings worth knowing

- **CPU only:** the embedding model uses `device="cpu"`, and Ollama gets
  `num_gpu: 0`. Check with `ollama ps`: it should say `100% CPU`.
- **No telemetry:** Chroma and Streamlit usage statistics are turned off.
- **Same question → same answer:** the AI runs with `temperature: 0`.

## About the content

The documentation is © the requests project. The downloaded copy stays in
`data/` on this computer (it is not committed to git) and is used only for
this learning project. Every answer links to its source page.

## Run chatbot again

## Run the chatbot again

```bash
ollama serve        # only if the 🦙 isn't already in the menu bar
uv run streamlit run src/rag_chatbot/app.py

uv run uvicorn rag_chatbot.api:app --host 127.0.0.1 --port 8000
```

