"""A small web server (the "kitchen") that answers questions from a web page.

Step 17: one endpoint, POST /ask. A web page sends {"question": "..."} and
gets back the answer, the docs quote and the source links, as JSON.
It reuses ask_docs() from Step 11, so it is the same chatbot.

Run with:  uv run uvicorn rag_chatbot.api:app --host 127.0.0.1 --port 8000
Chat page: http://localhost:8000/      (demo website with the chat widget)
Test page: http://localhost:8000/docs  (made automatically by FastAPI)
--host 127.0.0.1 means only this computer can reach it.
"""

from functools import cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_chatbot.embed import load_model
from rag_chatbot.rag import ask_docs
from rag_chatbot.store import get_client, get_collection

app = FastAPI(title="Requests Docs Chatbot")
WIDGET_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """The demo web page with the chat widget. Served from this same server,
    so the browser allows the page to call /ask without extra settings."""
    return FileResponse(WIDGET_PAGE)


class Question(BaseModel):
    """What the web page sends."""

    # Not empty, and not huge (the AI only needs a short question).
    question: str = Field(min_length=1, max_length=500)


class Answer(BaseModel):
    """What we send back."""

    answer: str
    quote: str  # the checked sentence from the docs ("" if none)
    sources: list[str]  # page links ([] when the answer is "I don't know.")


@cache
def load_search():
    """Load the search model and open the database once, on the first question."""
    return load_model(), get_collection(get_client())


# A plain "def" (not "async def"): FastAPI then runs it in a separate thread,
# so the slow AI call doesn't block the server.
@app.post("/ask")
def ask(body: Question) -> Answer:
    """Answer one question about requests, only from its documentation."""
    model, collection = load_search()
    result = ask_docs(body.question, model, collection)
    return Answer(answer=result["answer"], quote=result["quote"], sources=result["sources"])
