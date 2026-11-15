"""The whole chatbot: search the docs, then let the AI answer from those notes.

Step 11: joins search() from Step 8 with answer() from Step 10.

Run with:  uv run python -m rag_chatbot.rag "How do I set a timeout?"
(with no question, it runs three test questions)
"""

import sys
import time

from rag_chatbot.embed import load_model
from rag_chatbot.prompt import answer, find_sources, split_reply
from rag_chatbot.retrieve import search
from rag_chatbot.store import get_client, get_collection


def ask_docs(question: str, model, collection) -> dict:
    """Find the best notes for the question, then ask the AI to answer from them."""
    notes = search(question, model, collection)
    final, reply, sources = answer(question, notes)
    # Step 15: keep the AI's proof quote to show under the answer, but only if
    # there is an answer and the quote really is in the docs (checked again).
    quote = split_reply(reply)[0].strip()
    if not sources or not find_sources(quote, notes):
        quote = ""
    return {"answer": final, "reply": reply, "notes": notes, "sources": sources,
            "quote": quote}


def format_answer(result: dict) -> str:
    """The answer as the user sees it: the text, the docs quote, the source page(s)."""
    if not result["sources"]:
        return result["answer"]
    text = result["answer"]
    if result.get("quote"):
        # A code block keeps the docs' line breaks and code examples readable,
        # both in the terminal and in the Streamlit chat.
        text += f"\n\nFrom the docs:\n```text\n{result['quote']}\n```"
    links = "\n".join(f"  {url}" for url in result["sources"])
    return f"{text}\n\nSource:\n{links}"


def main() -> None:
    """Answer the question(s) and show the notes used and the time taken."""
    questions = [" ".join(sys.argv[1:])] if sys.argv[1:] else [
        "How do I set a timeout?",
        "How do I use a proxy?",
        "How do I send JSON in a POST request?",
        "How do I use requests.turbo_get()?",
    ]
    # Load the search model and the database once, then reuse them.
    model = load_model()
    collection = get_collection(get_client())

    for question in questions:
        start = time.time()
        result = ask_docs(question, model, collection)
        print(f"\nQuestion: {question}")
        print("Notes found by the search:")
        for note in result["notes"]:
            print(f"  - {note['id']}  (score {note['score']:.2f})")
        print(f"AI wrote: {' '.join(result['reply'].split())}")
        print(f"({time.time() - start:.1f} seconds)")
        print(f"\n{format_answer(result)}\n" + "-" * 60)


if __name__ == "__main__":
    main()
