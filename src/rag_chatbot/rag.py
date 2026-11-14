"""The whole chatbot: search the docs, then let the AI answer from those notes.

Step 11: joins search() from Step 8 with answer() from Step 10.

Run with:  uv run python -m rag_chatbot.rag "How do I set a timeout?"
(with no question, it runs three test questions)
"""

import sys
import time

from rag_chatbot.embed import load_model
from rag_chatbot.prompt import answer
from rag_chatbot.retrieve import search
from rag_chatbot.store import get_client, get_collection


def ask_docs(question: str, model, collection) -> dict:
    """Find the best notes for the question, then ask the AI to answer from them."""
    notes = search(question, model, collection)
    final, reply = answer(question, notes)
    return {"answer": final, "reply": reply, "notes": notes}


def main() -> None:
    """Answer the question(s) and show the notes used and the time taken."""
    questions = [" ".join(sys.argv[1:])] if sys.argv[1:] else [
        "How do I set a timeout?",
        "How do I use a proxy?",
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
        print(f"Answer:   {result['answer']}")
        print(f"({time.time() - start:.1f} seconds)")


if __name__ == "__main__":
    main()
