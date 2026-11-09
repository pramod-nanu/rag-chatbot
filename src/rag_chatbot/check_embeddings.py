"""Check the embeddings by asking a question and seeing which notes match.

Run with:  uv run python -m rag_chatbot.check_embeddings "How do I install requests?"
"""

import json
import sys

import numpy as np

from rag_chatbot.chunk import CHUNKS_FILE
from rag_chatbot.clean import load_pages
from rag_chatbot.embed import IDS_FILE, VECTORS_FILE, embed_question, load_model


def main() -> None:
    """Show the 3 notes whose meaning is closest to the question."""
    question = " ".join(sys.argv[1:]) or "How do I set a timeout?"

    vectors = np.load(VECTORS_FILE)
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    texts = {c["id"]: c["text"] for c in load_pages(CHUNKS_FILE)}

    scores = vectors @ embed_question(load_model(), question)

    print(f"\nQuestion: {question}\n")
    for rank, i in enumerate(scores.argsort()[::-1][:3], start=1):
        # Skip the title line and show the start of the note's real text.
        preview = " ".join(texts[ids[i]].split("\n", 2)[-1].split())[:200]
        print(f"{rank}. {ids[i]}   (score {scores[i]:.2f})")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
