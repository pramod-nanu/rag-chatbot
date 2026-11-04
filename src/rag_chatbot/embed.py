"""Give every chunk a "meaning label" (an embedding).

Step 6: read data/chunks/chunks.jsonl, turn each chunk's text into 384
numbers with bge-small (on CPU only), and save them to data/embeddings/.

Run with:  uv run python -m rag_chatbot.embed
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_chatbot.chunk import CHUNKS_FILE, EMBEDDING_MODEL
from rag_chatbot.clean import load_pages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
VECTORS_FILE = EMBEDDINGS_DIR / "chunks.npy"  # one row of 384 numbers per chunk
IDS_FILE = EMBEDDINGS_DIR / "ids.json"  # which chunk each row belongs to

# bge models were trained with this prefix on questions (not on chunks).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_model() -> SentenceTransformer:
    """Load bge-small. device="cpu": never use the M3's GPU (see CLAUDE.md)."""
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Turn texts into vectors. Normalized, so similarity = a simple dot product."""
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def embed_question(model: SentenceTransformer, question: str) -> np.ndarray:
    """Turn a user question into a vector (with bge's question prefix)."""
    return model.encode(QUERY_PREFIX + question, normalize_embeddings=True)


def quick_test(model: SentenceTransformer) -> None:
    """Check that related sentences score higher than unrelated ones."""
    question = embed_question(model, "How do I set a timeout?")
    sentences = [
        "You can tell Requests to stop waiting for a response after a given number of seconds.",
        "Requests was lovingly created by Kenneth Reitz.",
    ]
    for sentence, vector in zip(sentences, embed_texts(model, sentences)):
        # Both vectors are normalized, so the dot product is the similarity:
        # close to 1 = same meaning, lower = less related.
        print(f"  {float(question @ vector):.2f}  {sentence}")


def main() -> None:
    """Embed every chunk and save the vectors."""
    chunks = load_pages(CHUNKS_FILE)
    model = load_model()

    start = time.time()
    vectors = embed_texts(model, [c["text"] for c in chunks])
    print(f"Embedded {len(chunks)} chunks in {time.time() - start:.0f} seconds "
          f"-> {vectors.shape[0]} rows x {vectors.shape[1]} numbers.")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_FILE, vectors)
    IDS_FILE.write_text(json.dumps([c["id"] for c in chunks], indent=1), encoding="utf-8")
    print(f"Saved to {VECTORS_FILE.relative_to(PROJECT_ROOT)} and {IDS_FILE.relative_to(PROJECT_ROOT)}")

    print("Quick test - question: 'How do I set a timeout?'")
    quick_test(model)


if __name__ == "__main__":
    main()
