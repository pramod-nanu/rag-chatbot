"""Put the chunks into ChromaDB (the "filing cabinet").

Step 7: store every chunk's text, its vector (from Step 6) and its page
link + title in a Chroma collection saved in data/chroma/.

Run with:  uv run python -m rag_chatbot.store
"""

import json
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from rag_chatbot.chunk import CHUNKS_FILE
from rag_chatbot.clean import load_pages
from rag_chatbot.embed import IDS_FILE, VECTORS_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "requests_docs"  # Chroma names need 3+ characters


def get_client() -> chromadb.ClientAPI:
    """Open the Chroma database saved on disk in data/chroma/."""
    # anonymized_telemetry=False: Chroma sends usage statistics by default;
    # our rule is that nothing leaves this computer.
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Open (or create) our collection of chunks."""
    return client.get_or_create_collection(
        COLLECTION_NAME,
        # cosine: compare vectors by meaning (the angle between them).
        configuration={"hnsw": {"space": "cosine"}},
        # We always give Chroma our own bge-small vectors; turn off Chroma's
        # built-in model so a different model can never be mixed in.
        embedding_function=None,
    )


def main() -> None:
    """Fill the collection with all chunks, vectors and page links."""
    chunks = load_pages(CHUNKS_FILE)
    vectors = np.load(VECTORS_FILE)
    vector_ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))

    # Row i of the vectors must belong to chunk i; stop if the files don't match
    # (e.g. chunk.py was re-run but embed.py wasn't).
    if vector_ids != [c["id"] for c in chunks]:
        raise SystemExit("Chunks and vectors don't match: re-run rag_chatbot.embed first.")

    client = get_client()
    # Start from an empty collection every run, so re-running never duplicates.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = get_collection(client)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"url": c["url"], "title": c["title"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
    )
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")
    print(f"Saved to {CHROMA_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
