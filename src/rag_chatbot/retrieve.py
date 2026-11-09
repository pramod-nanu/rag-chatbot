"""Find the chunks that best answer a question, and test how well that works.

Step 8: search() asks Chroma for the closest chunks (the chatbot reuses it
later). Running this file gives the search a small exam: questions whose
right page we already know, and a score at the end. No LLM yet.

Run with:  uv run python -m rag_chatbot.retrieve
"""

from rag_chatbot.embed import embed_question, load_model
from rag_chatbot.scrape import ALLOWED_PREFIX
from rag_chatbot.store import get_client, get_collection

TOP_K = 3  # how many chunks to return per question

# (question, pages that contain the answer). A question passes when one of
# its pages is among the TOP_K results. Every page was checked by hand to
# really contain the topic.
EXAM = [
    ("How do I install requests?", ["user/install/"]),
    ("How do I send JSON in a POST request?", ["user/quickstart/"]),
    ("How do I set a timeout?", ["user/quickstart/", "user/advanced/"]),
    ("Who created requests?", ["dev/authors/"]),
    ("How do I use a proxy?", ["user/advanced/"]),
    ("How do I use HTTP basic authentication?", ["user/authentication/"]),
    ("How do I keep cookies across requests with a session?", ["user/advanced/"]),
    ("How do I check the response status code?", ["user/quickstart/"]),
    ("How do I turn off SSL certificate verification?", ["user/advanced/"]),
    ("Where do I report a security vulnerability?", ["community/vulnerabilities/"]),
]


def search(question: str, model, collection, k: int = TOP_K) -> list[dict]:
    """Return the k chunks closest in meaning to the question, best first."""
    result = collection.query(query_embeddings=[embed_question(model, question)], n_results=k)
    return [
        {
            "id": chunk_id,
            "url": meta["url"],
            "title": meta["title"],
            "text": text,
            # Chroma gives a distance (smaller = closer); turn it into a
            # similarity score (bigger = closer), like in Step 6.
            "score": 1 - distance,
        }
        for chunk_id, distance, meta, text in zip(
            result["ids"][0], result["distances"][0], result["metadatas"][0], result["documents"][0]
        )
    ]


def main() -> None:
    """Run the exam and print a score."""
    model = load_model()
    collection = get_collection(get_client())

    passed = 0
    for question, right_pages in EXAM:
        results = search(question, model, collection)
        pages = [r["url"].removeprefix(ALLOWED_PREFIX) for r in results]
        # Rank (1, 2, 3) of the first result that comes from a right page.
        rank = next((i for i, page in enumerate(pages, start=1) if page in right_pages), None)

        if rank:
            passed += 1
            print(f"PASS  {question}\n      right page found at rank {rank}: {pages[rank - 1]}")
        else:
            print(f"FAIL  {question}\n      expected {right_pages}, got {pages}")

    print(f"\nScore: {passed} / {len(EXAM)}")


if __name__ == "__main__":
    main()
