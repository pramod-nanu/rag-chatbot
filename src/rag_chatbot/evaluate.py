"""Give the whole chatbot an exam: 20 questions, a score and the speed.

Step 14: 14 questions the docs answer (the answer must come from a right
page) and 6 they don't (the bot must say "I don't know."). Re-run after
every change in Step 15 to see if the score goes up or down.

Run with:  uv run python -m rag_chatbot.evaluate   (takes about 5 minutes)
Every answer is also saved to data/eval/results.jsonl, to read afterwards.
"""

import json
import time
from pathlib import Path

from rag_chatbot.embed import load_model
from rag_chatbot.prompt import I_DONT_KNOW
from rag_chatbot.rag import ask_docs
from rag_chatbot.scrape import ALLOWED_PREFIX
from rag_chatbot.store import get_client, get_collection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results.jsonl"

# (question, right pages). Every page was checked to really contain the topic.
ANSWERABLE = [
    ("How do I install requests?", ["user/install/"]),
    ("How do I send JSON in a POST request?", ["user/quickstart/"]),
    ("How do I set a timeout?", ["user/quickstart/", "user/advanced/"]),
    ("How do I add custom headers to a request?", ["user/quickstart/", "user/advanced/"]),
    ("How do I pass URL query parameters?", ["user/quickstart/"]),
    ("How do I upload a file?", ["user/quickstart/", "user/advanced/"]),
    ("How do I use HTTP basic authentication?",
     ["user/authentication/", "user/advanced/", "api/"]),
    ("How do I check the response status code?", ["user/quickstart/"]),
    ("What does raise_for_status() do?", ["user/quickstart/", "api/"]),
    ("How do I download a large file in chunks?",
     ["user/quickstart/", "user/advanced/", "api/"]),
    ("How do I use a proxy?", ["user/advanced/", "api/"]),
    ("How do I turn off SSL certificate verification?", ["user/advanced/"]),
    ("How do I make requests retry failed connections?", ["user/advanced/"]),
    ("Who created requests?", ["dev/authors/", "community/faq/"]),
]

# Questions the docs don't answer: the bot must say "I don't know."
UNANSWERABLE = [
    "How do I use requests.turbo_get()?",  # made-up function
    "How do I connect to a MySQL database?",
    "How do I make a website with Django?",
    "How do I send an email with smtplib?",
    "What is the capital of France?",  # the AI knows it, but it's not in the docs
    "What is 25 times 4?",
]


def grade(result: dict, right_pages: list[str] | None) -> tuple[bool, str]:
    """Return (passed, reason). right_pages=None means "should say I don't know"."""
    said_idk = result["answer"] == I_DONT_KNOW
    pages = [url.removeprefix(ALLOWED_PREFIX) for url in result["sources"]]
    if right_pages is None:
        return said_idk, "said I don't know" if said_idk else "answered (should not)"
    if said_idk:
        return False, "said I don't know (should answer)"
    if not any(page in right_pages for page in pages):
        return False, f"wrong source page {pages}"
    return True, f"answered from {pages}"


def main() -> None:
    """Run all questions, print PASS/FAIL, the score and the speed."""
    model = load_model()
    collection = get_collection(get_client())
    exam = [(q, pages) for q, pages in ANSWERABLE] + [(q, None) for q in UNANSWERABLE]

    rows = []
    for number, (question, right_pages) in enumerate(exam, start=1):
        start = time.time()
        result = ask_docs(question, model, collection)
        seconds = time.time() - start
        passed, reason = grade(result, right_pages)
        print(f"{number:2}. {'PASS' if passed else 'FAIL'}  {question}  "
              f"({reason}, {seconds:.1f} s)", flush=True)
        rows.append({
            "question": question, "should_answer": right_pages is not None,
            "passed": passed, "reason": reason, "seconds": round(seconds, 1),
            "answer": result["answer"], "sources": result["sources"],
            "reply": result["reply"],
        })

    answerable = [r for r in rows if r["should_answer"]]
    unanswerable = [r for r in rows if not r["should_answer"]]
    times = [r["seconds"] for r in rows]
    print(f"\nShould answer:      {sum(r['passed'] for r in answerable)} / {len(answerable)}")
    print(f"Should not answer:  {sum(r['passed'] for r in unanswerable)} / {len(unanswerable)}")
    print(f"Total:              {sum(r['passed'] for r in rows)} / {len(rows)}")
    print(f"Speed: average {sum(times) / len(times):.1f} s, "
          f"fastest {min(times):.1f} s, slowest {max(times):.1f} s")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Answers saved to {RESULTS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
