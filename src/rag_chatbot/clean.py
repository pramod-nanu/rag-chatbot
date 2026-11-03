"""Clean the scraped pages.

Step 4: read data/raw/pages.jsonl, remove the small bits of leftover junk,
and write data/clean/pages.jsonl. The raw file is never changed, so cleaning
can be re-run any time without scraping the site again.

Run with:  uv run python -m rag_chatbot.clean
"""

import json
import re
from pathlib import Path

from rag_chatbot.scrape import save_pages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "pages.jsonl"
CLEAN_FILE = PROJECT_ROOT / "data" / "clean" / "pages.jsonl"

# "Quickstart — Requests 2.34.2 documentation" -> "Quickstart".
# [\d.]+ matches any version number, so this keeps working after 2.34.2.
TITLE_SUFFIX = re.compile(r"\s*—\s*Requests [\d.]+ documentation$")

# Curly quotes -> straight quotes (what people type), and the non-breaking
# space -> a normal space. Code blocks don't use curly quotes, so they're safe.
CHARACTER_FIXES = str.maketrans({
    "’": "'",  # ’
    "‘": "'",  # ‘
    "“": '"',  # “
    "”": '"',  # ”
    " ": " ",  # non-breaking space
})


def load_pages(path: Path) -> list[dict]:
    """Read a JSONL file: one page (a dict) per line."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_title(title: str) -> str:
    """Drop the "— Requests X.Y.Z documentation" suffix and normalize quotes."""
    return TITLE_SUFFIX.sub("", title).translate(CHARACTER_FIXES)


def clean_text(text: str) -> str:
    """Remove "[source]" link labels and normalize quotes and spaces."""
    # On the website "[source]" was a link to the code; in text it's noise.
    text = text.replace("[source]", "")
    return text.translate(CHARACTER_FIXES)


def main() -> None:
    """Clean every raw page and save the result."""
    pages = load_pages(RAW_FILE)
    cleaned = [
        {"url": p["url"], "title": clean_title(p["title"]), "text": clean_text(p["text"])}
        for p in pages
    ]
    save_pages(cleaned, CLEAN_FILE)

    before = sum(len(p["text"]) for p in pages)
    after = sum(len(p["text"]) for p in cleaned)
    print(f"Cleaned {len(cleaned)} pages: {before:,} -> {after:,} characters of text.")
    print(f"Saved to {CLEAN_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
