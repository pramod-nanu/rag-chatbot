"""Split the cleaned pages into chunks.

Step 5: read data/clean/pages.jsonl, cut each page into ~400-token chunks
with a 50-token overlap, put the page title at the start of each chunk,
and write data/chunks/chunks.jsonl.

Run with:  uv run python -m rag_chatbot.chunk
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from rag_chatbot.clean import load_pages
from rag_chatbot.scrape import ALLOWED_PREFIX, save_pages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_FILE = PROJECT_ROOT / "data" / "clean" / "pages.jsonl"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_MAX_TOKENS = 512  # bge-small silently ignores anything longer
CHUNK_SIZE = 400  # tokens of page text per chunk (before the title is added)
CHUNK_OVERLAP = 50  # tokens repeated from the end of the previous chunk


def load_tokenizer() -> Tokenizer:
    """Load bge-small's own tokenizer, so we count tokens exactly like the model."""
    tokenizer = Tokenizer.from_pretrained(EMBEDDING_MODEL)
    # The saved tokenizer cuts text at 512 tokens; turn that off, or every
    # long text would be counted as 512 and we couldn't spot oversized chunks.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def page_id(url: str) -> str:
    """Short readable name for a page: ".../en/latest/user/quickstart/" -> "user/quickstart/"."""
    return url.removeprefix(ALLOWED_PREFIX) or "home"


def main() -> None:
    """Chunk every cleaned page and save the chunks."""
    tokenizer = load_tokenizer()

    def count_tokens(text: str) -> int:
        # add_special_tokens=False: count only the text itself; the model's
        # 2 extra markers ([CLS] and [SEP]) are checked separately below.
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    # Tries to cut between paragraphs first, then lines, then words, and
    # measures every piece with count_tokens instead of counting characters.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page in load_pages(CLEAN_FILE):
        for i, piece in enumerate(splitter.split_text(page["text"])):
            # The title tells the search which page this chunk comes from.
            # A page's first chunk already starts with its heading, which is
            # usually the title, so don't add it twice.
            if piece.startswith(page["title"]):
                text = piece
            else:
                text = f"{page['title']}\n\n{piece}"
            chunks.append({
                "id": f"{page_id(page['url'])}#{i}",
                "url": page["url"],
                "title": page["title"],
                "chunk_index": i,
                "n_tokens": count_tokens(text),
                "text": text,
            })

    save_pages(chunks, CHUNKS_FILE)

    sizes = [c["n_tokens"] for c in chunks]
    print(f"Created {len(chunks)} chunks from {len({c['url'] for c in chunks})} pages.")
    print(f"Tokens per chunk: min {min(sizes)}, average {sum(sizes) // len(sizes)}, max {max(sizes)}")
    # +2 for [CLS] and [SEP], which the model adds to every text.
    too_long = [c["id"] for c in chunks if c["n_tokens"] + 2 > MODEL_MAX_TOKENS]
    print(f"Chunks over the {MODEL_MAX_TOKENS}-token model limit: {len(too_long)} {too_long}")
    print(f"Saved to {CHUNKS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
