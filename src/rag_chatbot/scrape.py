"""Scrape the requests documentation.

Step 3d: crawl every page under /en/latest/, starting from the docs home
page, and save the clean text of each page to data/raw/pages.jsonl.

Run with:  uv run python -m rag_chatbot.scrape
"""

import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

START_URL = "https://requests.readthedocs.io/en/latest/"
# Only follow links that start with this (skips /en/stable/, GitHub, etc.).
ALLOWED_PREFIX = "https://requests.readthedocs.io/en/latest/"
ROBOTS_URL = "https://requests.readthedocs.io/robots.txt"

# Sphinx helper pages that aren't documentation text: static files, raw
# sources, the term index and the JavaScript-only search page.
# /_modules/ is the library's Python source code ("[source]" links): it was
# 50% of all text and would crowd out the explained docs in retrieval.
SKIP_PATTERNS = (
    "/_static/", "/_sources/", "/_images/", "/genindex/", "/search/", "/py-modindex/",
    "/_modules/",
)

# Honest bot identity, as required by our scraping rules (see CLAUDE.md).
USER_AGENT = "rag-chatbot-learning-project (contact: https://github.com/pramod-nanu)"
DELAY_SECONDS = 1.0  # max 1 request per second
MAX_PAGES = 100  # safety stop in case something loops

# src/rag_chatbot/scrape.py -> parents[2] is the project root, so the file
# lands in the same place no matter which folder you run the command from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "pages.jsonl"


def load_robots() -> RobotFileParser:
    """Download robots.txt (with our User-Agent) and parse its rules."""
    response = requests.get(ROBOTS_URL, headers={"User-Agent": USER_AGENT}, timeout=10)
    response.raise_for_status()
    robots = RobotFileParser()
    robots.parse(response.text.splitlines())
    return robots


def normalize_url(url: str) -> str:
    """Remove #anchors and ?queries so one page has exactly one URL."""
    url, _fragment = urldefrag(url)
    scheme, netloc, path, _query, _ = urlsplit(url)
    return urlunsplit((scheme, netloc, path, "", ""))


def is_in_scope(url: str) -> bool:
    """True if the crawler is allowed to follow this link."""
    return url.startswith(ALLOWED_PREFIX) and not any(p in url for p in SKIP_PATTERNS)


def fetch_html(url: str) -> str | None:
    """Download a page. Return its HTML, or None if it failed or isn't HTML."""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        # One bad page (404, timeout, ...) shouldn't stop the whole crawl.
        print(f"    skipped: {error}")
        return None

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        print(f"    skipped: not HTML ({content_type})")
        return None
    return response.text


def extract_page(soup: BeautifulSoup) -> tuple[str, str]:
    """Return (title, main_text) from a Sphinx / Read the Docs page."""
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Sphinx puts the article in <div role="main">; the sidebar and footer
    # live outside it. Fall back to <body> if a page is built differently.
    article = soup.find("div", attrs={"role": "main"})
    if article is None:
        print("    WARNING: no <div role='main'> found, using <body> instead")
        article = soup.body

    # Sphinx adds a "¶" link after every heading; drop those links.
    for link in article.find_all("a", class_="headerlink"):
        link.decompose()

    # No separator: keep the page's own line breaks, so code blocks keep
    # their real lines and inline tags (<span>, <a>) join normally.
    text = article.get_text()
    # Collapse runs of blank lines into one, and trim trailing spaces.
    # Only rstrip: leading spaces are Python indentation in code blocks.
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return title, text


def extract_links(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Return the full, normalized, in-scope URLs of all links on a page."""
    links = []
    # Search the whole page, not just the article: the sidebar menu is
    # where most links to other pages are.
    for a in soup.find_all("a", href=True):
        # urljoin turns relative links like "../api/" into full URLs.
        url = normalize_url(urljoin(page_url, a["href"]))
        if is_in_scope(url):
            links.append(url)
    return links


def crawl() -> list[dict]:
    """Visit every in-scope page, breadth-first, and return the pages."""
    robots = load_robots()
    time.sleep(DELAY_SECONDS)

    queue = deque([START_URL])  # pages waiting to be visited
    seen = {START_URL}  # every URL ever added to the queue, so none repeats
    pages = []

    while queue and len(pages) < MAX_PAGES:
        url = queue.popleft()  # oldest first = breadth-first

        if not robots.can_fetch(USER_AGENT, url):
            print(f"    blocked by robots.txt: {url}")
            continue

        print(f"[{len(pages) + 1}] {url}")
        html = fetch_html(url)
        time.sleep(DELAY_SECONDS)  # be polite, even after a failed request
        if html is None:
            continue

        # Parse once; extract_links runs first because extract_page
        # removes the "¶" links from the tree.
        soup = BeautifulSoup(html, "html.parser")
        for link in extract_links(soup, url):
            if link not in seen:
                seen.add(link)
                queue.append(link)

        title, text = extract_page(soup)
        pages.append({"url": url, "title": title, "text": text})

    if queue:
        print(f"Stopped at MAX_PAGES={MAX_PAGES}; {len(queue)} URLs were not visited.")
    return pages


def save_pages(pages: list[dict], path: Path) -> None:
    """Write pages to a JSONL file: one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # "w" replaces the file on every run, so re-running never duplicates pages.
    with path.open("w", encoding="utf-8") as f:
        for page in pages:
            # ensure_ascii=False keeps characters like ’ and — readable.
            f.write(json.dumps(page, ensure_ascii=False) + "\n")


def main() -> None:
    """Crawl the docs and save all pages."""
    start = time.time()
    pages = crawl()
    save_pages(pages, OUTPUT_FILE)

    total_chars = sum(len(p["text"]) for p in pages)
    print(f"Done: {len(pages)} pages, {total_chars:,} characters of text, "
          f"in {time.time() - start:.0f} seconds.")
    print(f"Saved to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
