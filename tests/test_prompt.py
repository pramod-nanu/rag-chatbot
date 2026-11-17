"""Checks for the "show your proof" safety check in prompt.py.

No real AI is used: we hand-write pretend AI replies and check that Python
reacts correctly. Runs in about a second.

Run with:  uv run pytest
"""

import rag_chatbot.prompt as prompt
from rag_chatbot.prompt import I_DONT_KNOW, answer, find_sources, split_reply

QUICKSTART = "https://requests.readthedocs.io/en/latest/user/quickstart/"
ADVANCED = "https://requests.readthedocs.io/en/latest/user/advanced/"

# Two small hand-written notes, shaped like search() results.
NOTES = [
    {"url": QUICKSTART,
     "text": "You can tell Requests to stop waiting for a response after a given\n"
             "number of seconds with the timeout parameter."},
    {"url": ADVANCED,
     "text": "Requests supports proxies via the proxies argument."},
]


def fake_ai(reply):
    """Return a pretend chat() that always gives this reply (no real AI)."""
    return lambda messages: reply


# --- split_reply: cut the AI's reply into (quote, answer) ---

def test_split_reply_finds_quote_and_answer():
    assert split_reply("Quote: Some text. Answer: Do this.") == ("Some text.", "Do this.")


def test_split_reply_without_quote_label():
    assert split_reply("Answer: Do this.") == ("", "Do this.")


# --- find_sources: is the quote really in the notes, and on which page? ---

def test_real_quote_gives_its_page():
    assert find_sources("with the timeout parameter", NOTES) == [QUICKSTART]


def test_real_quote_ignores_line_breaks_and_capitals():
    quote = "YOU CAN TELL REQUESTS to stop waiting for a response after a given number"
    assert find_sources(quote, NOTES) == [QUICKSTART]


def test_made_up_quote_gives_no_page():
    # The fake proxy sentence the AI really invented in Step 10.
    assert find_sources("You can specify a proxy using the proxies parameter:", NOTES) == []


def test_quote_none_gives_no_page():
    assert find_sources("none", NOTES) == []
    assert find_sources("", NOTES) == []


def test_same_page_is_listed_once():
    notes = [NOTES[0], NOTES[0]]
    assert find_sources("the timeout parameter", notes) == [QUICKSTART]


# --- answer: the whole check, with a pretend AI ---

def test_answer_with_real_quote_is_kept(monkeypatch):
    monkeypatch.setattr(prompt, "chat", fake_ai(
        "Quote: with the timeout parameter. Answer: Use timeout=5."))
    final, _, sources = answer("How do I set a timeout?", NOTES)
    assert final == "Use timeout=5."
    assert sources == [QUICKSTART]


def test_answer_with_made_up_quote_becomes_i_dont_know(monkeypatch):
    monkeypatch.setattr(prompt, "chat", fake_ai(
        "Quote: Use requests.turbo_get() for speed. Answer: Call turbo_get()."))
    final, _, sources = answer("How do I use turbo_get()?", NOTES)
    assert final == I_DONT_KNOW
    assert sources == []


def test_answer_quote_none_becomes_i_dont_know(monkeypatch):
    monkeypatch.setattr(prompt, "chat", fake_ai("Quote: none Answer: I don't know"))
    final, _, sources = answer("What is the capital of France?", NOTES)
    assert final == I_DONT_KNOW
    assert sources == []


def test_answer_copied_from_note_counts_as_proof(monkeypatch):
    # Sometimes the AI skips "Quote:" and copies the note into its answer.
    monkeypatch.setattr(prompt, "chat", fake_ai(
        "Answer: Requests supports proxies via the proxies argument."))
    final, _, sources = answer("How do I use a proxy?", NOTES)
    assert final == "Requests supports proxies via the proxies argument."
    assert sources == [ADVANCED]


def test_answer_hides_note_numbers(monkeypatch):
    monkeypatch.setattr(prompt, "chat", fake_ai(
        "Quote: the proxies argument Answer: Use proxies, as shown in Note 2."))
    final, _, _ = answer("How do I use a proxy?", NOTES)
    assert final == "Use proxies, as shown in the docs."
