"""Build the message we send to the AI, and check its answer.

Step 10: the "exam paper" + a "show your proof" check.

Rules alone were not enough: the small 3B model still made up answers from
its own memory (e.g. about proxies). So the AI must first copy the sentence
from the notes that answers the question ("Quote: ..."), then answer
("Answer: ..."). Python then checks that the quote is really in the notes.
No real quote -> we reply "I don't know".

Run with:  uv run python -m rag_chatbot.prompt
(sends test questions with a hand-written note to the AI and gives a score)
"""

import re

from rag_chatbot.llm import chat

I_DONT_KNOW = "I don't know."

RULES = """You answer questions about the Python "requests" library, using ONLY the notes.
First, copy the sentence from the notes that answers the question, after "Quote:".
If no sentence in the notes answers it, write "Quote: none" and then "Answer: I don't know".
Otherwise write "Answer:" and a short answer based only on that quote."""


def build_messages(question: str, notes: list[dict]) -> list[dict]:
    """Put the rules, the notes and the question into messages for the AI.

    notes: chunks shaped like search() results (each has "text").
    """
    notes_text = "\n\n".join(
        f"Note {i}:\n{note['text']}" for i, note in enumerate(notes, start=1)
    )
    return [
        # "system" = instructions the AI should always follow.
        {"role": "system", "content": RULES},
        # "user" = the notes and the question for this turn.
        {"role": "user", "content": f"Notes:\n\n{notes_text}\n\nQuestion: {question}"},
    ]


def split_reply(reply: str) -> tuple[str, str]:
    """Split the AI's reply into (quote, answer). Missing parts become ""."""
    quote = re.search(r"Quote:\s*(.*?)\s*(?=Answer:|$)", reply, re.S)
    answer = re.search(r"Answer:\s*(.*)", reply, re.S)
    return (quote.group(1) if quote else ""), (answer.group(1).strip() if answer else "")


def squash(text: str) -> str:
    """Lowercase and turn all spaces/newlines into single spaces, for comparing."""
    return " ".join(text.lower().split())


def quote_is_real(quote: str, notes: list[dict]) -> bool:
    """True if the quote really appears (word for word) in one of the notes."""
    # Ignore quote marks / a trailing ":" or "." that the AI may add around it.
    quote = squash(quote).strip("\"' ").rstrip(":.")
    if not quote or quote == "none":
        return False
    return any(quote in squash(note["text"]) for note in notes)


def answer(question: str, notes: list[dict]) -> tuple[str, str]:
    """Ask the AI with the rules; return (final answer, the AI's raw reply)."""
    reply = chat(build_messages(question, notes))
    quote, text = split_reply(reply)
    # Sometimes the AI skips "Quote:" and copies the note straight into its
    # answer; then the answer itself is the proof.
    proof = quote or text
    if not quote_is_real(proof, notes) or not text:
        return I_DONT_KNOW, reply
    return text, reply


def main() -> None:
    """Test the rules with a hand-written note: some questions answerable, some not."""
    # A short note in the style of the real docs (written by hand for this test).
    notes = [{"text": (
        "You can tell Requests to stop waiting for a response after a given "
        "number of seconds with the timeout parameter:\n"
        "requests.get('https://github.com/', timeout=0.001)\n\n"
        "To send a file, pass a dict to the files parameter:\n"
        "requests.post(url, files={'file': open('report.xls', 'rb')})"
    )}]
    # (question, is the answer in the note?)
    tests = [
        ("How do I set a timeout?", True),
        ("How do I upload a file?", True),
        ("How do I use a proxy?", False),
        ("How do I use requests.turbo_get()?", False),
        ("How do I send cookies?", False),
        ("Who created requests?", False),
    ]

    print(f"Note given to the AI:\n{notes[0]['text']}\n")
    passed = 0
    for question, answerable in tests:
        final, reply = answer(question, notes)
        ok = (final != I_DONT_KNOW) == answerable
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {question}")
        print(f"      expected: {'an answer' if answerable else I_DONT_KNOW}")
        print(f"      AI wrote: {' '.join(reply.split())}")
        print(f"      final:    {' '.join(final.split())}\n")
    print(f"Score: {passed} / {len(tests)}")


if __name__ == "__main__":
    main()
