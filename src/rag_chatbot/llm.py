"""Ask the local AI (llama3.2:3b in Ollama) a question from Python.

Step 9: test the AI on its own, with no docs yet. Later steps reuse ask().

Run with:  uv run python -m rag_chatbot.llm "How do I set a timeout in requests?"
"""

import sys
import time

import ollama

LLM_MODEL = "llama3.2:3b"

# num_gpu 0: never use the M3's GPU, CPU only (see CLAUDE.md).
# temperature 0: no "creativity", so the AI gives its most likely answer.
LLM_OPTIONS = {"num_gpu": 0, "temperature": 0}


def chat(messages: list[dict]) -> str:
    """Send a list of messages (e.g. rules + question) to the AI, return its answer."""
    response = ollama.chat(model=LLM_MODEL, messages=messages, options=LLM_OPTIONS)
    return response.message.content


def ask(question: str) -> str:
    """Send one plain question (no rules, no notes) to the AI and return its answer."""
    return chat([{"role": "user", "content": question}])


def main() -> None:
    """Ask one question and show the answer and how long it took."""
    question = " ".join(sys.argv[1:]) or "In one sentence, who are you?"

    start = time.time()
    answer = ask(question)
    seconds = time.time() - start

    print(f"\nQuestion: {question}\n")
    print(answer)
    print(f"\nTook {seconds:.1f} seconds")


if __name__ == "__main__":
    main()
