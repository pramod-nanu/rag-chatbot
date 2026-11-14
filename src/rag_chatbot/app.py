"""A chat window in the browser for the requests docs chatbot.

Step 13: a small Streamlit page around ask_docs() (Step 11) and
format_answer() (Step 12). It runs only on this computer.

Run with:  uv run streamlit run src/rag_chatbot/app.py
then open http://localhost:8501 in the browser.
"""

import time

import streamlit as st

from rag_chatbot.embed import load_model
from rag_chatbot.rag import ask_docs, format_answer
from rag_chatbot.store import get_client, get_collection


# Streamlit re-runs this whole file on every message. cache_resource keeps the
# search model and the database loaded, so they load only once.
@st.cache_resource
def load_search():
    """Load the search model and open the Chroma collection (once)."""
    return load_model(), get_collection(get_client())


st.title("Requests Docs Chatbot")
st.caption(
    "Answers only from the requests documentation, with the source page. "
    "Runs locally on CPU, so an answer takes about 10-20 seconds."
)

model, collection = load_search()

# The conversation so far, kept between re-runs: a list of
# {"role": "user" or "assistant", "content": text}.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the earlier messages again (the page is redrawn on every re-run).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about requests..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            reply = format_answer(ask_docs(question, model, collection))
            seconds = time.time() - start
        st.markdown(reply)
        st.caption(f"{seconds:.1f} seconds")
    st.session_state.messages.append({"role": "assistant", "content": reply})
