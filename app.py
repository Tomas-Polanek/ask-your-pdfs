"""Streamlit UI for asking questions about the ingested PDFs."""

import os

import streamlit as st

from rag import DEFAULT_TOP_K, answer_question, load_embedder, open_collection

st.set_page_config(page_title="Ask Your PDFs")
st.title("Ask Your PDFs")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return load_embedder()


try:
    collection = open_collection()
except Exception:
    st.error("No index found. Run `python ingest.py` first, then reload this page.")
    st.stop()

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set. Add it to a `.env` file (see `.env.example`).")
    st.stop()

top_k = st.sidebar.slider("Chunks to retrieve (top-k)", 1, 15, DEFAULT_TOP_K)
st.sidebar.caption(f"{collection.count()} chunks indexed")

with st.form("question_form"):
    question = st.text_input("Question", placeholder="What does the encoder consist of?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Searching and generating..."):
            answer, chunks = answer_question(question, collection, get_embedder(), top_k)
    except Exception as error:
        st.error(f"Could not generate an answer: {error}")
        st.stop()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved chunks")
    for chunk in chunks:
        with st.expander(f"{chunk.label} (similarity {chunk.similarity:.2f})"):
            st.write(chunk.text)
