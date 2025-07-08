"""
Streamlit RAG Chatbot with Streaming & Source Display
====================================================
Run with:
    streamlit run streamlit_app.py

Requirements:
    pip install streamlit openai tiktoken  # plus your existing deps

Environment:
    export OPENAI_API_KEY="sk..."
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import os 
import sys
sys.path.append("..")
from src import rag_pipeline
from src.rag_pipeline import RAGPipeline

###############################################################################
# Configuration
###############################################################################

st.set_page_config(
    page_title="RAG Complaint Assistant",
    page_icon="🤖",
    layout="wide",
)

# -- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    openai_model = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )
    k_val = st.slider("Top‑k retrieved chunks", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        "**How it works**: We embed your question, retrieve the most relevant complaint "
        "chunks from a FAISS vector store, then feed them plus your question to an LLM."
    )

###############################################################################
# Load RAG pipeline (cache to avoid re‑loading each run)
###############################################################################

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_rag(model_name: str):
    return RAGPipeline(
        "vector_store/faiss.index",
        "vector_store/metadata.pkl",
        generator_backend="openai",
        openai_model=model_name,
    )

rag = load_rag(openai_model)

###############################################################################
# Session state for chat history
###############################################################################
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str, List[str]]] = []  # (user, bot, sources)

###############################################################################
# Helper: stream answer token‑by‑token
###############################################################################

def stream_answer(answer: str, container):
    words = answer.split()
    full = ""
    for w in words:
        full += w + " "
        container.markdown(full + "▌")
        time.sleep(0.03)  # typing illusion
    container.markdown(full)
    return full

###############################################################################
# Chat UI
###############################################################################

st.title("💬 Intelligent Complaint Analysis Chatbot")
chat_container = st.container()

# Display chat history
for user_msg, bot_msg, srcs in st.session_state.history:
    with chat_container.chat_message("user"):
        st.markdown(user_msg)
    with chat_container.chat_message("assistant"):
        st.markdown(bot_msg)
        if srcs:
            with st.expander("Sources"):
                for s in srcs:
                    st.markdown(f"- {s}")

# Chat input
prompt = st.chat_input("Type your question and hit Enter…")
if prompt:
    # Show user message immediately
    with chat_container.chat_message("user"):
        st.markdown(prompt)

    # Prepare assistant message placeholder
    with chat_container.chat_message("assistant"):
        answer_placeholder = st.empty()
        source_placeholder = st.empty()

    # Run RAG
    retrieved_answer, retrieved = rag.run(prompt, k=k_val)

    # Stream the answer
    final_answer = stream_answer(retrieved_answer, answer_placeholder)

    # Format sources (first 2 chunks)
    src_texts = [
        (chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk))[:300] + "…"
        for chunk, _ in retrieved[:2]
    ]
    source_placeholder.markdown("\n".join(f"🔹 {s}" for s in src_texts))

    # Append to history
    st.session_state.history.append((prompt, final_answer, src_texts))

###############################################################################
# Clear chat button
###############################################################################

if st.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
