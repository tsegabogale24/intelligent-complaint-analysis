"""
rag_pipeline.py
================
Enhanced RAG pipeline with optional OpenAI GPT model support.

Author: Tsega Bogale - updated July-2025
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Literal, Any

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

# Optional imports
try:
    import openai  # type: ignore
except ImportError:  # The user might not need OpenAI support
    openai = None  # type: ignore

try:
    from transformers import pipeline as hf_pipeline, Pipeline  # type: ignore
except ImportError:
    hf_pipeline = None  # type: ignore
    Pipeline = Any  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PROMPT = (
    "You are a financial analyst assistant for CrediTrust. "
    "Your task is to answer questions about customer complaints. "
    "Use the following retrieved complaint excerpts to formulate your answer. "
    "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """FAISS-based retrieval with pluggable generation backends (OpenAI GPT or HF)."""

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        prompt_template: str = DEFAULT_PROMPT,
        generator_backend: Literal["openai", "hf"] = "openai",
        # openai‑specific
        openai_model: str = "gpt-4o-mini",
        # HF‑specific
        hf_model: str | None = "google/gemma-2b-it",
        hf_max_new_tokens: int = 256,
        hf_temperature: float = 0.2,
    ) -> None:
        self.prompt_template = prompt_template
        self.generator_backend = generator_backend

        # Load embedding model
        self.embedder = SentenceTransformer(embed_model_name)

        # Load FAISS index
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        # Load chunk texts / metadata
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.suffix == ".pkl":
            self.chunks: List[str] = _load_pickle(self.metadata_path)
        else:
            self.chunks = json.loads(self.metadata_path.read_text())

        # Configure generator
        if generator_backend == "openai":
            if openai is None:
                raise ImportError("openai package not installed - `pip install openai`. ")
            if os.getenv("OPENAI_API_KEY") is None:
                raise EnvironmentError(
                    "Set the OPENAI_API_KEY environment variable to use the OpenAI backend."
                )
            self.openai_model = openai_model
            # Nothing else to init
        else:  # HF pipeline
            if hf_pipeline is None:
                raise ImportError("transformers not installed - `pip install transformers`. ")
            model_name = hf_model or "google/gemma-2b-it"
            self.hf_pipe: Pipeline = hf_pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=hf_max_new_tokens,
                do_sample=True,
                temperature=hf_temperature,
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, k: int = 5) -> List[Tuple[str, float]]:
        q_vec = self.embedder.encode([question], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(q_vec, k)
        retrieved: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            retrieved.append((self.chunks[idx], float(dist)))
        return retrieved

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_openai(self, prompt: str) -> str:
        resp = openai.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    def _generate_hf(self, prompt: str) -> str:
        out = self.hf_pipe(prompt)
        if isinstance(out, list) and isinstance(out[0], dict):
            return out[0]["generated_text"].replace(prompt, "").strip()
        return str(out)

    def generate(self, question: str, k: int = 5) -> str:
        retrieved = self.retrieve(question, k)
        context = "\n---\n".join(
            chunk["text"] if isinstance(chunk, dict)and "text" in chunk else str(chunk)
            for chunk, _ in retrieved           
         )
        prompt = self.prompt_template.format(context=context, question=question)
        if self.generator_backend == "openai":
            return self._generate_openai(prompt)
        return self._generate_hf(prompt)

    # Convenience wrapper
    def run(self, question: str, k: int = 5):
        retrieved = self.retrieve(question, k)
        answer = self.generate(question, k)
        return answer, retrieved


# ---------------------------------------------------------------------------
# Markdown Evaluation Table helper
# ---------------------------------------------------------------------------

def create_evaluation_table(
    questions: List[str],
    rag: RAGPipeline,
    k: int = 5,
    out_path: str | Path = "evaluation.md",
) -> Path:
    rows = []
    for q in questions:
        ans, retrieved = rag.run(q, k)
        src = "<br>".join(chunk["text"][:120].replace("\n", " ") + "…" for chunk, _ in retrieved[:2])
        rows.append(f"| {q.replace('|', '\\|')} | {ans.replace('|', '\\|')} | {src.replace('|', '\\|')} |  |  |")
    header = (
        "| Question | Generated Answer | Retrieved Sources (top‑2) | Quality Score (1‑5) | Comments |\n"
        "| --- | --- | --- | --- | --- |"
    )
    table_md = header + "\n" + "\n".join(rows)
    out_path = Path(out_path)
    out_path.write_text(table_md, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI demo (python rag_pipeline.py demo)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import textwrap

    if len(sys.argv) >= 2 and sys.argv[1] == "demo":
        INDEX = "vector_store/faiss.index"
        META = "vector_store/metadata.pkl"
        rag = RAGPipeline(INDEX, META, generator_backend="openai")
        print("Loaded RAG pipeline. Ask a question (blank to quit).\n")
        while True:
            try:
                question = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            answer, retrieved = rag.run(question)
            print("\nAnswer:\n", textwrap.fill(answer, 90))
            print("\nTop sources:")
            for i, (chunk, score) in enumerate(retrieved, 1):
                print(f"[{i}] {chunk[:140].replace('\n', ' ')}…  (sim={score:.3f})")
            print("\n---\n")
    else:
        print("Usage: python rag_pipeline.py demo")
