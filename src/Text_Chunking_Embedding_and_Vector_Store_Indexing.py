"""
Task-2 Pipeline - Chunking, Embedding & Vector-Store Indexing
=============================================================

This script turns the **82-K cleaned consumer-complaint narratives** into an
FAISS vector store ready for semantic search.

Key design choices (justify in report):
• **Chunk size / overlap**- 256/64 tokens.  ➜>90% of narratives fit in one
  chunk (median≈114tokens); longer texts are split with context retained.
• **Embedding model**- `sentence-transformers/all-MiniLM-L6-v2` (384-d, fast,
  strong STS score, tiny≈90MB).
• **Vector DB**- FAISS `IndexFlatIP` (cos-sim via inner-product) + separate
  `metadata.pkl` with complaint-ID & product for traceability.

Deliverables produced:
✓ `vector_store/faiss.index`- binary FAISS index
✓ `vector_store/metadata.pkl`- list-of-dicts with metadata

Run `python task2_vector_pipeline.py --help` for CLI flags.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# LangChain utilities ---------------------------------------------------------
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional FAISS import (separate pip: faiss‑cpu) -----------------------------
import faiss

# ---------------------------------------------------------------------------
# Configuration defaults (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 256  # tokens
DEFAULT_CHUNK_OVERLAP = 64  # tokens
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLEAN_COL = "Cleaned Narrative"  # text column in the CSV
default_csv_path = Path("data/processed/filtered_complaints.csv")
DEFAULT_SAVE_DIR = Path("vector_store")


# ---------------------------------------------------------------------------
# Pipeline helper functions
# ---------------------------------------------------------------------------

def load_clean_data(csv_path: Path | str) -> pd.DataFrame:
    """Load the pre‑cleaned CSV produced in Task 1."""
    print(f"[+] Loading cleaned data from: {csv_path}")
    return pd.read_csv(csv_path)


def filter_by_length(
    df: pd.DataFrame,
    min_tokens: int = 5,
    max_tokens: int = 1024,
    text_col: str = CLEAN_COL,
) -> pd.DataFrame:
    """Remove extremely short / long narratives (already done once, but guard).
    Args:
        min_tokens: below this ⇒ drop
        max_tokens: above this ⇒ drop
    """
    token_counts = df[text_col].astype(str).str.split().apply(len)
    mask = token_counts.between(min_tokens, max_tokens)
    removed = (~mask).sum()
    if removed:
        print(f"[!] Removed {removed:,} outlier narratives by length filter")
    return df.loc[mask].copy()


def chunk_documents(
    df: pd.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    text_col: str = CLEAN_COL,
) -> List[Document]:
    """Return a list of LangChain `Document` chunks with metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda txt: len(txt.split()),  # token‑ish
    )

    docs: List[Document] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        base_meta = {
            "complaint_id": row.get("Complaint ID", None),
            "product": row.get("Product", None),
        }
        chunks = splitter.split_text(str(row[text_col]))
        for i, chunk in enumerate(chunks):
            meta = base_meta | {"chunk_index": i}
            docs.append(Document(page_content=chunk, metadata=meta))
    print(f"[+] Produced {len(docs):,} chunks from {len(df):,} narratives")
    return docs


def embed_documents(
    docs: List[Document],
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute embeddings for each chunk."""
    model = SentenceTransformer(model_name)
    texts = [d.page_content for d in docs]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create cosine-similarity FAISS index (IP on L2-normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def persist_vector_store(
    index: faiss.Index,
    docs: List[Document],
    save_dir: Path = DEFAULT_SAVE_DIR,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_dir / "faiss.index"))
    metadata = [doc.metadata for doc in docs]
    with open(save_dir / "metadata.pkl", "wb") as fh:
        pickle.dump(metadata, fh)
    print(f"[✓] Vector store saved to {save_dir.resolve()}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def pipeline(
    csv_path: Path | str = default_csv_path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_tokens: int = 5,
    max_tokens: int = 1024,
    save_dir: Path | str = DEFAULT_SAVE_DIR,
):
    """Full Task-2 pipeline: load → filter → chunk → embed → index → persist."""
    save_dir = Path(save_dir)

    df = load_clean_data(csv_path)
    df = filter_by_length(df, min_tokens, max_tokens)
    docs = chunk_documents(df, chunk_size, chunk_overlap)

    emb = embed_documents(docs)
    index = build_faiss_index(emb)
    persist_vector_store(index, docs, save_dir)
    return index


# ---------------------------------------------------------------------------
# CLI Entry‑Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Chunk → Embed → Index")
    parser.add_argument("--csv_path", default=default_csv_path, help="Input CSV file")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--min_tokens", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR, help="Output dir for FAISS store")
    args = parser.parse_args()

    pipeline(
        csv_path=args.csv_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        save_dir=args.save_dir,
    )
