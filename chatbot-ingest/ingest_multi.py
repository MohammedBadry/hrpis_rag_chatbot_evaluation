# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:28:04 2025

@author: Firas.Alhawari
"""

#!/usr/bin/env python3
"""
ingest_multi.py

Builds FAISS vector stores for multiple chunk sizes and a BM25 baseline index.
Saves metadata and chunk files for reproducible evaluation/ablation experiments.

Outputs (example):
  vectordb_faiss_250/        # FAISS index + metadata
  vectordb_faiss_500/
  vectordb_faiss_750/
  vectordb_bm25/             # BM25 tokenized corpus + metadata
  chunks_250.jsonl           # chunk list used to build indexes
  ingest_metadata.json       # summary of all created indexes

Requirements:
  pip install langchain faiss-cpu huggingface-hub rank_bm25 sentence-transformers
  (or alternative embedding providers supported by LangChain)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# BM25 - lightweight baseline (install: pip install rank_bm25)
from rank_bm25 import BM25Okapi

# -----------------------------
# Configurable parameters
# -----------------------------
DOCS_PATH = Path("./docs")
OUTPUT_DIR = Path("./vectordb_multi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZES = [250, 500, 750]        # ablation values (overlap fixed below)
CHUNK_OVERLAP = 50                   # fixed as requested

# Embedding models you might want to test (add/remove)
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # current
    "all-mpnet-base-v2",  # example alternative
]

# BM25 output dir
BM25_DIR = OUTPUT_DIR / "vectordb_bm25"
BM25_DIR.mkdir(exist_ok=True)

# Metadata output
METADATA_PATH = OUTPUT_DIR / "ingest_metadata.json"

# Safety checks
if not DOCS_PATH.exists() or not any(DOCS_PATH.iterdir()):
    raise SystemExit(f"Error: docs directory '{DOCS_PATH}' does not exist or is empty.")

# -----------------------------
# Helper functions
# -----------------------------
def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Load documents
# -----------------------------
print("Loading documents from:", DOCS_PATH)
loader = DirectoryLoader(str(DOCS_PATH), show_progress=True)
documents = loader.load()
print(f"Loaded {len(documents)} document objects.")

# Attach source filenames to metadata for traceability (used for citations)
for doc in documents:
    if "source" in doc.metadata:
        src_path = Path(doc.metadata["source"])
        doc.metadata["source"] = src_path.name  # keep only filename, not full path
    else:
        doc.metadata["source"] = "Unknown_Document"

# keep metadata for overall ingest
ingest_summary = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "doc_count": len(documents),
    "configs": [],
}

# -----------------------------
# Create chunk sets and indexes
# -----------------------------
for chunk_size in CHUNK_SIZES:
    print(f"\n--- Processing chunk_size={chunk_size}, overlap={CHUNK_OVERLAP} ---")
    # 1. split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f" -> Produced {len(chunks)} chunks.")

    # prepare a serializable chunk list (id, text, metadata)
    chunk_records = []
    for i, doc in enumerate(chunks):
        rec = {
            "id": f"chunk_{chunk_size}_{i}",
            "text": doc.page_content,
            "metadata": getattr(doc, "metadata", {}) or {}
        }
        chunk_records.append(rec)

    # save chunks to disk for reproducibility and for BM25 building
    chunks_jsonl = OUTPUT_DIR / f"chunks_{chunk_size}.jsonl"
    save_jsonl(chunks_jsonl, chunk_records)
    print(f" -> Saved chunks to {chunks_jsonl}")

    # For each embedding model, build a FAISS index (so you can ablate embeddings)
    for emb_model in EMBEDDING_MODELS:
        emb_name_safe = emb_model.replace("/", "_").replace(":", "_")
        out_dir_name = f"vectordb_faiss_{chunk_size}_{emb_name_safe}"
        out_dir = OUTPUT_DIR / out_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"   Building FAISS with embeddings='{emb_model}' -> {out_dir_name}")
        embedding = HuggingFaceEmbeddings(model_name=emb_model)

        # Build minimal LangChain Document objects from chunk_records
        from langchain.schema import Document
        docs_for_faiss = [Document(page_content=r["text"], metadata=r["metadata"]) for r in chunk_records]

        faiss_index = FAISS.from_documents(docs_for_faiss, embedding)
        faiss_index.save_local(str(out_dir))
        print(f"   -> FAISS index saved to {out_dir}")

        # Write metadata for this index
        idx_meta = {
            "index_dir": str(out_dir),
            "chunk_size": chunk_size,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": emb_model,
            "num_chunks": len(chunk_records),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        save_json(out_dir / "index_metadata.json", idx_meta)

        ingest_summary["configs"].append(idx_meta)

# -----------------------------
# Build BM25 baseline index (single copy using default chunk size)
# We'll build BM25 using the chunks for the default/favored chunk size (500)
# and also save tokenized corpus for reproducibility
# -----------------------------
default_chunk_size = CHUNK_SIZES[1] if len(CHUNK_SIZES) > 1 else CHUNK_SIZES[0]
chunks_file = OUTPUT_DIR / f"chunks_{default_chunk_size}.jsonl"
if not chunks_file.exists():
    raise SystemExit(f"Expected chunks file {chunks_file} not found.")

# load chunk texts
chunk_texts = []
chunk_meta = []
with open(chunks_file, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        chunk_texts.append(rec["text"])
        chunk_meta.append({"id": rec["id"], "metadata": rec.get("metadata", {})})

print(f"\nBuilding BM25 index from chunk file: {chunks_file} (count={len(chunk_texts)})")
# simple whitespace tokenization
tokenized_corpus = [t.lower().split() for t in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

# Save BM25 artifacts: tokenized corpus and mapping (we will not store the model object)
bm25_artifacts = {
    "chunk_ids": [m["id"] for m in chunk_meta],
    "num_chunks": len(chunk_texts),
    "created_at": datetime.utcnow().isoformat() + "Z"
}
# Save tokenized corpus as jsonl to allow rebuild
tokenized_path = BM25_DIR / "tokenized_corpus.jsonl"
save_jsonl(tokenized_path, [{"tokens": tokens} for tokens in tokenized_corpus])
save_json(BM25_DIR / "bm25_metadata.json", bm25_artifacts)
print(f" -> BM25 artifacts saved under {BM25_DIR}")

# Note: BM25 object is not pickled here; later evaluation script will rebuild BM25 via tokenized corpus.

# -----------------------------
# Save overall ingest metadata
# -----------------------------
ingest_summary["bm25"] = {
    "path": str(BM25_DIR),
    "chunk_source_file": str(chunks_file)
}
ingest_summary["note"] = (
    "Chunk overlap fixed at 50. Embedding models contained in EMBEDDING_MODELS list. "
    "Use the saved chunk files and index directories for reproducible evaluation and ablation studies. "
    "Evaluation scripts should run experiments for k in [5,10,15], chunk_size in [250,500,750], "
    "and compute per-question distributions, IC95%, and Cohen's kappa for inter-rater agreement."
)
ingest_summary["created_at"] = datetime.utcnow().isoformat() + "Z"

save_json(METADATA_PATH, ingest_summary)
print(f"\n✅ Ingest complete. Metadata saved to {METADATA_PATH}")