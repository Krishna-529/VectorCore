"""End-to-end semantic search (RAG retrieval) demo for VectorCore.

Pipeline:
    text corpus  ->  sentence-transformers embeddings (float32)
                 ->  vectorcore.HnswIndex (cosine)
                 ->  top-k semantic neighbors for a natural-language query

This is the retrieval half of a Retrieval-Augmented Generation system: given a
question, find the most semantically relevant documents. It demonstrates the
"two-language" design — embeddings come from the Python ML ecosystem, the
nearest-neighbor search runs in the C++ engine via a zero-copy NumPy bridge.

Run:
    pip install .                       # build vectorcore
    pip install sentence-transformers   # embedding model (one-time)
    python examples/rag_demo.py
"""

from __future__ import annotations

import sys

import numpy as np

import vectorcore

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# A small knowledge base. Note the queries below use *different words* than the
# documents — only a semantic (not keyword) match can connect them.
CORPUS = [
    "The mitochondrion is the powerhouse of the cell, producing ATP.",
    "Photosynthesis converts sunlight, water, and CO2 into glucose in plants.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Python is a high-level programming language known for readability.",
    "Rust guarantees memory safety without a garbage collector.",
    "A vector database indexes embeddings for approximate nearest-neighbor search.",
    "Transformers use self-attention to model long-range dependencies in text.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Espresso is brewed by forcing hot water through finely-ground coffee.",
    "Dogs are domesticated descendants of wolves and are loyal companions.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Neural networks learn hierarchical features from labeled training data.",
    "The human heart pumps blood through the circulatory system.",
    "Quantum computers exploit superposition and entanglement for computation.",
]

QUERIES = [
    "How do cells generate energy?",
    "Which programming language is memory safe by design?",
    "Tell me about a famous landmark in France.",
    "What animal is descended from wolves?",
    "How does similarity search over embeddings work?",
]


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with sentence-transformers; return contiguous float32."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers is not installed.\n"
            "  pip install sentence-transformers   (CPU torch is sufficient)\n"
            "Then re-run this demo."
        )
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def main() -> None:
    print(f"vectorcore {vectorcore.__version__}")
    print(f"Embedding {len(CORPUS)} documents with {MODEL_NAME} ...")
    doc_vecs = embed(CORPUS)
    dim = doc_vecs.shape[1]
    print(f"  embedding dim = {dim}\n")

    # Cosine HNSW: VectorCore normalizes internally, so raw embeddings are fine.
    index = vectorcore.HnswIndex(dim=dim, M=16, metric="cosine", ef_construction=200)
    index.ef_search = 64
    index.add(doc_vecs, np.arange(len(CORPUS), dtype=np.uint64))
    print(f"Indexed {index.size} documents into HnswIndex (cosine, ef_search={index.ef_search})\n")

    # Batched query (uses the new (m, dim) HNSW search path).
    query_vecs = embed(QUERIES)
    k = 3
    ids, scores = index.search(query_vecs, k)

    for qi, question in enumerate(QUERIES):
        print(f"Q: {question}")
        for rank in range(k):
            doc_id = int(ids[qi, rank])
            sim = float(scores[qi, rank])
            print(f"   {rank + 1}. (cos={sim:.3f}) {CORPUS[doc_id]}")
        print()


if __name__ == "__main__":
    main()
