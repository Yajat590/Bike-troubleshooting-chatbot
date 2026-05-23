"""Per-bike contextual hybrid retrieval.

Loads one bike's saved index and exposes a retriever that combines:
  - vector search   (semantic similarity — understands meaning)
  - BM25 search      (keyword match — catches exact part names / terms)
into a single hybrid retriever via reciprocal-rank fusion.

The BM25 half is rebuilt from the saved chunks each time a bike is loaded,
so it always stays in sync with the vector index.
"""
import logging

from llama_index.core import (StorageContext, Settings,
                              load_index_from_storage)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config

# We call Sarvam directly, so LlamaIndex's own LLM is disabled.
Settings.llm = None
logging.getLogger("bm25s").setLevel(logging.ERROR)

# The embedding model is loaded once and reused for every bike.
_embed_model = None


def get_embed_model():
    """Load the local embedding model once (cached for the process)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    return _embed_model


def load_retriever(bike_key: str):
    """Return a hybrid retriever for one bike.

    Raises FileNotFoundError if that bike's index has not been built yet.
    """
    persist_dir = config.STORAGE_DIR / bike_key
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No search index found for '{bike_key}'. "
            f"Build it first with:  python ingest.py"
        )

    embed_model = get_embed_model()

    # Load the saved vector index.
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    # Rebuild the BM25 keyword index from the same saved chunks.
    nodes = list(index.docstore.docs.values())

    vector_retriever = index.as_retriever(similarity_top_k=config.TOP_K)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=config.TOP_K)

    # Fuse the two retrievers. num_queries=1 means "no LLM query expansion"
    # — we have no LlamaIndex LLM and don't need it.
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=config.TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    return hybrid_retriever


def retrieve_context(retriever, query: str) -> str:
    """Run a hybrid search and return the matching manual text as one block.

    Each retrieved chunk is passed WHOLE — the detail that answers a question
    often sits in the middle or end of a chunk (after a heading), so chunks
    must not be trimmed. The prompt is kept bounded by TOP_K instead.
    """
    hits = retriever.retrieve(query)
    return "\n\n---\n\n".join(hit.node.get_content().strip()
                              for hit in hits if hit.node.get_content().strip())


def retrieve_context_multi(retriever, queries, max_chunks=None):
    """Search with multiple queries, deduplicate by node ID, return top results."""
    if max_chunks is None:
        max_chunks = config.AGENTIC_MAX_CHUNKS

    seen_ids = set()
    all_hits = []
    for query in queries:
        for hit in retriever.retrieve(query):
            nid = hit.node.node_id
            if nid not in seen_ids:
                seen_ids.add(nid)
                all_hits.append(hit)

    all_hits.sort(key=lambda h: h.score or 0, reverse=True)
    all_hits = all_hits[:max_chunks]

    return "\n\n---\n\n".join(hit.node.get_content().strip()
                              for hit in all_hits
                              if hit.node.get_content().strip())
