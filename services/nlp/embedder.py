# services/nlp/embedder.py
import chromadb
from chromadb.utils import embedding_functions
from services.ingestion.base import Signal
import os

class SignalEmbedder:
    """
    Local ChromaDB vector store with free sentence-transformers embeddings.
    No API keys, no hosting costs. Perfect for demo.
    """
    def __init__(self, persist_path: str = "./data/chroma_db"):
        # Persist to disk so demo data survives restarts
        self.client = chromadb.PersistentClient(path=persist_path)

        # Free local embeddings — no API key needed
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"   # 80MB, downloads once, runs locally
        )
        self.collection = self.client.get_or_create_collection(
            name="apde_signals",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_signals(self, signals: list[Signal]) -> int:
        """Embed and store signals. Skips duplicates."""
        existing_ids = set(self.collection.get()["ids"])
        new_signals = [s for s in signals if s.id not in existing_ids]
        if not new_signals:
            return 0

        self.collection.add(
            ids=[s.id for s in new_signals],
            documents=[s.content for s in new_signals],
            metadatas=[{
                "source_type": s.source_type,
                "created_at": s.created_at.isoformat(),
                **{k: str(v) for k, v in s.metadata.items()}
            } for s in new_signals],
        )
        return len(new_signals)

    def retrieve(self, query: str, top_k: int = 15,
             source_filter: list = None) -> list[dict]:
        """
        Semantic search with source diversity guarantee.
        Fetches top results overall PLUS top 3 from each
        source type to ensure multi-source representation.
        """
        total = self.collection.count()
        if total == 0:
            return []

        all_sources = ["review", "ticket", "sales", "market", "internal"]
        active_sources = source_filter if source_filter else all_sources

        collected = {}  # id -> doc, deduplication

        # ── Pass 1: broad search across all sources ────────────────────────
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, total),
                include=["documents", "metadatas", "distances"]
            )
            for i, doc_id in enumerate(results["ids"][0]):
                src = results["metadatas"][0][i].get("source_type", "unknown")
                if src in active_sources:
                    collected[doc_id] = {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "source_type": src,
                        "similarity": round(1 - results["distances"][0][i], 3),
                        "created_at": results["metadatas"][0][i].get("created_at"),
                    }
        except Exception as e:
            print(f"Pass 1 retrieval error: {e}")

        # ── Pass 2: per-source top-3 to guarantee diversity ─────────────────
        for source in active_sources:
            try:
                # Count how many docs exist for this source
                source_results = self.collection.get(
                    where={"source_type": source},
                    limit=1
                )
                if not source_results["ids"]:
                    continue  # no docs for this source, skip

                # Count total for this source type
                all_source = self.collection.get(
                    where={"source_type": source}
                )
                source_count = len(all_source["ids"])
                if source_count == 0:
                    continue

                per_source = self.collection.query(
                    query_texts=[query],
                    n_results=min(3, source_count),
                    where={"source_type": source},
                    include=["documents", "metadatas", "distances"]
                )
                for i, doc_id in enumerate(per_source["ids"][0]):
                    if doc_id not in collected:
                        collected[doc_id] = {
                            "id": doc_id,
                            "content": per_source["documents"][0][i],
                            "metadata": per_source["metadatas"][0][i],
                            "source_type": source,
                            "similarity": round(
                                1 - per_source["distances"][0][i], 3
                            ),
                            "created_at": per_source["metadatas"][0][i].get(
                                "created_at"
                            ),
                        }
            except Exception as e:
                print(f"Pass 2 error for {source}: {e}")
                continue

        # ── Sort by similarity and return top_k ────────────────────────────
        docs = sorted(
            collected.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]

        return docs

    def count(self) -> int:
        return self.collection.count()