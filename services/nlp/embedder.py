# services/nlp/embedder.py
from services.nlp.vector_store import LocalVectorStore
from services.ingestion.base import Signal


class SignalEmbedder:
    """
    Thin wrapper around LocalVectorStore.
    Maintains the same interface the rest of the app expects.
    """

    def __init__(self,
                 persist_path: str = "./data/vector_store.json"):
        self.store = LocalVectorStore(persist_path)

    def ingest_signals(self, signals: list[Signal]) -> int:
        return self.store.add(signals)

    def retrieve(self, query: str, top_k: int = 15,
                 source_filter: list = None) -> list[dict]:
        return self.store.query(query, top_k, source_filter)

    def count(self) -> int:
        return self.store.count()

    def clear(self):
        self.store.clear()
    def query_by_theme(self, theme_keyword: str,
                    top_k: int = 5) -> list[dict]:
        return self.store.query_by_theme(theme_keyword, top_k)
    # Keep collection attribute for any legacy references
    @property
    def collection(self):
        return self