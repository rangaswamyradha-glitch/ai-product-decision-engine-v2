# services/nlp/vector_store.py
"""
Lightweight vector store using numpy only.
No chromadb, no torch, no external dependencies.
Persists to a simple JSON file.
"""
import json
import math
import hashlib
from pathlib import Path
from services.ingestion.base import Signal


class LocalVectorStore:

    def __init__(self,
                 persist_path: str = "./data/vector_store.json"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Store as dict keyed by id — never gets out of sync
        self.store = {}  # id -> {doc fields + embedding}
        self._load()

    # ── Persistence ────────────────────────────────────────────────

    def _load(self):
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "r",
                          encoding="utf-8") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}

    def _save(self):
        with open(self.persist_path, "w",
                  encoding="utf-8") as f:
            json.dump(self.store, f)

    # ── Embedding ──────────────────────────────────────────────────

    def _embed(self, text: str, dim: int = 512) -> list[float]:
        """
        Keyword-weighted n-gram embedding.
        Includes synonym expansion for domain terms.
        """
        synonyms = {
            "export": [
                "export", "download", "csv", "excel", "bulk",
                "extract", "reporting", "report", "compliance",
                "audit", "finance", "data export", "bulk export",
            ],
            "sso": [
                "sso", "saml", "okta", "auth", "authentication",
                "login", "security", "enterprise", "identity",
                "single sign", "sign on", "it policy", "ldap",
                "active directory", "oauth", "provisioning",
                "okta integration", "saml integration",
            ],
            "slow": [
                "slow", "performance", "speed", "lag", "latency",
                "load", "loading", "fast", "optimise", "optimize",
                "timeout", "freeze", "hang", "unresponsive",
            ],
            "data": [
                "data", "loss", "save", "autosave", "disappear",
                "missing", "unsaved", "lost", "deleted", "gone",
                "switch", "project switch", "tab switch",
            ],
            "enterprise": [
                "enterprise", "compliance", "audit", "finance",
                "corporate", "b2b", "security", "it",
                "procurement", "contract", "renewal", "deal",
                "pipeline", "arr", "gartner", "forrester",
                "analyst", "benchmark", "industry",
            ],
            "market": [
                "market", "competitor", "gartner", "forrester",
                "analyst", "benchmark", "industry", "report",
                "g2", "category", "landscape", "competitive",
            ],
        }

        text_lower = text.lower()
        expansions = []
        for group, words in synonyms.items():
            if any(w in text_lower for w in words):
                expansions.extend(words)
        if expansions:
            text_lower = text_lower + " " + " ".join(expansions)

        vec = [0.0] * dim

        # Character n-grams (2,3,4)
        for n in [2, 3, 4]:
            for i in range(len(text_lower) - n + 1):
                gram = text_lower[i:i + n]
                h = int(hashlib.md5(
                    gram.encode()).hexdigest(), 16)
                vec[h % dim] += 1.0

        # Word unigrams — highest weight
        words_list = text_lower.split()
        for word in words_list:
            h = int(hashlib.md5(
                word.encode()).hexdigest(), 16)
            vec[h % dim] += 3.0

        # Word bigrams
        for i in range(len(words_list) - 1):
            bigram = words_list[i] + " " + words_list[i + 1]
            h = int(hashlib.md5(
                bigram.encode()).hexdigest(), 16)
            vec[h % dim] += 2.0

        # Normalise to unit length
        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0:
            vec = [x / magnitude for x in vec]

        return vec

    def _cosine_sim(self, a: list[float],
                    b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    # ── CRUD ───────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self.store)

    def get_ids(self) -> set:
        return set(self.store.keys())

    def clear(self):
        self.store = {}
        self._save()

    def add(self, signals: list[Signal]) -> int:
        existing = self.get_ids()
        new = [s for s in signals if s.id not in existing]
        if not new:
            return 0
        for s in new:
            self.store[s.id] = {
                "id":          s.id,
                "content":     s.content,
                "source_type": s.source_type,
                "created_at":  s.created_at.isoformat(),
                "metadata":    {k: str(v)
                                for k, v in s.metadata.items()},
                "embedding":   self._embed(s.content),
            }
        self._save()
        return len(new)

    # ── Query ──────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 15,
              source_filter: list = None) -> list[dict]:
        """
        Three-pass retrieval with guaranteed source diversity.
        Pass 1 — broad similarity across all sources.
        Pass 2 — top 5 per source type.
        Pass 3 — force at least 2 from market and internal.
        """
        if not self.store:
            return []

        all_sources = ["review", "ticket", "sales",
                       "market", "internal"]
        active = source_filter if source_filter else all_sources

        query_emb = self._embed(query_text)

        # Score every document — id keyed so no index mismatch
        def score_doc(doc_id):
            doc = self.store[doc_id]
            if doc["source_type"] not in active:
                return None
            sim = self._cosine_sim(
                query_emb, doc["embedding"]
            )
            return {
                "id":          doc["id"],
                "content":     doc["content"],
                "source_type": doc["source_type"],
                "metadata":    doc["metadata"],
                "created_at":  doc["created_at"],
                "similarity":  round(sim, 4),
            }

        scored = []
        for doc_id in self.store:
            result = score_doc(doc_id)
            if result:
                scored.append(result)

        scored.sort(key=lambda x: x["similarity"], reverse=True)

        collected = {}

        # ── Pass 1: top-k overall ──────────────────────────────────
        for doc in scored[:top_k * 2]:
            collected[doc["id"]] = doc

        # ── Pass 2: top-5 per source type ─────────────────────────
        for source in active:
            src_docs = [d for d in scored
                        if d["source_type"] == source]
            for doc in src_docs[:5]:
                if doc["id"] not in collected:
                    collected[doc["id"]] = doc

        # ── Pass 3: force market and internal ─────────────────────
        # These use formal language and score lower on similarity
        # but must always be represented in results
        for source in ["market", "internal"]:
            if source not in active:
                continue

            # Get all docs for this source directly from store
            src_docs = [
                {
                    "id":          self.store[did]["id"],
                    "content":     self.store[did]["content"],
                    "source_type": self.store[did]["source_type"],
                    "metadata":    self.store[did]["metadata"],
                    "created_at":  self.store[did]["created_at"],
                    "similarity":  round(self._cosine_sim(
                        query_emb,
                        self.store[did]["embedding"]
                    ), 4),
                }
                for did in self.store
                if self.store[did]["source_type"] == source
            ]

            if not src_docs:
                continue

            src_docs.sort(
                key=lambda x: x["similarity"], reverse=True
            )

            # Force top 3 from this source into results
            for doc in src_docs[:3]:
                doc_copy = dict(doc)
                # Ensure minimum visibility score
                doc_copy["similarity"] = max(
                    doc["similarity"], 0.20
                )
                collected[doc_copy["id"]] = doc_copy

        # ── Final sort and return ──────────────────────────────────
        results = sorted(
            collected.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]
        return results
    def query_by_theme(self, theme_keyword: str,
                    top_k: int = 5) -> list[dict]:
        """
        Retrieve signals by theme tag.
        Checks metadata at both top level and nested.
        """
        theme_map = {
            "sso":         "sso_auth",
            "auth":        "sso_auth",
            "saml":        "sso_auth",
            "okta":        "sso_auth",
            "sign":        "sso_auth",
            "export":      "bulk_export",
            "bulk":        "bulk_export",
            "csv":         "bulk_export",
            "compliance":  "bulk_export",
            "reporting":   "bulk_export",
            "data":        "data_loss",
            "loss":        "data_loss",
            "switch":      "data_loss",
            "autosave":    "data_loss",
            "slow":        "slow_dashboard",
            "performance": "slow_dashboard",
            "dashboard":   "slow_dashboard",
            "loading":     "slow_dashboard",
            "speed":       "slow_dashboard",
        }

        query_lower = theme_keyword.lower()
        theme_tag = None
        for keyword, tag in theme_map.items():
            if keyword in query_lower:
                theme_tag = tag
                break

        if not theme_tag:
            return []

        matched = []
        for doc_id, doc in self.store.items():
            # Check theme in multiple possible locations
            meta = doc.get("metadata", {})
        
            # Location 1: nested metadata dict
            doc_theme = meta.get("theme", "")
        
            # Location 2: top level (fallback)
            if not doc_theme:
                doc_theme = doc.get("theme", "")

            if doc_theme == theme_tag:
                matched.append({
                    "id":          doc["id"],
                    "content":     doc["content"],
                    "source_type": doc["source_type"],
                    "metadata":    meta,
                    "created_at":  doc["created_at"],
                    "similarity":  0.75,
                })

        # Sort: prioritise market and internal first
        # Separate by source type priority
        market_docs   = [d for d in matched if d["source_type"] == "market"]
        internal_docs = [d for d in matched if d["source_type"] == "internal"]
        other_docs    = [d for d in matched
                        if d["source_type"] not in ["market", "internal"]]

        import random
        random.shuffle(other_docs)

        # Always include at least 2 market + 2 internal + rest up to top_k
        guaranteed = market_docs[:2] + internal_docs[:2]
        remaining  = [d for d in (market_docs[2:] + internal_docs[2:] + other_docs)
                    if d["id"] not in {g["id"] for g in guaranteed}]

        return (guaranteed + remaining)[:max(top_k, len(guaranteed))]

        import random
        # Keep market/internal at top, shuffle the rest
        priority = [d for d in matched
                    if d["source_type"] in ["market", "internal"]]
        others = [d for d in matched
                if d["source_type"] not in ["market", "internal"]]
        random.shuffle(others)

        return (priority + others)[:top_k]