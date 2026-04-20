# services/scoring/confidence.py
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

@dataclass
class ConfidenceScore:
    volume: float       # 0-1
    diversity: float    # 0-1
    coherence: float    # 0-1
    recency: float      # 0-1

    @property
    def composite(self) -> float:
        return (self.volume * 0.30 + self.diversity * 0.25 +
                self.coherence * 0.25 + self.recency * 0.20)

    @property
    def tier(self) -> str:
        c = self.composite
        if c >= 0.75: return "HIGH"
        if c >= 0.55: return "MEDIUM"
        if c >= 0.35: return "LOW"
        return "INSUFFICIENT"

def calculate_confidence(retrieved_docs: list) -> ConfidenceScore:
    if not retrieved_docs:
        return ConfidenceScore(0, 0, 0, 0)
    volume   = min(len(retrieved_docs) / 30.0, 1.0)
    types    = [d.get("source_type", "") for d in retrieved_docs]
    diversity = min(len(set(types)) / 4.0, 1.0)
    sims     = [d.get("similarity", 0.5) for d in retrieved_docs]
    coherence = sum(sims) / len(sims)
    now = datetime.now(timezone.utc)
    ages = []
    for d in retrieved_docs:
        raw = d.get("created_at") or d.get("metadata", {}).get("created_at")
        if raw:
            try:
                dt = datetime.fromisoformat(str(raw))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ages.append((now - dt).days)
            except Exception:
                pass
    recency = max(0.0, 1.0 - (sum(ages)/len(ages)/180.0)) if ages else 0.5
    return ConfidenceScore(volume, diversity, coherence, recency)