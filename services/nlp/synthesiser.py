# services/nlp/synthesiser.py
import anthropic
import os
import json
import re
from services.scoring.confidence import calculate_confidence


class FeatureSynthesiser:
    def __init__(self, embedder):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.embedder = embedder

    def generate_hypothesis(self, theme: str, okrs: list[str]) -> dict:
        docs = self.embedder.retrieve(theme, top_k=20)
        confidence = calculate_confidence(docs)

        if confidence.tier == "INSUFFICIENT":
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "theme": theme,
                "confidence_tier": "INSUFFICIENT",
                "confidence": vars(confidence)
            }

        context = "\n\n".join([
            f"[SRC-{i+1}] ({d['source_type']}, relevance={d['similarity']:.2f}):\n{d['content'][:350]}"
            for i, d in enumerate(docs[:12])
        ])

        prompt = f"""You are a rigorous product analyst. Use ONLY the provided evidence.

OKRs: {chr(10).join(okrs)}

EVIDENCE:
{context}

Analyse theme: "{theme}"

Rules:
- Cite [SRC-N] for every claim. No citation = no claim.
- Never invent numbers. Use only numbers from evidence.
- If evidence is weak, say so.

Respond ONLY in JSON with no extra text before or after:
{{
  "feature_name": "string (max 8 words)",
  "hypothesis": "string (max 60 words, with [SRC-N] citations)",
  "cited_sources": ["SRC-1", "SRC-2"],
  "source_types_used": ["review", "ticket"],
  "signal_volume": 0,
  "okr_alignment": "which OKR",
  "unsupported_claims": [],
  "analyst_note": "any uncertainty or caveats"
}}"""

        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = resp.content[0].text

        # Robustly extract JSON block
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
        except Exception:
            result = {
                "feature_name": theme,
                "hypothesis": raw,
                "cited_sources": [],
                "source_types_used": [],
                "signal_volume": len(docs)
            }

        result["confidence_tier"] = confidence.tier
        result["confidence"] = vars(confidence)
        result["retrieved_doc_count"] = len(docs)
        return result