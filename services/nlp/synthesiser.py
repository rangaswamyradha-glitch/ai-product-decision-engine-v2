# services/nlp/synthesiser.py
import anthropic
import os
import json
import re
from services.scoring.confidence import calculate_confidence


class FeatureSynthesiser:
    def __init__(self, embedder):
        import os
        from dotenv import load_dotenv
        load_dotenv()

        # Try multiple ways to get the API key
        api_key = os.getenv("ANTHROPIC_API_KEY")

        # Fallback to streamlit secrets if running in cloud
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                pass

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Check your .env file or Streamlit secrets."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.embedder = embedder

    def generate_hypothesis(self, theme: str, okrs: list[str]) -> dict:
        docs = self.embedder.retrieve(theme, top_k=30)

        # Add theme-based docs for market + internal coverage
        theme_docs = self.embedder.query_by_theme(theme, top_k=15)
        all_results = {d["id"]: d for d in docs}
        for d in theme_docs:
            if d["id"] not in all_results:
                all_results[d["id"]] = d

        all_docs_list = list(all_results.values())
        all_source_types = list(set(
            d["source_type"] for d in all_docs_list
        ))
        confidence = calculate_confidence(docs, all_source_types)

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
        result["confidence"] = {
            "volume":    confidence.volume,
            "diversity": confidence.diversity,
            "coherence": confidence.coherence,
            "recency":   confidence.recency,
            "composite": confidence.composite,
        }
        result["retrieved_doc_count"] = len(all_docs_list)
        result["source_types_used"] = all_source_types
        return result