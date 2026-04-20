# services/scoring/engine.py
import anthropic
import os
import json
import re


class ScoringEngine:
    def __init__(self, okrs: list[str]):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.okrs = okrs

    def score(self, hypothesis: dict) -> dict:
        prompt = f"""You are a product prioritisation expert. Score this feature.

Feature: {hypothesis.get('feature_name')}
Evidence: {hypothesis.get('hypothesis')}
Signal volume: {hypothesis.get('retrieved_doc_count', 0)}
Source types: {hypothesis.get('source_types_used', [])}
Confidence: {hypothesis.get('confidence_tier')}
OKRs: {self.okrs}

Score each dimension 1-10. Respond ONLY in JSON with no extra text before or after:
{{
  "reach": 5,
  "impact": 5,
  "confidence_score": 5,
  "effort": 5,
  "strategic_fit": 5,
  "effort_weeks": 4.0,
  "rationale": {{
    "reach": "explanation of reach score",
    "impact": "explanation with metric reference",
    "effort": "engineering effort explanation"
  }}
}}"""

        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
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

        data = json.loads(raw)

        # Compute composite score
        data["composite_score"] = round(
            data.get("reach", 5)            * 2.0 +
            data.get("impact", 5)           * 2.5 +
            data.get("confidence_score", 5) * 2.0 +
            data.get("effort", 5)           * 2.0 +
            data.get("strategic_fit", 5)    * 1.5,
            1
        )

        # SHAP-style breakdown
        data["shap"] = {
            "Reach":         round(data.get("reach", 5)            * 2.0, 1),
            "Impact":        round(data.get("impact", 5)           * 2.5, 1),
            "Confidence":    round(data.get("confidence_score", 5) * 2.0, 1),
            "Effort":        round(data.get("effort", 5)           * 2.0, 1),
            "Strategic Fit": round(data.get("strategic_fit", 5)    * 1.5, 1),
        }

        # Carry forward context from hypothesis
        data["feature_name"]    = hypothesis.get("feature_name")
        data["hypothesis"]      = hypothesis.get("hypothesis")
        data["confidence_tier"] = hypothesis.get("confidence_tier")
        data["okr_alignment"]   = hypothesis.get("okr_alignment")
        data["effort_weeks"]    = data.get("effort_weeks", 4.0)

        return data