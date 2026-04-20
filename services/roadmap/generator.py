# services/roadmap/generator.py
import anthropic
import os
import json
import re
import io
import csv


class RoadmapGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def generate(self, features: list[dict], okrs: list[str],
                 arr: float) -> dict:
        top = sorted(
            features,
            key=lambda x: x.get("composite_score", 0),
            reverse=True
        )[:5]

        summary = "\n".join([
            f"- {f['feature_name']}: Score {f.get('composite_score', 0):.0f}, "
            f"ROI ${f.get('roi_base', 0):,.0f}, "
            f"Effort {f.get('effort_weeks', '?')}wks, "
            f"Confidence {f.get('confidence_tier', '?')}"
            for f in top
        ])

        prompt = f"""You are a CPO writing an executive product strategy memo.

OKRs:
{chr(10).join(okrs)}

Top features ranked by evidence score:
{summary}

Company ARR: ${arr:,.0f}

Write a concise executive roadmap. Respond ONLY in JSON with no extra text before or after:
{{
  "q_now": "feature name for immediate sprint",
  "q_next": "feature name for next quarter",
  "q_later": "feature name for quarter after that",
  "exec_narrative": "200 word executive narrative explaining the roadmap logic, key risks, and expected business impact",
  "total_roi_base": 0.0,
  "top_risk": "biggest single risk to this roadmap",
  "human_gates_needed": [
    "what needs PM sign-off",
    "what needs Finance sign-off",
    "what needs Engineering sign-off"
  ]
}}"""

        resp = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = resp.content[0].text

        # Robustly extract JSON block
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    def to_csv(self, features: list[dict]) -> str:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=[
            "Feature", "Score", "Quarter", "Effort_wks",
            "ROI_base", "Confidence", "OKR", "PL_net"
        ])
        writer.writeheader()

        sorted_features = sorted(
            features,
            key=lambda x: x.get("composite_score", 0),
            reverse=True
        )

        for i, f in enumerate(sorted_features):
            quarter = (
                "Q2 2026" if i == 0 else
                "Q3 2026" if i == 1 else
                "Q4 2026"
            )
            writer.writerow({
                "Feature":    f.get("feature_name", ""),
                "Score":      f.get("composite_score", 0),
                "Quarter":    quarter,
                "Effort_wks": f.get("effort_weeks", ""),
                "ROI_base":   f.get("roi_base", 0),
                "Confidence": f.get("confidence_tier", ""),
                "OKR":        f.get("okr_alignment", ""),
                "PL_net":     f.get("roi_net", 0),
            })

        return out.getvalue()