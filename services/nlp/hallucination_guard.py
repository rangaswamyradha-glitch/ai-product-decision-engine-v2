# services/nlp/hallucination_guard.py
import re

class HallucinationGuard:
    def __init__(self, retrieved_docs: list[dict]):
        self.source_text = " ".join(d["content"] for d in retrieved_docs).lower()

    def verify(self, generated_text: str) -> dict:
        # 1. Check citations exist
        cited = re.findall(r'\[SRC-\d+\]', generated_text)
        has_citations = len(cited) > 0

        # 2. Check numbers are grounded
        numbers = re.findall(r'\b\d{2,}(?:,\d{3})*\b', generated_text)
        unverified_nums = [n for n in numbers
                           if n.replace(",","") not in self.source_text]

        # 3. Flag suspicious phrases
        hallucination_phrases = [
            "many users", "most customers", "majority of",
            "studies show", "research indicates", "it is well known"
        ]
        flagged = [p for p in hallucination_phrases
                   if p in generated_text.lower()]

        passed = has_citations and len(unverified_nums) == 0 and len(flagged) == 0
        return {
            "passed": passed,
            "status": "VERIFIED" if passed else "FLAGGED",
            "has_citations": has_citations,
            "unverified_numbers": unverified_nums,
            "flagged_phrases": flagged,
            "action": "proceed" if passed else "review_before_roadmap"
        }