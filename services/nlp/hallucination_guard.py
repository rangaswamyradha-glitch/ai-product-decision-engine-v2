# services/nlp/hallucination_guard.py
import re


class HallucinationGuard:
    def __init__(self, retrieved_docs: list[dict]):
        self.source_text = " ".join(
            d["content"] for d in retrieved_docs
        ).lower()

    def verify_citations(self, generated_text: str) -> dict:
        """Check that [SRC-N] citations exist in the text."""
        cited = re.findall(r'\[SRC-\d+\]', generated_text)
        has_citations = len(cited) > 0
        return {
            "has_citations": has_citations,
            "cited": cited,
            "missing": [],
        }

    def verify_numbers(self, generated_text: str) -> dict:
        """
        Check that specific data numbers in the output
        can be found in retrieved source documents.
        Ignores source reference numbers like [SRC-N].
        """
        # Remove SRC citation references before checking numbers
        # so [SRC-11] doesn't get flagged as the number 11
        text_clean = re.sub(r'\[SRC-\d+\]', '', generated_text)

        # Only check numbers that look like real data:
        # dollar amounts, percentages, or 4+ digit numbers
        numbers = re.findall(
            r'\$[\d,]+|\d+%|\b\d{4,}\b', text_clean
        )

        source_text_clean = " ".join(self.source_text.split())
        unverified = [
            n for n in numbers
            if n.replace(",", "").replace(
                "$", ""
            ).replace("%", "") not in source_text_clean
        ]

        return {
            "numbers_found": numbers,
            "unverified": unverified,
        }

    def verify_phrases(self, generated_text: str) -> dict:
        """Flag suspicious hallucination phrases."""
        hallucination_phrases = [
            "many users",
            "most customers",
            "majority of",
            "studies show",
            "research indicates",
            "it is well known",
            "everyone knows",
            "it is widely accepted",
        ]
        flagged = [
            p for p in hallucination_phrases
            if p in generated_text.lower()
        ]
        return {"flagged": flagged}

    def verify(self, generated_text: str) -> dict:
        """
        Run all hallucination checks.
        Returns pass/fail with full details.
        """
        if not generated_text:
            return {
                "passed": False,
                "status": "FLAGGED",
                "has_citations": False,
                "unverified_numbers": [],
                "flagged_phrases": [],
                "action": "review_before_roadmap",
            }

        citations = self.verify_citations(generated_text)
        numbers   = self.verify_numbers(generated_text)
        phrases   = self.verify_phrases(generated_text)

        all_passed = (
            citations["has_citations"] and
            len(numbers["unverified"]) == 0 and
            len(phrases["flagged"]) == 0
        )

        return {
            "passed":             all_passed,
            "status":             "VERIFIED" if all_passed else "FLAGGED",
            "has_citations":      citations["has_citations"],
            "unverified_numbers": numbers["unverified"],
            "flagged_phrases":    phrases["flagged"],
            "action":             "proceed" if all_passed else "review_before_roadmap",
        }
