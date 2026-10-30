
import re


class ComplianceAuditor:
    def __init__(self):
        self.forbidden_bias_terms = [
            "manpower",
            "preferred gender",
            "blacklist",
            "illegal alien"
        ]

        self.abstention_phrases = [
            "i cannot verify",
            "does not provide sufficiently relevant evidence",
            "does not contain this information",
            "please consult the relevant department"
        ]

    def audit_response(
        self,
        query: str,
        context_chunks: list,
        ai_response: str
    ) -> dict:

        audit_results = {
            "hallucination_detected": False,
            "hallucination_score": 0.0,
            "ethical_violation": False,
            "flagged_reasons": []
        }

        safe_response = (
            str(ai_response).strip()
            if ai_response is not None else ""
        )

        safe_context = " ".join(
            str(chunk) for chunk in (context_chunks or [])
        )

        # 1. Empty evidence check
        if not safe_context.strip():
            audit_results["hallucination_detected"] = True
            audit_results["hallucination_score"] = 1.0
            audit_results["flagged_reasons"].append(
                "Evidence Failure: No source context was retrieved."
            )

        # 2. Empty response check
        if not safe_response:
            audit_results["hallucination_detected"] = True
            audit_results["hallucination_score"] = 1.0
            audit_results["flagged_reasons"].append(
                "Response Failure: No answer was generated."
            )

        # 3. Abstention detection
        is_abstention = any(
            phrase in safe_response.lower()
            for phrase in self.abstention_phrases
        )

        if is_abstention:
            audit_results["flagged_reasons"].append(
                "Abstention: The system declined to answer due to "
                "insufficient supporting evidence."
            )

        # 4. Basic token alignment check
        if safe_response and safe_context and not is_abstention:

            response_tokens = set(
                re.findall(r"\b\w{4,}\b", safe_response.lower())
            )

            context_tokens = set(
                re.findall(r"\b\w{4,}\b", safe_context.lower())
            )

            if response_tokens:
                missing_tokens = response_tokens - context_tokens

                ratio = len(missing_tokens) / len(response_tokens)

                audit_results["hallucination_score"] = round(
                    ratio, 2
                )

                if ratio > 0.30:
                    audit_results["hallucination_detected"] = True
                    audit_results["flagged_reasons"].append(
                        "Factual Verification Warning: Some answer "
                        "terms do not appear in retrieved source text."
                    )

        # 5. Responsible AI terminology check
        for term in self.forbidden_bias_terms:
            if term in safe_response.lower():
                audit_results["ethical_violation"] = True
                audit_results["flagged_reasons"].append(
                    "Policy Guardrail Warning: Potentially "
                    f"non-inclusive term detected: '{term}'"
                )

        return audit_results
