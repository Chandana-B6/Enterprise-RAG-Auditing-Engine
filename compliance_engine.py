import re

class ComplianceAuditor:
    def __init__(self):
        # Mapped to global corporate exclusion policies (Responsible AI)
        self.forbidden_bias_terms = ["manpower", "preferred gender", "blacklist", "illegal alien"]

    def audit_response(self, query: str, context_chunks: list, ai_response: str) -> dict:
        """
        Evaluates system outputs safely against empty states.
        Tracks token alignments and runs strict policy checks.
        """
        audit_results = {
            "hallucination_detected": False,
            "hallucination_score": 0.0,
            "ethical_violation": False,
            "flagged_reasons": []
        }

        # DEFENSIVE PROGRAMMING BLOCK: Convert input to clean string safely
        safe_response = str(ai_response) if ai_response is not None else ""
        safe_context = " ".join(context_chunks) if context_chunks else ""

        # 1. Linguistic Hallucination Evaluation (Token Alignment Check)
        combined_context = safe_context.lower()
        response_tokens = re.findall(r'\b\w{4,}\b', safe_response.lower())
        
        if response_tokens:
            missing_tokens = [t for t in response_tokens if t not in combined_context]
            hallucination_ratio = len(missing_tokens) / len(response_tokens)
            audit_results["hallucination_score"] = round(hallucination_ratio, 2)
            
            # If over 30% of words cannot be traced to the text source
            if hallucination_ratio > 0.30:
                audit_results["hallucination_detected"] = True
                audit_results["flagged_reasons"].append(
                    "Factual Verification Failure: AI assertions are ungrounded in source documents."
                )

        # 2. Responsible AI Compliance Audit (Ethics and Inclusion Scanning)
        for term in self.forbidden_bias_terms:
            if term in safe_response.lower():
                audit_results["ethical_violation"] = True
                audit_results["flagged_reasons"].append(
                    f"Policy Guardrail Violation: Non-inclusive/biased term detected: '{term}'"
                )

        return audit_results
