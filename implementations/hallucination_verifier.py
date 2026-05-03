from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import re
import json
import numpy as np
from pydantic import BaseModel


class ClaimType(str, Enum):
    """
    FinGround's 6-type financial claim taxonomy

    Each type has specialized verification logic
    """
    NUMERICAL = "numerical"           # "Revenue was $383.29B"
    TEMPORAL = "temporal"             # "In Q3 2024..."
    ENTITY_ATTRIBUTE = "entity"       # "Apple is a technology company"
    COMPARATIVE = "comparative"       # "Revenue increased 5% YoY"
    REGULATORY = "regulatory"         # "Subject to SEC Rule 10-K"
    COMPUTATIONAL = "computational"   # "Gross margin = 44.1%"


@dataclass
class AtomicClaim:
    """
    Decomposed claim for verification

    Example:
        text: "Apple's revenue increased 5% to $383B in FY2024"
        claim_type: ClaimType.COMPARATIVE
        has_numbers: True
        has_dates: True
        extracted_values: {"revenue": 383e9, "growth": 0.05, "year": 2024}
    """

    text: str
    claim_type: ClaimType
    confidence: float = 1.0

    # Extracted components
    has_numbers: bool = False
    has_dates: bool = False
    has_formulas: bool = False

    # For computational claims
    formula: Optional[str] = None
    components: List[str] = field(default_factory=list)

    # Verification results
    verified: bool = False
    verification_method: Optional[str] = None
    supporting_evidence: Optional[str] = None
    confidence_after_verification: float = 0.0


@dataclass
class VerificationResult:
    """Result of claim verification"""
    claim: AtomicClaim
    verified: bool
    confidence: float
    evidence: str
    method: str
    error_message: Optional[str] = None


class FinGroundVerifier:
    """
    FinGround-style hallucination detection and prevention

    Three-stage pipeline:
    1. Decompose answer into atomic claims
    2. Route each claim to type-specific verifier
    3. Regenerate with only verified claims

    Usage:
        verifier = FinGroundVerifier(llm)

        answer = "Apple's revenue was $383B in 2024, up 5% YoY"
        verified_answer = verifier.verify_and_regenerate(answer, context)
    """

    def __init__(self, llm_client=None, embedding_model=None):
        """
        Initialize verifier

        Args:
            llm_client: LLM client for decomposition and generation
            embedding_model: For semantic evidence matching
        """

        self.llm = llm_client
        self.embedding_model = embedding_model

        # Verification thresholds
        self.exact_match_threshold = 0.01  # 1% for exact match
        self.fuzzy_match_threshold = 0.05   # 5% for fuzzy match

    def decompose(self, answer: str) -> List[AtomicClaim]:
        """
        Stage 1: Decompose answer into atomic claims

        Uses LLM to extract minimal complete statements
        """

        if not self.llm:
            # Fallback: simple sentence splitting
            return self._simple_decompose(answer)

        prompt = f"""
        Decompose this financial answer into atomic claims.

        Answer: "{answer}"

        For each claim, extract:
        1. text: The minimal complete statement
        2. claim_type: One of [numerical, temporal, entity, comparative, regulatory, computational]
        3. has_numbers: true/false
        4. has_dates: true/false
        5. formula: If computational, the formula (e.g., "gross_margin = (revenue - cogs) / revenue")

        Return as JSON array:
        [
            {{
                "text": "...",
                "claim_type": "numerical",
                "has_numbers": true,
                "has_dates": false,
                "formula": null
            }},
            ...
        ]
        """

        try:
            response = self.llm.generate(prompt)
            claims_data = json.loads(self._extract_json(response))

            claims = []
            for data in claims_data:
                claim = AtomicClaim(
                    text=data['text'],
                    claim_type=ClaimType(data['claim_type']),
                    has_numbers=data.get('has_numbers', False),
                    has_dates=data.get('has_dates', False),
                    formula=data.get('formula')
                )

                # Extract values for numerical claims
                if claim.has_numbers:
                    claim.components = self._extract_numbers(claim.text)

                claims.append(claim)

            return claims

        except Exception as e:
            print(f"Decomposition error: {e}")
            return self._simple_decompose(answer)

    def _simple_decompose(self, answer: str) -> List[AtomicClaim]:
        """Fallback: split by sentences"""

        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

        claims = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue

            # Classify claim type
            claim_type = self._classify_claim_type(sentence)

            claim = AtomicClaim(
                text=sentence.strip(),
                claim_type=claim_type,
                has_numbers=bool(re.search(r'\d+', sentence)),
                has_dates=bool(re.search(r'\d{4}|Q\d|January|February|March|April|May|June|July|August|September|October|November|December', sentence, re.I))
            )

            claims.append(claim)

        return claims

    def _classify_claim_type(self, text: str) -> ClaimType:
        """Classify claim type based on keywords and patterns"""

        text_lower = text.lower()

        # Computational indicators
        if any(kw in text_lower for kw in ['=', 'equals', 'calculated as', 'computed as']):
            return ClaimType.COMPUTATIONAL

        # Comparative indicators
        if any(kw in text_lower for kw in ['increased', 'decreased', 'grew', 'declined', 'vs', 'compared to', 'higher than', 'lower than', '%', 'percent']):
            return ClaimType.COMPARATIVE

        # Regulatory indicators
        if any(kw in text_lower for kw in ['sec', 'gaap', 'ifrs', 'regulation', 'rule', 'compliance', 'regulatory']):
            return ClaimType.REGULATORY

        # Temporal indicators
        if re.search(r'\d{4}|Q\d|fiscal year|quarter', text_lower):
            return ClaimType.TEMPORAL

        # Numerical (has numbers but not comparative)
        if re.search(r'[$€£]?\d+[.,]?\d*[mbk]?', text_lower):
            return ClaimType.NUMERICAL

        # Default: entity attribute
        return ClaimType.ENTITY_ATTRIBUTE

    def verify(
        self,
        claims: List[AtomicClaim],
        context: List[str]
    ) -> List[AtomicClaim]:
        """
        Stage 2: Verify each claim against context

        Routes to type-specific verifier
        """

        verified_claims = []

        for claim in claims:
            # Route to appropriate verifier
            if claim.claim_type == ClaimType.NUMERICAL:
                verified_claim = self._verify_numerical(claim, context)
            elif claim.claim_type == ClaimType.COMPARATIVE:
                verified_claim = self._verify_comparative(claim, context)
            elif claim.claim_type == ClaimType.COMPUTATIONAL:
                verified_claim = self._verify_computational(claim, context)
            elif claim.claim_type == ClaimType.TEMPORAL:
                verified_claim = self._verify_temporal(claim, context)
            else:
                verified_claim = self._verify_generic(claim, context)

            verified_claims.append(verified_claim)

        return verified_claims

    def _verify_numerical(
        self,
        claim: AtomicClaim,
        context: List[str]
    ) -> AtomicClaim:
        """
        Verify numerical claims

        Key insight: Values within ±5% are hard to detect
        Our approach: Flag near-matches for human review
        """

        # Extract claimed value
        claimed_value = self._extract_primary_number(claim.text)

        if claimed_value is None:
            claim.verified = False
            claim.verification_method = "no_number_found"
            return claim

        # Search context for matching numbers
        best_match = None
        best_distance = float('inf')

        for ctx in context:
            ctx_numbers = self._extract_all_numbers_with_context(ctx)

            for ctx_value, ctx_text in ctx_numbers:
                # Calculate relative distance
                distance = abs(ctx_value - claimed_value) / max(claimed_value, 1)

                if distance < best_distance:
                    best_distance = distance
                    best_match = (ctx_value, ctx_text)

        # Evaluate match
        if best_match is None:
            claim.verified = False
            claim.supporting_evidence = "No matching number found in context"
            claim.verification_method = "numerical_no_match"

        elif best_distance < self.exact_match_threshold:
            claim.verified = True
            claim.supporting_evidence = f"Exact match: {best_match[0]} in context"
            claim.verification_method = "numerical_exact_match"
            claim.confidence_after_verification = 0.95

        elif best_distance < self.fuzzy_match_threshold:
            # Near match - flag for review
            claim.verified = False
            claim.supporting_evidence = f"Near match: {best_match[0]} vs claimed {claimed_value} (diff: {best_distance*100:.1f}%)"
            claim.verification_method = "numerical_near_match"
            claim.confidence_after_verification = 0.3

        else:
            claim.verified = False
            claim.supporting_evidence = f"Mismatch: found {best_match[0]} vs claimed {claimed_value}"
            claim.verification_method = "numerical_mismatch"

        return claim

    def _verify_comparative(
        self,
        claim: AtomicClaim,
        context: List[str]
    ) -> AtomicClaim:
        """
        Verify comparative claims

        Example: "Revenue increased 5% YoY"
        → Find both values and verify the percentage
        """

        # Extract percentage/growth value
        growth_value = self._extract_percentage(claim.text)

        if growth_value is None:
            # Try to verify directionally
            return self._verify_comparative_directional(claim, context)

        # Find base values in context
        values = self._find_comparative_values(claim.text, context)

        if values and len(values) >= 2:
            # Calculate actual growth
            actual_growth = (values[1] - values[0]) / abs(values[0]) if values[0] != 0 else 0

            if abs(actual_growth - growth_value) < 0.01:
                claim.verified = True
                claim.supporting_evidence = f"Calculated growth: {actual_growth*100:.1f}%"
                claim.verification_method = "comparative_calculated"
            elif abs(actual_growth - growth_value) < 0.05:
                claim.verified = False
                claim.supporting_evidence = f"Near match: calculated {actual_growth*100:.1f}% vs claimed {growth_value*100:.1f}%"
                claim.verification_method = "comparative_near_match"
            else:
                claim.verified = False
                claim.supporting_evidence = f"Mismatch: calculated {actual_growth*100:.1f}% vs claimed {growth_value*100:.1f}%"
                claim.verification_method = "comparative_mismatch"
        else:
            claim.verified = False
            claim.supporting_evidence = "Could not find comparative values in context"
            claim.verification_method = "comparative_not_found"

        return claim

    def _verify_computational(
        self,
        claim: AtomicClaim,
        context: List[str]
    ) -> AtomicClaim:
        """
        Verify computational claims by recomputing

        Example: "Gross margin was 44.1%"
        → Find revenue and COGS
        → Compute: (Revenue - COGS) / Revenue
        """

        # Parse formula or infer from claim
        formula = claim.formula or self._infer_formula(claim.text)

        if not formula:
            claim.verified = False
            claim.supporting_evidence = "Could not infer formula"
            return claim

        # Find component values
        components = self._find_formula_components(formula, context)

        if not components:
            claim.verified = False
            claim.supporting_evidence = f"Could not find components for: {formula}"
            claim.verification_method = "computational_missing_components"
            return claim

        # Compute
        try:
            computed_value = self._evaluate_formula(formula, components)
            claimed_value = self._extract_primary_number(claim.text)

            if claimed_value is None:
                claim.verified = False
                claim.supporting_evidence = "Could not extract claimed value"
                return claim

            # Compare
            if abs(computed_value - claimed_value) / max(claimed_value, 1) < 0.01:
                claim.verified = True
                claim.supporting_evidence = f"Computed: {computed_value:.4f}"
                claim.verification_method = "computational_verified"
                claim.confidence_after_verification = 0.9
            else:
                claim.verified = False
                claim.supporting_evidence = f"Computed: {computed_value:.4f} vs claimed: {claimed_value}"
                claim.verification_method = "computational_mismatch"

        except Exception as e:
            claim.verified = False
            claim.supporting_evidence = f"Computation error: {str(e)}"
            claim.verification_method = "computational_error"

        return claim

    def _verify_temporal(
        self,
        claim: AtomicClaim,
        context: List[str]
    ) -> AtomicClaim:
        """
        Verify temporal claims

        Check if the date/period matches the context
        """

        # Extract date from claim
        claim_date = self._extract_date(claim.text)

        if not claim_date:
            claim.verified = False
            claim.supporting_evidence = "Could not extract date from claim"
            return claim

        # Search for matching date in context
        for ctx in context:
            if claim_date in ctx or self._dates_match(claim_date, ctx):
                claim.verified = True
                claim.supporting_evidence = f"Date verified in context"
                claim.verification_method = "temporal_match"
                claim.confidence_after_verification = 0.85
                return claim

        claim.verified = False
        claim.supporting_evidence = f"Date '{claim_date}' not found in context"
        claim.verification_method = "temporal_not_found"

        return claim

    def _verify_generic(
        self,
        claim: AtomicClaim,
        context: List[str]
    ) -> AtomicClaim:
        """
        Generic verification for entity/regulatory claims

        Uses semantic similarity
        """

        if not self.embedding_model:
            # Fallback: keyword matching
            claim_text_lower = claim.text.lower()

            for ctx in context:
                if claim_text_lower in ctx.lower():
                    claim.verified = True
                    claim.supporting_evidence = "Found in context"
                    claim.verification_method = "generic_keyword_match"
                    claim.confidence_after_verification = 0.8
                    return claim

            claim.verified = False
            claim.supporting_evidence = "Not found in context"
            claim.verification_method = "generic_not_found"
            return claim

        # Use embeddings for semantic matching
        claim_embedding = self.embedding_model.encode([claim.text])[0]

        best_similarity = 0
        best_ctx = None

        for ctx in context:
            ctx_embedding = self.embedding_model.encode([ctx])[0]
            similarity = self._cosine_similarity(claim_embedding, ctx_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_ctx = ctx

        if best_similarity > 0.85:
            claim.verified = True
            claim.supporting_evidence = f"Semantic match (similarity: {best_similarity:.2f})"
            claim.verification_method = "generic_semantic_match"
            claim.confidence_after_verification = best_similarity
        else:
            claim.verified = False
            claim.supporting_evidence = f"No semantic match (best: {best_similarity:.2f})"
            claim.verification_method = "generic_no_semantic_match"

        return claim

    def regenerate_verified(
        self,
        claims: List[AtomicClaim],
        original_query: str
    ) -> Tuple[str, float]:
        """
        Stage 3: Regenerate answer with only verified claims

        Returns:
            (verified_answer, confidence_score)
        """

        verified = [c for c in claims if c.verified]
        unverified = [c for c in claims if not c.verified]

        if not verified:
            return (
                "I cannot verify any claims from the retrieved context. I cannot provide a reliable answer.",
                0.0
            )

        if not self.llm:
            # Simple concatenation fallback
            answer = " ".join([c.text for c in verified])
            confidence = sum(c.confidence_after_verification for c in verified) / len(verified)
            return answer, confidence

        prompt = f"""
        Regenerate the answer using ONLY these verified claims.

        Original query: "{original_query}"

        Verified claims (USE THESE):
        {[c.text for c in verified]}

        Unverified claims (DO NOT USE, but mention if important information is missing):
        {[c.text for c in unverified]}

        Requirements:
        1. Only use verified claims
        2. Include citations like [Source: context]
        3. If important information was unverified, say "Could not verify: ..."
        4. End with confidence score (0-1)

        Answer:
        """

        response = self.llm.generate(prompt)

        # Extract confidence from response
        confidence = self._extract_confidence_score(response)

        return response, confidence

    # ==================== Helper Methods ====================

    def _extract_primary_number(self, text: str) -> Optional[float]:
        """Extract the primary numerical value from text"""

        # Handle $, B, M, K suffixes
        pattern = r'[$€£]?([\d,]+\.?\d*)\s*([BbMmKk])?'
        match = re.search(pattern, text)

        if not match:
            return None

        value = float(match.group(1).replace(',', ''))
        suffix = match.group(2)

        if suffix:
            suffix = suffix.upper()
            if suffix == 'B':
                value *= 1e9
            elif suffix == 'M':
                value *= 1e6
            elif suffix == 'K':
                value *= 1e3

        return value

    def _extract_all_numbers_with_context(
        self,
        text: str
    ) -> List[Tuple[float, str]]:
        """Extract all numbers with surrounding context"""

        results = []
        pattern = r'[$€£]?([\d,]+\.?\d*)\s*([BbMmKk])?'

        for match in re.finditer(pattern, text):
            value = float(match.group(1).replace(',', ''))
            suffix = match.group(2)

            if suffix:
                suffix = suffix.upper()
                if suffix == 'B':
                    value *= 1e9
                elif suffix == 'M':
                    value *= 1e6
                elif suffix == 'K':
                    value *= 1e3

            # Get surrounding context (10 chars)
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end]

            results.append((value, context))

        return results

    def _extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage value"""

        pattern = r'(\d+\.?\d*)\s*%'
        match = re.search(pattern, text)

        if match:
            return float(match.group(1)) / 100

        # Handle "X percent" format
        pattern = r'(\d+\.?\d*)\s*percent'
        match = re.search(pattern, text, re.I)

        if match:
            return float(match.group(1)) / 100

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date/year from text"""

        # Year pattern
        match = re.search(r'\b(19|20)\d{2}\b', text)
        if match:
            return match.group()

        # Quarter pattern
        match = re.search(r'Q[1-4]\s*\d{4}', text)
        if match:
            return match.group()

        # Fiscal year pattern
        match = re.search(r'fiscal\s*year?\s*(\d{4})', text, re.I)
        if match:
            return f"FY{match.group(1)}"

        return None

    def _cosine_similarity(self, a, b) -> float:
        """Compute cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        # Find JSON between brackets
        match = re.search(r'\[.*\]', text, re.DOTALL)
        return match.group() if match else text

    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from response"""

        pattern = r'confidence[:\s]*([0-9.]+)'
        match = re.search(pattern, text, re.I)

        if match:
            return min(1.0, max(0.0, float(match.group(1))))

        return 0.5  # Default

    def _infer_formula(self, text: str) -> Optional[str]:
        """
        Infer common finance formulas from text.
        """

        text = text.lower()

        formula_patterns = {
            "gross margin": "(revenue-cogs)/revenue",
            "operating margin": "operating_income/revenue",
            "net margin": "net_income/revenue",
            "profit margin": "net_income/revenue",
            "eps": "net_income/shares_outstanding",
            "debt ratio": "total_debt/total_assets",
            "current ratio": "current_assets/current_liabilities",
        }

        for keyword, formula in formula_patterns.items():
            if keyword in text:
                return formula

        return None


    def _find_formula_components(
        self,
        formula: str,
        context: List[str]
    ) -> Optional[Dict]:
        """
        Extract variables required by formula from context.
        """

        variables = set(re.findall(r"[a-zA-Z_]+", formula))
        components = {}

        for var in variables:
            pattern = rf"{var.replace('_', ' ')}.*?([$€£]?\d+[.,]?\d*[BbMmKk]?)"

            for ctx in context:
                match = re.search(pattern, ctx, re.I)
                if match:
                    value = self._extract_primary_number(match.group(1))
                    if value is not None:
                        components[var] = value
                        break

        return components if components else None


    def _evaluate_formula(
        self,
        formula: str,
        components: Dict
    ) -> float:
        """
        Safely evaluate formula.
        """

        safe_formula = formula

        for key, value in components.items():
            safe_formula = safe_formula.replace(key, str(value))

        allowed_chars = set(
            "0123456789.+-*/() "
        )

        if not all(c in allowed_chars for c in safe_formula):
            raise ValueError("Unsafe formula")

        return float(eval(safe_formula))


    def _verify_comparative_directional(
        self,
        claim: AtomicClaim,
        context: List[str]
    ) -> AtomicClaim:
        """
        Verify directional changes like increased/decreased.
        """

        values = self._find_comparative_values(claim.text, context)

        if not values or len(values) < 2:
            claim.verified = False
            claim.verification_method = "directional_not_found"
            claim.supporting_evidence = "Not enough values"
            return claim

        direction = None
        text = claim.text.lower()

        if "increase" in text or "grew" in text:
            direction = "up"
        elif "decrease" in text or "declined" in text:
            direction = "down"

        actual_direction = "up" if values[1] > values[0] else "down"

        if direction == actual_direction:
            claim.verified = True
            claim.verification_method = "directional_verified"
            claim.supporting_evidence = f"{values[0]} -> {values[1]}"
            claim.confidence_after_verification = 0.8
        else:
            claim.verified = False
            claim.verification_method = "directional_mismatch"
            claim.supporting_evidence = f"{values[0]} -> {values[1]}"

        return claim


    def _find_comparative_values(
        self,
        text: str,
        context: List[str]
    ) -> Optional[List[float]]:
        """
        Find numbers that likely form a comparison.
        """

        values = []

        for ctx in context:
            nums = self._extract_all_numbers_with_context(ctx)

            for num, _ in nums:
                values.append(num)

        if len(values) >= 2:
            return values[:2]

        return None


    def _extract_numbers(
        self,
        text: str
    ) -> List[str]:
        """
        Extract all raw numeric strings from text.
        """

        return re.findall(
            r'[$€£]?\d+[.,]?\d*[BbMmKk]?',
            text
        )

    def _dates_match(self, date_str: str, context: str) -> bool:
        """Check if a date string appears in the context text."""
        return date_str.lower() in context.lower()


# Example usage
if __name__ == "__main__":
    # Mock LLM client for testing
    class MockLLM:
        def generate(self, prompt: str) -> str:
            return "Mock response"

    # Initialize verifier
    verifier = FinGroundVerifier(llm_client=MockLLM())

    # Test claim decomposition
    answer = """
    Apple's revenue was $383.29 billion in fiscal year 2024, up 2% year-over-year.
    Gross margin was 44.1% compared to 44.9% in the prior year.
    The company is subject to SEC Rule 10-K reporting requirements.
    """

    claims = verifier.decompose(answer)

    print(f"Decomposed into {len(claims)} claims:\n")
    for i, claim in enumerate(claims):
        print(f"{i+1}. [{claim.claim_type.value}] {claim.text}")
        print(f"   Numbers: {claim.has_numbers}, Dates: {claim.has_dates}")
        print()

    # Test verification
    context = [
        "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
        "Gross margin was 44.1% compared to 44.9% in fiscal 2023.",
        "As a public company, Apple files annual 10-K reports with the SEC."
    ]

    verified_claims = verifier.verify(claims, context)

    print("\nVerification results:\n")
    for claim in verified_claims:
        status = "✓" if claim.verified else "✗"
        print(f"{status} {claim.text}")
        print(f"  Method: {claim.verification_method}")
        print(f"  Evidence: {claim.supporting_evidence}")
        print()
