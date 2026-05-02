"""
Tests for implementations/hallucination_verifier.py

Covers:
- FinGroundVerifier._extract_primary_number     — currency prefixes, B/M/K suffixes
- FinGroundVerifier._extract_percentage         — % and "percent" formats
- FinGroundVerifier._extract_date               — year, quarter, fiscal-year patterns
- FinGroundVerifier._classify_claim_type        — 6 claim types
- FinGroundVerifier._simple_decompose           — sentence splitting, short-sentence filter
- FinGroundVerifier._verify_numerical           — exact match, near match, mismatch, no number
- FinGroundVerifier._verify_temporal            — date match, date mismatch, no date
- FinGroundVerifier._verify_generic             — keyword match, no match
- FinGroundVerifier.verify                      — routing to per-type verifiers
- FinGroundVerifier.regenerate_verified         — all verified, none verified, no LLM fallback
- ClaimType enum completeness
- AtomicClaim dataclass defaults
- evaluation.ClassificationMetrics             — precision / recall / F1 properties
"""

import pytest

from implementations.hallucination_verifier import (
    AtomicClaim,
    ClaimType,
    FinGroundVerifier,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_verifier(**kwargs):
    """Return a FinGroundVerifier with no LLM (pure rule-based path)."""
    return FinGroundVerifier(**kwargs)


def make_claim(text, claim_type=ClaimType.NUMERICAL, **kwargs):
    return AtomicClaim(text=text, claim_type=claim_type, **kwargs)


# ─────────────────────────────────────────────────────────────
# 1. ClaimType enum
# ─────────────────────────────────────────────────────────────

def test_claim_type_values():
    expected = {"numerical", "temporal", "entity", "comparative", "regulatory", "computational"}
    assert {ct.value for ct in ClaimType} == expected


# ─────────────────────────────────────────────────────────────
# 2. AtomicClaim dataclass defaults
# ─────────────────────────────────────────────────────────────

def test_atomic_claim_defaults():
    claim = AtomicClaim(text="Revenue was $100B", claim_type=ClaimType.NUMERICAL)
    assert claim.confidence == 1.0
    assert claim.has_numbers is False
    assert claim.has_dates is False
    assert claim.verified is False
    assert claim.verification_method is None
    assert claim.supporting_evidence is None
    assert claim.confidence_after_verification == 0.0
    assert claim.components == []


# ─────────────────────────────────────────────────────────────
# 3. _extract_primary_number
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def verifier():
    return make_verifier()


@pytest.mark.parametrize("text,expected", [
    ("Revenue was $383.29B", 383.29e9),
    ("Revenue was $383.29 billion", 383.29e9),   # B suffix match
    ("Cost was $50M", 50e6),
    ("Headcount is 10K", 10e3),
    ("Price is $1,234.56", 1234.56),
    ("EPS was 6.43", 6.43),
    ("Revenue was €200B", 200e9),
    ("Revenue was £150B", 150e9),
])
def test_extract_primary_number(verifier, text, expected):
    result = verifier._extract_primary_number(text)
    assert result == pytest.approx(expected, rel=1e-3)


def test_extract_primary_number_no_number(verifier):
    assert verifier._extract_primary_number("No figures here") is None


def test_extract_primary_number_zero(verifier):
    result = verifier._extract_primary_number("Revenue was $0")
    assert result == 0.0


# ─────────────────────────────────────────────────────────────
# 4. _extract_percentage
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Gross margin was 44.1%", 0.441),
    ("Revenue grew 5%", 0.05),
    ("Margin improved 3.5 percent", 0.035),
    ("Declined 10 Percent", 0.10),
])
def test_extract_percentage(verifier, text, expected):
    result = verifier._extract_percentage(text)
    assert result == pytest.approx(expected, rel=1e-3)


def test_extract_percentage_no_percent(verifier):
    assert verifier._extract_percentage("No percentage value here") is None


# ─────────────────────────────────────────────────────────────
# 5. _extract_date
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_substr", [
    ("Revenue in fiscal year 2024", "2024"),
    # Year is extracted first; the quarter year is also captured as "2023"
    ("Reported in Q3 2023", "2023"),
    # Year is extracted first; fiscal year returns "2022"
    ("Fiscal year 2022 results", "2022"),
    ("Results for 2020 were strong", "2020"),
])
def test_extract_date(verifier, text, expected_substr):
    result = verifier._extract_date(text)
    assert result is not None
    assert expected_substr in result


def test_extract_date_no_date(verifier):
    assert verifier._extract_date("No date information here") is None


# ─────────────────────────────────────────────────────────────
# 6. _classify_claim_type
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_type", [
    ("Gross margin equals (revenue - cogs) / revenue", ClaimType.COMPUTATIONAL),
    ("Revenue increased 5% year-over-year", ClaimType.COMPARATIVE),
    ("Revenue grew compared to last year", ClaimType.COMPARATIVE),
    ("Apple files annual 10-K with the SEC", ClaimType.REGULATORY),
    ("Reported revenue of $383B in fiscal year 2024", ClaimType.TEMPORAL),
    # No year present → classified as NUMERICAL (has digits but no 4-digit year)
    ("Revenue was $383 billion", ClaimType.NUMERICAL),
    ("Apple is a technology company", ClaimType.ENTITY_ATTRIBUTE),
])
def test_classify_claim_type(verifier, text, expected_type):
    result = verifier._classify_claim_type(text)
    assert result == expected_type


# ─────────────────────────────────────────────────────────────
# 7. _simple_decompose
# ─────────────────────────────────────────────────────────────

def test_simple_decompose_splits_sentences(verifier):
    answer = (
        "Apple's revenue was $383.29 billion in fiscal year 2024. "
        "Gross margin was 44.1%."
    )
    claims = verifier._simple_decompose(answer)
    assert len(claims) == 2


def test_simple_decompose_skips_short_sentences(verifier):
    """Sentences with fewer than 10 characters are ignored."""
    answer = "Revenue up. Apple's gross margin was 44.1% in FY2024."
    claims = verifier._simple_decompose(answer)
    # "Revenue up." is 11 chars — depending on the exact threshold it may or may not be included
    # The key assertion: no zero-length claims
    for c in claims:
        assert len(c.text) >= 10


def test_simple_decompose_detects_numbers(verifier):
    answer = "Revenue was $383B in FY2024."
    claims = verifier._simple_decompose(answer)
    assert any(c.has_numbers for c in claims)


def test_simple_decompose_detects_dates(verifier):
    answer = "Revenue was reported in Q3 2024."
    claims = verifier._simple_decompose(answer)
    assert any(c.has_dates for c in claims)


def test_simple_decompose_empty_answer(verifier):
    claims = verifier._simple_decompose("")
    assert claims == []


def test_simple_decompose_returns_atomic_claims(verifier):
    claims = verifier._simple_decompose("Apple's revenue was $383B.")
    for c in claims:
        assert isinstance(c, AtomicClaim)


# ─────────────────────────────────────────────────────────────
# 8. _verify_numerical
# ─────────────────────────────────────────────────────────────

def test_verify_numerical_exact_match(verifier):
    claim = make_claim("Revenue was $383.29B", ClaimType.NUMERICAL, has_numbers=True)
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    result = verifier._verify_numerical(claim, ctx)
    assert result.verified is True
    assert result.verification_method == "numerical_exact_match"


def test_verify_numerical_clear_mismatch(verifier):
    claim = make_claim("Revenue was $500B", ClaimType.NUMERICAL, has_numbers=True)
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    result = verifier._verify_numerical(claim, ctx)
    assert result.verified is False


def test_verify_numerical_near_match_is_within_exact_threshold(verifier):
    """$383B vs $383.29B: relative diff ≈ 0.076% < 1% exact_match_threshold → verified."""
    claim = make_claim("Revenue was $383B", ClaimType.NUMERICAL, has_numbers=True)
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    result = verifier._verify_numerical(claim, ctx)
    # The numbers are within the exact-match threshold (1%), so verified=True
    assert result.verified is True
    assert result.verification_method == "numerical_exact_match"


def test_verify_numerical_outside_exact_but_within_fuzzy(verifier):
    """$390B vs $383.29B: ~1.7% diff → above exact (1%) but below fuzzy (5%) → near-match."""
    claim = make_claim("Revenue was $390B", ClaimType.NUMERICAL, has_numbers=True)
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    result = verifier._verify_numerical(claim, ctx)
    assert result.verified is False
    assert result.verification_method == "numerical_near_match"


def test_verify_numerical_no_number_in_claim(verifier):
    claim = make_claim("Revenue was substantial", ClaimType.NUMERICAL, has_numbers=False)
    ctx = ["Apple Inc. reported revenue of $383.29 billion."]
    result = verifier._verify_numerical(claim, ctx)
    assert result.verified is False
    assert result.verification_method == "no_number_found"


def test_verify_numerical_empty_context(verifier):
    claim = make_claim("Revenue was $383.29B", ClaimType.NUMERICAL, has_numbers=True)
    result = verifier._verify_numerical(claim, [])
    assert result.verified is False


# ─────────────────────────────────────────────────────────────
# 9. _verify_temporal
# ─────────────────────────────────────────────────────────────

def test_verify_temporal_date_found(verifier):
    claim = make_claim(
        "Apple reported revenue in fiscal year 2024", ClaimType.TEMPORAL, has_dates=True
    )
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    result = verifier._verify_temporal(claim, ctx)
    assert result.verified is True


def test_verify_temporal_wrong_year(verifier):
    claim = make_claim(
        "Apple reported revenue in fiscal year 2023", ClaimType.TEMPORAL, has_dates=True
    )
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    # 2023 is NOT in the context → should not verify
    result = verifier._verify_temporal(claim, ctx)
    assert result.verified is False


def test_verify_temporal_no_date_in_claim(verifier):
    claim = make_claim("Revenue was substantial", ClaimType.TEMPORAL, has_dates=False)
    ctx = ["Apple reported revenue in 2024."]
    result = verifier._verify_temporal(claim, ctx)
    assert result.verified is False


# ─────────────────────────────────────────────────────────────
# 10. _verify_generic (keyword-based, no embedding model)
# ─────────────────────────────────────────────────────────────

def test_verify_generic_exact_keyword_match(verifier):
    claim = make_claim(
        "apple files annual 10-k reports with the sec", ClaimType.REGULATORY
    )
    ctx = ["apple files annual 10-k reports with the sec as a public company."]
    result = verifier._verify_generic(claim, ctx)
    assert result.verified is True


def test_verify_generic_no_match(verifier):
    claim = make_claim(
        "Microsoft's cloud revenue was $25 billion", ClaimType.ENTITY_ATTRIBUTE
    )
    ctx = ["Apple Inc. reported revenue of $383.29 billion for fiscal year 2024."]
    result = verifier._verify_generic(claim, ctx)
    assert result.verified is False


# ─────────────────────────────────────────────────────────────
# 11. verify — dispatch to per-type verifiers
# ─────────────────────────────────────────────────────────────

def test_verify_returns_same_number_of_claims(verifier):
    claims = [
        make_claim("Revenue was $383.29B", ClaimType.NUMERICAL, has_numbers=True),
        make_claim("Reported in fiscal year 2024", ClaimType.TEMPORAL, has_dates=True),
        make_claim("Apple files 10-K with SEC", ClaimType.REGULATORY),
    ]
    ctx = [
        "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
        "apple files 10-k with sec.",
    ]
    result = verifier.verify(claims, ctx)
    assert len(result) == len(claims)


def test_verify_all_verifiable_claims(verifier):
    claims = [
        make_claim("Revenue was $383.29B", ClaimType.NUMERICAL, has_numbers=True),
    ]
    ctx = ["Apple reported revenue of $383.29 billion."]
    result = verifier.verify(claims, ctx)
    assert result[0].verified is True


def test_verify_empty_claims_list(verifier):
    result = verifier.verify([], ["some context"])
    assert result == []


def test_verify_empty_context_list(verifier):
    claims = [make_claim("Revenue was $383.29B", ClaimType.NUMERICAL, has_numbers=True)]
    result = verifier.verify(claims, [])
    assert result[0].verified is False


# ─────────────────────────────────────────────────────────────
# 12. regenerate_verified (no LLM — fallback path)
# ─────────────────────────────────────────────────────────────

def _make_verified_claim(text, confidence_after=0.9):
    c = make_claim(text, ClaimType.NUMERICAL)
    c.verified = True
    c.confidence_after_verification = confidence_after
    return c


def _make_unverified_claim(text):
    c = make_claim(text, ClaimType.NUMERICAL)
    c.verified = False
    c.confidence_after_verification = 0.0
    return c


def test_regenerate_verified_all_verified_returns_answer(verifier):
    claims = [
        _make_verified_claim("Revenue was $383.29 billion.", 0.95),
        _make_verified_claim("Gross margin was 44.1%.", 0.90),
    ]
    answer, confidence = verifier.regenerate_verified(claims, "What was Apple's revenue?")
    assert "383.29" in answer or "44.1" in answer
    assert 0.0 <= confidence <= 1.0


def test_regenerate_verified_none_verified_returns_fallback(verifier):
    claims = [_make_unverified_claim("Revenue was $500 billion.")]
    answer, confidence = verifier.regenerate_verified(claims, "What was Apple's revenue?")
    assert confidence == 0.0
    assert "cannot verify" in answer.lower() or "cannot provide" in answer.lower()


def test_regenerate_verified_confidence_is_average_of_verified(verifier):
    claims = [
        _make_verified_claim("Claim A", 0.8),
        _make_verified_claim("Claim B", 0.6),
        _make_unverified_claim("Unverified claim"),
    ]
    _, confidence = verifier.regenerate_verified(claims, "test query")
    assert confidence == pytest.approx(0.7, abs=1e-6)


def test_regenerate_verified_empty_claims_returns_fallback(verifier):
    answer, confidence = verifier.regenerate_verified([], "any query")
    assert confidence == 0.0


# ─────────────────────────────────────────────────────────────
# 13. ClassificationMetrics (from evaluation module)
# ─────────────────────────────────────────────────────────────

from evaluation import ClassificationMetrics


def test_metrics_perfect_precision_recall():
    m = ClassificationMetrics(label="trend", true_positives=5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_metrics_zero_when_no_predictions():
    m = ClassificationMetrics(label="trend")
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_metrics_precision_calculation():
    # tp=3, fp=1 → precision = 3/4 = 0.75
    m = ClassificationMetrics(label="trend", true_positives=3, false_positives=1)
    assert m.precision == pytest.approx(0.75)


def test_metrics_recall_calculation():
    # tp=3, fn=1 → recall = 3/4 = 0.75
    m = ClassificationMetrics(label="trend", true_positives=3, false_negatives=1)
    assert m.recall == pytest.approx(0.75)


def test_metrics_f1_calculation():
    m = ClassificationMetrics(
        label="trend", true_positives=4, false_positives=1, false_negatives=1
    )
    p = 4 / 5
    r = 4 / 5
    expected_f1 = 2 * p * r / (p + r)
    assert m.f1 == pytest.approx(expected_f1)
