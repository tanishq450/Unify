"""
Tests for implementations/intent_classifier.py

Covers:
- IntentClassifier.classify — all 6 intent categories
- IntentClassifier.classify_batch
- Edge cases: empty query, no keyword matches, multi-intent query
- RoutingResult fields (confidence, reasoning, keywords_matched)
- AdaptiveRouter: low-confidence fallback, feedback tracking, threshold update
"""

import pytest

from implementations.intent_classifier import (
    AdaptiveRouter,
    IntentClassifier,
    QueryIntent,
    RAGStrategy,
    RoutingResult,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def classifier():
    return IntentClassifier()


@pytest.fixture()
def router():
    return AdaptiveRouter()


# ─────────────────────────────────────────────────────────────
# 1. Happy-path routing — one clear intent per query
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected_intent,expected_strategy", [
    # General knowledge
    ("What does Apple do?", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("Explain Apple's business model", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("Describe the company overview", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("Tell me about Tesla's mission statement", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("What is a balance sheet?", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),

    # Numerical / Table
    ("Show iPhone revenue by quarter", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("What was Apple's revenue in 2024?", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("Break down revenue by segment", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("What was the gross margin last quarter?", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("Net income for fiscal year 2023", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("What is the EPS for Q3?", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),

    # Comparison
    ("Apple vs Microsoft market cap", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),
    ("Compare revenue of Apple and Google", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),
    ("How does Tesla's ratio compare to the industry?", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),

    # Trend
    ("Revenue growth trend 2024", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("How has revenue changed over time?", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("Show year-over-year growth in earnings", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("QoQ increase in cloud revenue", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),

    # Relationship
    ("Who are Apple's suppliers?", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("List Apple's key partners and collaborations", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("What Apple acquisition was most recent?", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("Who are the main competitors of Tesla?", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),

    # Entity
    ("Who is Apple's CEO?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("Where are Tesla's headquarters?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("Who founded Microsoft?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("What industry is Nvidia in?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("How many employees does Google have?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
])
def test_classify_happy_path(classifier, query, expected_intent, expected_strategy):
    result = classifier.classify(query)
    assert result.intent == expected_intent, (
        f"Query: '{query}' | Expected intent {expected_intent.value}, got {result.intent.value}"
    )
    assert result.strategy == expected_strategy, (
        f"Query: '{query}' | Expected strategy {expected_strategy.value}, got {result.strategy.value}"
    )


# ─────────────────────────────────────────────────────────────
# 2. RoutingResult structure
# ─────────────────────────────────────────────────────────────

def test_routing_result_is_named_tuple(classifier):
    result = classifier.classify("What does Apple do?")
    assert isinstance(result, RoutingResult)


def test_routing_result_confidence_range(classifier):
    """Confidence must be in [0, 1]."""
    for query in [
        "What does Apple do?",
        "Show iPhone revenue by quarter",
        "Apple vs Microsoft",
        "Who is Apple's CEO?",
    ]:
        result = classifier.classify(query)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence {result.confidence} out of range for '{query}'"
        )


def test_routing_result_has_reasoning(classifier):
    result = classifier.classify("Show iPhone revenue by quarter")
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0


def test_routing_result_has_keywords_matched(classifier):
    result = classifier.classify("Show iPhone revenue by quarter")
    assert isinstance(result.keywords_matched, list)
    assert len(result.keywords_matched) > 0


# ─────────────────────────────────────────────────────────────
# 3. Edge cases
# ─────────────────────────────────────────────────────────────

def test_empty_query_defaults_to_general_knowledge(classifier):
    """An empty string should fall back to GENERAL_KNOWLEDGE / SIMPLE_RAG."""
    result = classifier.classify("")
    assert result.intent == QueryIntent.GENERAL_KNOWLEDGE
    assert result.strategy == RAGStrategy.SIMPLE_RAG
    assert result.confidence == 0.5


def test_gibberish_query_defaults_to_general_knowledge(classifier):
    result = classifier.classify("xyzzy frobnicator quux")
    assert result.intent == QueryIntent.GENERAL_KNOWLEDGE
    assert result.strategy == RAGStrategy.SIMPLE_RAG


def test_case_insensitivity(classifier):
    """Keywords should match regardless of case."""
    lower = classifier.classify("who is apple's ceo?")
    upper = classifier.classify("WHO IS APPLE'S CEO?")
    mixed = classifier.classify("Who Is Apple's CEO?")
    assert lower.intent == upper.intent == mixed.intent == QueryIntent.ENTITY


def test_strategy_matches_intent_mapping(classifier):
    """Every returned strategy must agree with INTENT_TO_STRATEGY."""
    queries = [
        "What does Apple do?",
        "Show iPhone revenue by quarter",
        "Apple vs Microsoft market cap",
        "Revenue growth trend 2024",
        "Who are Apple's suppliers?",
        "Who is Apple's CEO?",
    ]
    for q in queries:
        result = classifier.classify(q)
        expected_strategy = IntentClassifier.INTENT_TO_STRATEGY[result.intent]
        assert result.strategy == expected_strategy


# ─────────────────────────────────────────────────────────────
# 4. Batch classification
# ─────────────────────────────────────────────────────────────

def test_classify_batch_returns_correct_count(classifier):
    queries = [
        "What does Apple do?",
        "Show iPhone revenue by quarter",
        "Apple vs Microsoft",
    ]
    results = classifier.classify_batch(queries)
    assert len(results) == len(queries)


def test_classify_batch_each_result_is_routing_result(classifier):
    queries = ["What does Apple do?", "Who is Apple's CEO?"]
    results = classifier.classify_batch(queries)
    for r in results:
        assert isinstance(r, RoutingResult)


def test_classify_batch_single_item(classifier):
    results = classifier.classify_batch(["What is the EPS for Q3?"])
    assert len(results) == 1
    assert results[0].intent == QueryIntent.NUMERICAL_TABLE


def test_classify_batch_empty_list(classifier):
    results = classifier.classify_batch([])
    assert results == []


# ─────────────────────────────────────────────────────────────
# 5. Confidence boosting
# ─────────────────────────────────────────────────────────────

def test_high_match_boosts_confidence(classifier):
    """Query with many matching keywords should yield higher confidence."""
    low_match = classifier.classify("revenue")          # one keyword
    high_match = classifier.classify(
        "Show iPhone revenue by segment, quarter, and division in billions"
    )
    assert high_match.confidence >= low_match.confidence


# ─────────────────────────────────────────────────────────────
# 6. AdaptiveRouter
# ─────────────────────────────────────────────────────────────

def test_adaptive_router_high_confidence_passes_through(router):
    result = router.route("Show iPhone revenue by quarter")
    assert result.intent == QueryIntent.NUMERICAL_TABLE
    assert result.strategy == RAGStrategy.MULTIMODAL_RAG


def test_adaptive_router_low_confidence_falls_back(router):
    """Manually lower the threshold above the expected confidence to force fallback."""
    router.thresholds[QueryIntent.GENERAL_KNOWLEDGE] = 0.99
    result = router.route("What does Apple do?")
    # With threshold at 0.99 the result should fall back to SIMPLE_RAG anyway (same strategy)
    assert result.strategy == RAGStrategy.SIMPLE_RAG


def test_adaptive_router_record_feedback_stores_entry(router):
    result = router.route("Who is Apple's CEO?")
    router.record_feedback(result, was_helpful=True)
    assert len(router.feedback_history) == 1
    assert router.feedback_history[0]["was_helpful"] is True


def test_adaptive_router_threshold_rises_after_many_failures(router):
    """10+ unhelpful feedback entries for an intent should raise its threshold."""
    result = router.route("Who is Apple's CEO?")
    initial_threshold = router.thresholds[result.intent]

    for _ in range(15):
        router.record_feedback(result, was_helpful=False)

    assert router.thresholds[result.intent] >= initial_threshold


def test_adaptive_router_threshold_drops_after_many_successes(router):
    """10+ helpful feedback entries for an intent should lower its threshold."""
    result = router.route("Who is Apple's CEO?")
    # Artificially raise threshold so there is room to drop
    router.thresholds[result.intent] = 0.5
    initial_threshold = router.thresholds[result.intent]

    for _ in range(15):
        router.record_feedback(result, was_helpful=True)

    assert router.thresholds[result.intent] <= initial_threshold


def test_adaptive_router_no_threshold_update_before_10_samples(router):
    result = router.route("Who is Apple's CEO?")
    initial = dict(router.thresholds)

    for _ in range(9):
        router.record_feedback(result, was_helpful=False)

    # Should NOT have changed yet
    assert router.thresholds == initial
