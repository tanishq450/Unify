"""
Evaluation Suite for Unify — Intelligent Finance RAG
=====================================================

Evaluates every stage of the pipeline:
  1. Intent Classification  — precision / recall / F1 per intent
  2. Hallucination Verifier — claim decomposition & verification accuracy
  3. End-to-End Pipeline    — answer faithfulness, relevance, confidence calibration

Usage:
    python evaluation.py                    # run all evaluations
    python evaluation.py --component intent # run only intent classifier eval
    python evaluation.py --component verifier
    python evaluation.py --component e2e
    python evaluation.py --verbose          # detailed per-case output
"""

import argparse
import json
import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ── project imports ──────────────────────────────────────────────────────────
from implementations.intent_classifier import (
    IntentClassifier,
    AdaptiveRouter,
    QueryIntent,
    RAGStrategy,
)
from implementations.hallucination_verifier import (
    FinGroundVerifier,
    AtomicClaim,
    ClaimType,
)


# ============================================================================
# Data classes for evaluation results
# ============================================================================

@dataclass
class ClassificationMetrics:
    """Per-class precision / recall / F1."""
    label: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class VerifierTestCase:
    """A single verifier evaluation case."""
    description: str
    answer: str
    context: List[str]
    expected_claim_count: Optional[int] = None
    expected_verified_count: Optional[int] = None
    expected_unverified_count: Optional[int] = None


@dataclass
class E2ETestCase:
    """End-to-end pipeline test case (offline — no live services needed)."""
    query: str
    context_chunks: List[str]
    expected_intent: str
    expected_strategy: str
    # Ground-truth claims: list of (claim_text, should_be_verified)
    ground_truth_claims: List[Tuple[str, bool]] = field(default_factory=list)


# ============================================================================
# 1. Intent Classifier Evaluation
# ============================================================================

# Comprehensive test set covering edge cases
INTENT_TEST_SET: List[Tuple[str, QueryIntent, RAGStrategy]] = [
    # ── General Knowledge ──
    ("What does Apple do?", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("Explain Apple's business model", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("Describe the company overview", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("Tell me about Tesla's mission statement", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),
    ("What is a balance sheet?", QueryIntent.GENERAL_KNOWLEDGE, RAGStrategy.SIMPLE_RAG),

    # ── Numerical / Table ──
    ("Show iPhone revenue by quarter", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("What was Apple's revenue in 2024?", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("Break down revenue by segment", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("What was the gross margin last quarter?", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("Net income for fiscal year 2023", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("What is the EPS for Q3?", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),
    ("Revenue breakdown by product division", QueryIntent.NUMERICAL_TABLE, RAGStrategy.MULTIMODAL_RAG),

    # ── Comparison ──
    ("Apple vs Microsoft market cap", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),
    ("Compare revenue of Apple and Google", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),
    ("What is the difference between GAAP and non-GAAP earnings?", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),
    ("How does Tesla's ratio compare to the industry?", QueryIntent.COMPARISON, RAGStrategy.MULTIMODAL_RAG),

    # ── Trend ──
    ("Revenue growth trend 2024", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("How has revenue changed over time?", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("Show year-over-year growth in earnings", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("What is the trajectory of operating margins?", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),
    ("QoQ increase in cloud revenue", QueryIntent.TREND, RAGStrategy.MULTIMODAL_RAG),

    # ── Relationship ──
    ("Who are Apple's suppliers?", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("List Apple's key partners and collaborations", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("Which companies did Apple acquire?", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("Who are the main competitors of Tesla?", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),
    ("Apple's supply chain partners", QueryIntent.RELATIONSHIP, RAGStrategy.GRAPH_RAG),

    # ── Entity ──
    ("Who is Apple's CEO?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("Where is Tesla headquartered?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("Who founded Microsoft?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("What industry is Nvidia in?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
    ("How many employees does Google have?", QueryIntent.ENTITY, RAGStrategy.GRAPH_RAG),
]


def evaluate_intent_classifier(verbose: bool = False) -> Dict:
    """
    Evaluate the IntentClassifier on a curated test set.

    Returns dict with per-class and aggregate metrics.
    """
    classifier = IntentClassifier()

    # Per-class metrics
    metrics: Dict[str, ClassificationMetrics] = {
        intent.value: ClassificationMetrics(label=intent.value)
        for intent in QueryIntent
    }

    correct = 0
    total = len(INTENT_TEST_SET)
    strategy_correct = 0
    latencies: List[float] = []
    failures: List[dict] = []

    for query, expected_intent, expected_strategy in INTENT_TEST_SET:
        t0 = time.perf_counter()
        result = classifier.classify(query)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        intent_ok = result.intent == expected_intent
        strategy_ok = result.strategy == expected_strategy

        if intent_ok:
            correct += 1
            metrics[expected_intent.value].true_positives += 1
        else:
            metrics[expected_intent.value].false_negatives += 1
            metrics[result.intent.value].false_positives += 1
            failures.append({
                "query": query,
                "expected": expected_intent.value,
                "predicted": result.intent.value,
                "confidence": result.confidence,
            })

        if strategy_ok:
            strategy_correct += 1

        if verbose:
            status = "✓" if intent_ok else "✗"
            print(f"  {status} [{latency_ms:5.1f}ms] {query}")
            if not intent_ok:
                print(f"      expected={expected_intent.value}  got={result.intent.value}")

    # Aggregate
    accuracy = correct / total if total else 0
    strategy_accuracy = strategy_correct / total if total else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    # Macro-average F1
    f1_scores = [m.f1 for m in metrics.values() if (m.true_positives + m.false_negatives) > 0]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return {
        "accuracy": accuracy,
        "strategy_accuracy": strategy_accuracy,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "total": total,
        "correct": correct,
        "per_class": {k: {"precision": v.precision, "recall": v.recall, "f1": v.f1} for k, v in metrics.items()},
        "failures": failures,
    }


# ============================================================================
# 2. Hallucination Verifier Evaluation
# ============================================================================

VERIFIER_TEST_CASES: List[VerifierTestCase] = [
    # ── Case 1: All claims verifiable ──
    VerifierTestCase(
        description="All claims match context exactly",
        answer="Apple's revenue was $383.29 billion in fiscal year 2024. Gross margin was 44.1%.",
        context=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
            "Gross margin was 44.1% compared to 44.9% in fiscal 2023.",
        ],
        expected_verified_count=2,
        expected_unverified_count=0,
    ),

    # ── Case 2: Hallucinated number ──
    VerifierTestCase(
        description="Hallucinated revenue figure (not in context)",
        answer="Apple's revenue was $500 billion in 2024.",
        context=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
        ],
        expected_unverified_count=1,
    ),

    # ── Case 3: Mixed verified / unverified ──
    VerifierTestCase(
        description="Mix of correct and hallucinated claims",
        answer=(
            "Apple's revenue was $383.29 billion in fiscal year 2024, up 2% year-over-year. "
            "The company plans to launch a foldable iPhone in 2025."
        ),
        context=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
            "Revenue grew 2% compared to the prior fiscal year.",
        ],
    ),

    # ── Case 4: Near-match number (subtle hallucination) ──
    VerifierTestCase(
        description="Near-match number — 383 vs 383.29 (within fuzzy threshold)",
        answer="Apple's revenue was $383 billion.",
        context=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
        ],
    ),

    # ── Case 5: Temporal mismatch ──
    VerifierTestCase(
        description="Correct number, wrong date",
        answer="Apple's revenue was $383.29 billion in fiscal year 2023.",
        context=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
        ],
    ),

    # ── Case 6: Comparative claim ──
    VerifierTestCase(
        description="Comparative percentage claim",
        answer="Revenue increased 5% year-over-year.",
        context=[
            "Revenue for FY2024 was $383 billion, up from $365 billion in FY2023.",
        ],
    ),

    # ── Case 7: Entity / regulatory claim ──
    VerifierTestCase(
        description="Regulatory claim verification",
        answer="Apple files annual 10-K reports with the SEC.",
        context=[
            "As a public company, Apple files annual 10-K reports with the SEC.",
        ],
        expected_verified_count=1,
    ),

    # ── Case 8: No context available ──
    VerifierTestCase(
        description="No relevant context — everything should be unverified",
        answer="Microsoft's cloud revenue was $25 billion in Q3 2024.",
        context=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
        ],
        expected_verified_count=0,
    ),
]


def evaluate_verifier(verbose: bool = False) -> Dict:
    """
    Evaluate the FinGroundVerifier on curated test cases.

    Measures:
      - Claim decomposition count
      - Verification precision & recall (where ground truth is available)
      - Per-type verification method distribution
    """
    verifier = FinGroundVerifier()  # No LLM — uses rule-based fallback

    results = []
    method_counts: Dict[str, int] = defaultdict(int)
    type_counts: Dict[str, int] = defaultdict(int)
    total_claims = 0
    total_verified = 0
    total_unverified = 0

    for case in VERIFIER_TEST_CASES:
        # Decompose
        claims = verifier.decompose(case.answer)

        # Verify
        verified_claims = verifier.verify(claims, case.context)

        n_verified = sum(1 for c in verified_claims if c.verified)
        n_unverified = sum(1 for c in verified_claims if not c.verified)

        total_claims += len(verified_claims)
        total_verified += n_verified
        total_unverified += n_unverified

        # Track methods and types
        for c in verified_claims:
            if c.verification_method:
                method_counts[c.verification_method] += 1
            type_counts[c.claim_type.value] += 1

        # Check expectations
        decomposition_ok = True
        verification_ok = True

        if case.expected_claim_count is not None:
            decomposition_ok = len(claims) == case.expected_claim_count

        if case.expected_verified_count is not None:
            verification_ok = n_verified == case.expected_verified_count

        if case.expected_unverified_count is not None:
            verification_ok = verification_ok and (n_unverified == case.expected_unverified_count)

        case_result = {
            "description": case.description,
            "original_answer": case.answer,
            "claims_found": len(claims),
            "verified": n_verified,
            "unverified": n_unverified,
            "decomposition_ok": decomposition_ok,
            "verification_ok": verification_ok,
            "claim_details": [
                {
                    "text": c.text[:80],
                    "type": c.claim_type.value,
                    "verified": c.verified,
                    "method": c.verification_method,
                    "evidence": c.supporting_evidence,
                }
                for c in verified_claims
            ],
        }
        results.append(case_result)

        if verbose:
            status = "✓" if verification_ok else "✗"
            print(f"\n  {status} {case.description}")
            print(f"    Claims: {len(claims)} | Verified: {n_verified} | Unverified: {n_unverified}")
            for c in verified_claims:
                v = "✓" if c.verified else "✗"
                print(f"      {v} [{c.claim_type.value}] {c.text[:60]}...")
                print(f"        method={c.verification_method}  evidence={c.supporting_evidence}")

    # Regeneration test
    regen_answer, regen_confidence = verifier.regenerate_verified(
        verifier.verify(verifier.decompose(VERIFIER_TEST_CASES[0].answer), VERIFIER_TEST_CASES[0].context),
        "What was Apple's revenue?",
    )

    return {
        "total_cases": len(VERIFIER_TEST_CASES),
        "total_claims": total_claims,
        "total_verified": total_verified,
        "total_unverified": total_unverified,
        "verification_rate": total_verified / total_claims if total_claims else 0,
        "method_distribution": dict(method_counts),
        "type_distribution": dict(type_counts),
        "regeneration_test": {
            "answer_length": len(regen_answer),
            "confidence": regen_confidence,
            "non_empty": len(regen_answer) > 0,
        },
        "case_results": results,
    }


# ============================================================================
# 3. End-to-End Pipeline Evaluation (Offline)
# ============================================================================

E2E_TEST_CASES: List[E2ETestCase] = [
    E2ETestCase(
        query="What was Apple's revenue in 2024?",
        context_chunks=[
            "Apple Inc. reported revenue of $383.29 billion for fiscal year 2024.",
            "Revenue grew 2% compared to the prior fiscal year.",
            "Services segment revenue reached $96.17 billion.",
        ],
        expected_intent="numerical_table",
        expected_strategy="multimodal_rag",
        ground_truth_claims=[
            ("Revenue was $383.29 billion", True),
            ("in fiscal year 2024", True),
        ],
    ),
    E2ETestCase(
        query="Who are Apple's key suppliers?",
        context_chunks=[
            "Apple's major suppliers include TSMC for chip fabrication.",
            "Foxconn is the primary assembly partner for iPhones.",
        ],
        expected_intent="relationship",
        expected_strategy="graphrag",
    ),
    E2ETestCase(
        query="Revenue growth trend over the last 3 years",
        context_chunks=[
            "FY2022 revenue: $394.3B. FY2023 revenue: $383.3B. FY2024 revenue: $391.0B.",
            "Revenue declined 2.8% in FY2023 before recovering 2% in FY2024.",
        ],
        expected_intent="trend",
        expected_strategy="multimodal_rag",
    ),
    E2ETestCase(
        query="Compare Apple and Microsoft cloud revenue",
        context_chunks=[
            "Apple Services revenue was $96.17 billion in FY2024.",
            "Microsoft Intelligent Cloud revenue was $105.4 billion in FY2024.",
        ],
        expected_intent="comparison",
        expected_strategy="multimodal_rag",
    ),
    E2ETestCase(
        query="What does Apple do?",
        context_chunks=[
            "Apple Inc. designs, manufactures, and markets smartphones, tablets, and computers.",
            "The company also provides digital content and software services.",
        ],
        expected_intent="general_knowledge",
        expected_strategy="simple_rag",
    ),
]


def evaluate_e2e(verbose: bool = False) -> Dict:
    """
    Offline end-to-end evaluation.

    Tests intent routing + verifier together without needing live Qdrant / Neo4j.
    """
    classifier = IntentClassifier()
    verifier = FinGroundVerifier()  # rule-based (no LLM)

    results = []
    intent_correct = 0
    strategy_correct = 0

    for case in E2E_TEST_CASES:
        # Step 1: Intent classification
        routing = classifier.classify(case.query)
        i_ok = routing.intent.value == case.expected_intent
        s_ok = routing.strategy.value == case.expected_strategy

        if i_ok:
            intent_correct += 1
        if s_ok:
            strategy_correct += 1

        # Step 2: Simulate answer generation (concatenate context as proxy)
        simulated_answer = " ".join(case.context_chunks)

        # Step 3: Verify
        claims = verifier.decompose(simulated_answer)
        verified_claims = verifier.verify(claims, case.context_chunks)
        verified_answer, confidence = verifier.regenerate_verified(verified_claims, case.query)

        n_verified = sum(1 for c in verified_claims if c.verified)
        n_total = len(verified_claims)
        faithfulness = n_verified / n_total if n_total else 0

        case_result = {
            "query": case.query,
            "draft_answer": simulated_answer,
            "verified_answer": verified_answer,
            "intent_correct": i_ok,
            "strategy_correct": s_ok,
            "predicted_intent": routing.intent.value,
            "predicted_strategy": routing.strategy.value,
            "routing_confidence": routing.confidence,
            "claims_total": n_total,
            "claims_verified": n_verified,
            "faithfulness": faithfulness,
            "answer_confidence": confidence,
            "answer_non_empty": len(verified_answer.strip()) > 0,
        }
        results.append(case_result)

        if verbose:
            i_s = "✓" if i_ok else "✗"
            s_s = "✓" if s_ok else "✗"
            print(f"\n  Query: {case.query}")
            print(f"    Intent:   {i_s} {routing.intent.value} (expected {case.expected_intent})")
            print(f"    Strategy: {s_s} {routing.strategy.value} (expected {case.expected_strategy})")
            print(f"    Claims: {n_verified}/{n_total} verified | Faithfulness: {faithfulness:.0%}")
            print(f"    Confidence: {confidence:.2f}")

    total = len(E2E_TEST_CASES)

    return {
        "total_cases": total,
        "intent_accuracy": intent_correct / total if total else 0,
        "strategy_accuracy": strategy_correct / total if total else 0,
        "avg_faithfulness": (
            sum(r["faithfulness"] for r in results) / total if total else 0
        ),
        "avg_confidence": (
            sum(r["answer_confidence"] for r in results) / total if total else 0
        ),
        "all_answers_non_empty": all(r["answer_non_empty"] for r in results),
        "case_results": results,
    }


# ============================================================================
# Pretty Printer
# ============================================================================

def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_metrics(data: Dict, indent: int = 2):
    prefix = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_metrics(v, indent + 4)
        elif isinstance(v, list):
            print(f"{prefix}{k}: [{len(v)} items]")
        elif isinstance(v, float):
            print(f"{prefix}{k}: {v:.4f}")
        else:
            print(f"{prefix}{k}: {v}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unify RAG Evaluation Suite")
    parser.add_argument(
        "--component",
        choices=["intent", "verifier", "e2e", "all"],
        default="all",
        help="Which component to evaluate",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case details")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    all_results = {}

    # ── Intent Classifier ──
    if args.component in ("intent", "all"):
        print_section("1. Intent Classifier Evaluation")
        intent_results = evaluate_intent_classifier(verbose=args.verbose)
        all_results["intent_classifier"] = intent_results

        print(f"\n  Accuracy:          {intent_results['accuracy']:.1%} ({intent_results['correct']}/{intent_results['total']})")
        print(f"  Strategy Accuracy: {intent_results['strategy_accuracy']:.1%}")
        print(f"  Macro F1:          {intent_results['macro_f1']:.4f}")
        print(f"  Avg Latency:       {intent_results['avg_latency_ms']:.2f} ms")
        print(f"  P95 Latency:       {intent_results['p95_latency_ms']:.2f} ms")

        if intent_results["failures"]:
            print(f"\n  Failures ({len(intent_results['failures'])}):")
            for f in intent_results["failures"]:
                print(f"    ✗ \"{f['query']}\"  →  predicted={f['predicted']}  expected={f['expected']}")

        print("\n  Per-Class Metrics:")
        for cls, m in intent_results["per_class"].items():
            print(f"    {cls:25s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    # ── Hallucination Verifier ──
    if args.component in ("verifier", "all"):
        print_section("2. Hallucination Verifier Evaluation")
        verifier_results = evaluate_verifier(verbose=args.verbose)
        all_results["hallucination_verifier"] = verifier_results

        print(f"\n  Test Cases:        {verifier_results['total_cases']}")
        print(f"  Total Claims:      {verifier_results['total_claims']}")
        print(f"  Verified:          {verifier_results['total_verified']}")
        print(f"  Unverified:        {verifier_results['total_unverified']}")
        print(f"  Verification Rate: {verifier_results['verification_rate']:.1%}")

        print("\n  Verification Methods:")
        for method, count in sorted(verifier_results["method_distribution"].items()):
            print(f"    {method:40s}  {count}")

        print("\n  Claim Types:")
        for ctype, count in sorted(verifier_results["type_distribution"].items()):
            print(f"    {ctype:20s}  {count}")

        regen = verifier_results["regeneration_test"]
        print(f"\n  Regeneration Test:")
        print(f"    Answer non-empty: {regen['non_empty']}")
        print(f"    Confidence:       {regen['confidence']:.2f}")

    # ── End-to-End ──
    if args.component in ("e2e", "all"):
        print_section("3. End-to-End Pipeline Evaluation (Offline)")
        e2e_results = evaluate_e2e(verbose=args.verbose)
        all_results["end_to_end"] = e2e_results

        print(f"\n  Test Cases:          {e2e_results['total_cases']}")
        print(f"  Intent Accuracy:     {e2e_results['intent_accuracy']:.1%}")
        print(f"  Strategy Accuracy:   {e2e_results['strategy_accuracy']:.1%}")
        print(f"  Avg Faithfulness:    {e2e_results['avg_faithfulness']:.1%}")
        print(f"  Avg Confidence:      {e2e_results['avg_confidence']:.2f}")
        print(f"  All Answers Valid:   {e2e_results['all_answers_non_empty']}")

    # ── Summary ──
    print_section("Summary")
    if "intent_classifier" in all_results:
        print(f"  Intent Classifier:      {all_results['intent_classifier']['accuracy']:.1%} accuracy")
    if "hallucination_verifier" in all_results:
        print(f"  Hallucination Verifier: {all_results['hallucination_verifier']['verification_rate']:.1%} verification rate")
    if "end_to_end" in all_results:
        print(f"  E2E Faithfulness:       {all_results['end_to_end']['avg_faithfulness']:.1%}")
    print()

    # ── JSON output ──
    if args.json:
        # Strip non-serializable bits
        print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
