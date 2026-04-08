"""
test_lead_engine.py - Chapter 2: The Restaurant at the End of the Pipeline
DSPy: The Mostly Harmless Guide

Tests that verify all Chapter 2 code patterns work correctly.
Run with: poetry run python test_lead_engine.py
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

import dspy
from pydantic import ValidationError

from lead_engine import (
    CompanyIntel,
    IntentClassification,
    ResearchCompany,
    ClassifyIntent,
    ScoreLead,
    GenerateOutreach,
    LeadIntelligenceEngine,
)


def test_pydantic_models():
    """Verify Pydantic models validate correctly."""
    print("TEST: Pydantic models...")

    intel = CompanyIntel(
        name="Test Corp",
        industry="Technology",
        size_estimate="startup",
        recent_developments=["Launched v2", "Raised Series A"],
        potential_pain_points=["Scaling infrastructure"],
        tech_stack_signals=["Python", "AWS"],
    )
    assert intel.name == "Test Corp"
    assert intel.size_estimate == "startup"
    assert len(intel.recent_developments) == 2
    print("  PASSED: CompanyIntel creation works")


def test_literal_constraints():
    """Verify Literal types reject invalid values."""
    print("TEST: Literal constraints...")

    try:
        CompanyIntel(
            name="X",
            industry="Y",
            size_estimate="huge",  # Not in Literal
            recent_developments=[],
            potential_pain_points=[],
            tech_stack_signals=[],
        )
        print("  FAILED: Should have rejected 'huge'")
        return False
    except ValidationError:
        print("  PASSED: Pydantic rejects invalid Literal values")

    try:
        IntentClassification(
            intent_level="blazing",  # Not in Literal
            intent_signals=[],
            buyer_persona="Test",
        )
        print("  FAILED: Should have rejected 'blazing'")
        return False
    except ValidationError:
        print("  PASSED: IntentClassification Literal works too")

    return True


def test_signature_fields():
    """Verify all Signatures have correct input/output fields."""
    print("TEST: Signature fields...")

    # ResearchCompany
    assert "company_name" in ResearchCompany.input_fields
    assert "prospect_role" in ResearchCompany.input_fields
    assert "intel" in ResearchCompany.output_fields

    # ClassifyIntent
    assert "company_intel" in ClassifyIntent.input_fields
    assert "engagement_context" in ClassifyIntent.input_fields
    assert "classification" in ClassifyIntent.output_fields

    # ScoreLead
    assert "company_intel" in ScoreLead.input_fields
    assert "classification" in ScoreLead.input_fields
    assert "lead_score" in ScoreLead.output_fields
    assert "recommended_action" in ScoreLead.output_fields

    # GenerateOutreach
    assert "prospect_name" in GenerateOutreach.input_fields
    assert "subject_line" in GenerateOutreach.output_fields
    assert "email_body" in GenerateOutreach.output_fields

    print("  PASSED: All 4 Signatures have correct fields")


def test_module_structure():
    """Verify the pipeline module has correct predictors."""
    print("TEST: Module structure...")

    engine = LeadIntelligenceEngine()
    named = engine.named_predictors()
    names = [n for n, _ in named]

    assert len(named) == 4, f"Expected 4 predictors, got {len(named)}"
    assert "research.predict" in names, "Missing research predictor"
    assert "classify.predict" in names, "Missing classify predictor"
    assert "score" in names, "Missing score predictor"
    assert "outreach" in names, "Missing outreach predictor"

    print("  PASSED: 4 predictors found (research, classify, score, outreach)")


def test_per_predictor_lm():
    """Verify different LMs can be assigned to different steps."""
    print("TEST: Per-predictor LM assignment...")

    engine = LeadIntelligenceEngine()
    powerful = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7)
    fast = dspy.LM("anthropic/claude-haiku-4-5-20251001", temperature=0.3)

    engine.research.set_lm(powerful)
    engine.classify.set_lm(fast)
    engine.score.lm = fast
    engine.outreach.lm = powerful

    assert engine.research.predict.lm == powerful
    assert engine.classify.predict.lm == fast
    assert engine.score.lm == fast
    assert engine.outreach.lm == powerful

    print("  PASSED: Per-predictor LM assignment works")


def test_adapters():
    """Verify adapter instantiation and configuration."""
    print("TEST: Adapter configuration...")

    lm = dspy.LM("anthropic/claude-sonnet-4-6")

    # ChatAdapter (default)
    dspy.configure(lm=lm)
    assert True  # No error means it worked

    # JSONAdapter
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    # XMLAdapter
    dspy.configure(lm=lm, adapter=dspy.XMLAdapter())

    print("  PASSED: All 3 adapters configure without error")


def test_live_pipeline():
    """Test the full pipeline with live API calls."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("TEST: Live pipeline... SKIPPED (no ANTHROPIC_API_KEY)")
        return

    print("TEST: Live pipeline (4 LLM calls)...")

    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)

    engine = LeadIntelligenceEngine()
    result = engine(
        company_name="Notion",
        prospect_name="Alex Rivera",
        prospect_role="Head of Platform",
        engagement_context="Attended our webinar on API scaling patterns",
    )

    # Verify all outputs
    assert isinstance(result.intel, CompanyIntel)
    assert result.intel.size_estimate in ["startup", "smb", "mid-market", "enterprise"]
    assert isinstance(result.classification, IntentClassification)
    assert result.classification.intent_level in ["hot", "warm", "cold", "unknown"]
    assert isinstance(result.lead_score, int)
    assert 1 <= result.lead_score <= 100
    assert result.recommended_action in [
        "fast_track", "standard_nurture", "long_term_nurture", "disqualify"
    ]
    assert result.subject_line is not None
    assert result.email_body is not None

    print(f"  Company: {result.intel.name} ({result.intel.size_estimate})")
    print(f"  Intent: {result.classification.intent_level}")
    print(f"  Score: {result.lead_score}/100 -> {result.recommended_action}")
    print(f"  Subject: {result.subject_line}")
    print("  PASSED: Full pipeline completed with valid outputs")


def main():
    print("=" * 60)
    print("Chapter 2 Tests: The Restaurant at the End of the Pipeline")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print()

    tests = [
        test_pydantic_models,
        test_literal_constraints,
        test_signature_fields,
        test_module_structure,
        test_per_predictor_lm,
        test_adapters,
        test_live_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
