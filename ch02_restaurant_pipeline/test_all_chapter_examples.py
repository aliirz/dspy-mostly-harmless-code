"""
test_all_chapter_examples.py - Chapter 2: EVERY code block tested
DSPy: The Mostly Harmless Guide

This file tests EVERY code example from the chapter markdown.
If it appears in the chapter, it gets run here.
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

import dspy
from pydantic import BaseModel, Field
from typing import Literal

from lead_engine import (
    CompanyIntel, IntentClassification,
    ResearchCompany, ClassifyIntent, ScoreLead, GenerateOutreach,
    LeadIntelligenceEngine,
)

API_KEY = os.getenv("ANTHROPIC_API_KEY")


def setup_lm():
    lm = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7, max_tokens=2000)
    dspy.configure(lm=lm)
    return lm


# ============================================================
# EXAMPLE 1: Summarizer (line 24-32 of chapter)
# ============================================================
def test_summarizer():
    """Test the basic Summarizer module from the chapter."""
    print("TEST: Summarizer module...")

    class Summarizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.summarize = dspy.Predict("text -> summary")

        def forward(self, text):
            return self.summarize(text=text)

    lm = setup_lm()
    s = Summarizer()
    result = s(text="DSPy is a framework for programming language models. "
                    "It replaces prompt engineering with structured programming.")
    assert result.summary is not None
    assert len(result.summary) > 10
    print(f"  Summary: {result.summary[:80]}...")
    print("  PASSED")


# ============================================================
# EXAMPLE 2: ResearchAndSummarize (line 37-46 of chapter)
# ============================================================
def test_research_and_summarize():
    """Test the chained ResearchAndSummarize module."""
    print("TEST: ResearchAndSummarize module...")

    class ResearchAndSummarize(dspy.Module):
        def __init__(self):
            super().__init__()
            self.research = dspy.ChainOfThought("topic -> findings")
            self.summarize = dspy.Predict("findings -> summary")

        def forward(self, topic):
            findings = self.research(topic=topic)
            return self.summarize(findings=findings.findings)

    lm = setup_lm()
    rs = ResearchAndSummarize()
    result = rs(topic="Benefits of using DSPy over raw prompt engineering")
    assert result.summary is not None
    assert len(result.summary) > 20
    print(f"  Summary: {result.summary[:80]}...")
    print("  PASSED")


# ============================================================
# EXAMPLE 3: Full LeadIntelligenceEngine (already tested, but run again)
# ============================================================
def test_lead_intelligence_engine():
    """Test the main pipeline from the chapter."""
    print("TEST: LeadIntelligenceEngine full pipeline...")

    lm = setup_lm()
    engine = LeadIntelligenceEngine()

    result = engine(
        company_name="Stripe",
        prospect_name="Sarah Chen",
        prospect_role="VP of Engineering",
        engagement_context="Downloaded our API performance whitepaper and "
                           "visited the enterprise pricing page twice this week",
    )

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

    print(f"  Score: {result.lead_score}/100 -> {result.recommended_action}")
    print("  PASSED")


# ============================================================
# EXAMPLE 4: Adapter switching (line 370-402 of chapter)
# ============================================================
def test_adapter_switching():
    """Test running with JSONAdapter as shown in chapter."""
    print("TEST: Adapter switching (JSONAdapter)...")

    lm = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7, max_tokens=2000)
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    engine = LeadIntelligenceEngine()
    result = engine(
        company_name="Notion",
        prospect_name="Alex Rivera",
        prospect_role="Head of Platform",
        engagement_context="Requested a demo after seeing our talk at a conference",
    )

    assert isinstance(result.intel, CompanyIntel)
    assert isinstance(result.lead_score, int)
    assert result.subject_line is not None

    print(f"  Score: {result.lead_score}/100 (via JSONAdapter)")
    print("  PASSED")

    # Reset to default adapter
    dspy.configure(lm=lm)


# ============================================================
# EXAMPLE 5: Auto-fallback disable (line 414-417 of chapter)
# ============================================================
def test_auto_fallback_config():
    """Test that auto-fallback config works (structural only)."""
    print("TEST: Auto-fallback configuration...")

    lm = dspy.LM("anthropic/claude-sonnet-4-6")
    adapter = dspy.ChatAdapter()
    adapter.use_json_adapter_fallback = False
    dspy.configure(lm=lm, adapter=adapter)
    assert adapter.use_json_adapter_fallback is False

    # Reset
    dspy.configure(lm=lm)
    print("  PASSED")


# ============================================================
# EXAMPLE 6: CostOptimizedEngine with dspy.context() (line 429-484)
# ============================================================
def test_cost_optimized_engine():
    """Test the CostOptimizedEngine EXACTLY as written in the chapter."""
    print("TEST: CostOptimizedEngine with dspy.context()...")

    class CostOptimizedEngine(dspy.Module):
        def __init__(self, powerful_lm, fast_lm):
            super().__init__()
            self.powerful_lm = powerful_lm
            self.fast_lm = fast_lm

            self.research = dspy.ChainOfThought(ResearchCompany)
            self.classify = dspy.ChainOfThought(ClassifyIntent)
            self.score = dspy.Predict(ScoreLead)
            self.outreach = dspy.Predict(GenerateOutreach)

        def forward(
            self,
            company_name: str,
            prospect_name: str,
            prospect_role: str,
            engagement_context: str,
        ) -> dspy.Prediction:

            # Research needs the powerful model
            with dspy.context(lm=self.powerful_lm):
                research_result = self.research(
                    company_name=company_name,
                    prospect_role=prospect_role,
                )

            # Classification and scoring are simpler — use the fast model
            with dspy.context(lm=self.fast_lm):
                classify_result = self.classify(
                    company_intel=research_result.intel,
                    prospect_role=prospect_role,
                    engagement_context=engagement_context,
                )
                score_result = self.score(
                    company_intel=research_result.intel,
                    classification=classify_result.classification,
                )

            # Outreach needs creativity — back to the powerful model
            with dspy.context(lm=self.powerful_lm):
                outreach_result = self.outreach(
                    company_intel=research_result.intel,
                    classification=classify_result.classification,
                    lead_score=score_result.lead_score,
                    prospect_name=prospect_name,
                )

            return dspy.Prediction(
                intel=research_result.intel,
                classification=classify_result.classification,
                lead_score=score_result.lead_score,
                score_reasoning=score_result.score_reasoning,
                recommended_action=score_result.recommended_action,
                subject_line=outreach_result.subject_line,
                email_body=outreach_result.email_body,
            )

    # Exactly as shown in "Usage:" section
    powerful = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7, max_tokens=2000)
    fast = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.0, max_tokens=1000)
    dspy.configure(lm=powerful)

    engine = CostOptimizedEngine(powerful_lm=powerful, fast_lm=fast)
    result = engine(
        company_name="Figma",
        prospect_name="Jordan Park",
        prospect_role="CTO",
        engagement_context="Visited pricing page and requested a demo",
    )

    assert isinstance(result.intel, CompanyIntel)
    assert isinstance(result.lead_score, int)
    assert result.subject_line is not None

    # Verify LM routing: 2 powerful (research + outreach), 2 fast (classify + score)
    assert len(powerful.history) == 2, (
        f"Expected 2 powerful calls, got {len(powerful.history)}"
    )
    assert len(fast.history) == 2, (
        f"Expected 2 fast calls, got {len(fast.history)}"
    )

    print(f"  Score: {result.lead_score}/100")
    print(f"  LM routing: {len(powerful.history)} powerful + {len(fast.history)} fast")
    print("  PASSED")


# ============================================================
# EXAMPLE 7: SmartLeadEngine branching (line 522-574)
# ============================================================
def test_smart_lead_engine_branching():
    """Test the branching pipeline from the chapter."""
    print("TEST: SmartLeadEngine branching pipeline...")

    # We need a GenerateNurtureSequence signature that the chapter references
    class GenerateNurtureSequence(dspy.Signature):
        """Generate a nurture email sequence plan for a prospect who
        isn't ready to buy yet."""

        company_intel: CompanyIntel = dspy.InputField(
            desc="Company intelligence"
        )
        classification: IntentClassification = dspy.InputField(
            desc="Intent classification"
        )
        prospect_name: str = dspy.InputField(desc="Prospect's name")
        plan: str = dspy.OutputField(
            desc="A 3-step nurture sequence plan with timing and content themes"
        )

    class SmartLeadEngine(dspy.Module):
        def __init__(self):
            super().__init__()
            self.research = dspy.ChainOfThought(ResearchCompany)
            self.classify = dspy.ChainOfThought(ClassifyIntent)
            self.score = dspy.Predict(ScoreLead)
            self.outreach = dspy.Predict(GenerateOutreach)
            self.nurture = dspy.Predict(GenerateNurtureSequence)

        def forward(self, company_name, prospect_name, prospect_role,
                    engagement_context):
            research_result = self.research(
                company_name=company_name,
                prospect_role=prospect_role,
            )
            classify_result = self.classify(
                company_intel=research_result.intel,
                prospect_role=prospect_role,
                engagement_context=engagement_context,
            )
            score_result = self.score(
                company_intel=research_result.intel,
                classification=classify_result.classification,
            )

            # Branch based on score
            if score_result.lead_score >= 70:
                outreach = self.outreach(
                    company_intel=research_result.intel,
                    classification=classify_result.classification,
                    lead_score=score_result.lead_score,
                    prospect_name=prospect_name,
                )
                return dspy.Prediction(
                    path="direct_outreach",
                    lead_score=score_result.lead_score,
                    subject_line=outreach.subject_line,
                    email_body=outreach.email_body,
                )
            else:
                nurture = self.nurture(
                    company_intel=research_result.intel,
                    classification=classify_result.classification,
                    prospect_name=prospect_name,
                )
                return dspy.Prediction(
                    path="nurture_sequence",
                    lead_score=score_result.lead_score,
                    nurture_plan=nurture.plan,
                )

    lm = setup_lm()
    smart_engine = SmartLeadEngine()

    # Test with a likely hot lead
    result = smart_engine(
        company_name="Stripe",
        prospect_name="Sarah Chen",
        prospect_role="VP of Engineering",
        engagement_context="Downloaded whitepaper and visited pricing page 3 times",
    )

    assert result.lead_score is not None
    assert result.path in ["direct_outreach", "nurture_sequence"]

    if result.path == "direct_outreach":
        assert result.subject_line is not None
        assert result.email_body is not None
        print(f"  Score: {result.lead_score}/100 -> direct_outreach")
        print(f"  Subject: {result.subject_line}")
    else:
        assert result.nurture_plan is not None
        print(f"  Score: {result.lead_score}/100 -> nurture_sequence")
        print(f"  Plan: {result.nurture_plan[:80]}...")

    print("  PASSED")


# ============================================================
# EXAMPLE 8: Parallel / batch execution (line 612-658)
# ============================================================
def test_batch_execution():
    """Test the batch execution pattern from the chapter."""
    print("TEST: Batch execution (2 prospects)...")

    lm = setup_lm()
    engine = LeadIntelligenceEngine()

    # Using batch() as shown in chapter (fewer prospects to save API calls)
    prospects = [
        dspy.Example(
            company_name="Stripe",
            prospect_name="Sarah Chen",
            prospect_role="VP of Engineering",
            engagement_context="Downloaded whitepaper",
        ).with_inputs("company_name", "prospect_name", "prospect_role",
                      "engagement_context"),
        dspy.Example(
            company_name="Notion",
            prospect_name="Alex Rivera",
            prospect_role="Head of Platform",
            engagement_context="Attended webinar",
        ).with_inputs("company_name", "prospect_name", "prospect_role",
                      "engagement_context"),
    ]

    results = engine.batch(
        prospects,
        num_threads=2,
        return_failed_examples=False,
    )

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    for i, result in enumerate(results):
        assert isinstance(result.lead_score, int)
        print(f"  {prospects[i].company_name}: Score {result.lead_score}/100")

    print("  PASSED")


# ============================================================
# MAIN
# ============================================================
def main():
    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set. All tests require live API.")
        sys.exit(1)

    print("=" * 60)
    print("Chapter 2: EVERY Code Example Tested")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print()

    tests = [
        test_summarizer,
        test_research_and_summarize,
        test_lead_intelligence_engine,
        test_adapter_switching,
        test_auto_fallback_config,
        test_cost_optimized_engine,
        test_smart_lead_engine_branching,
        test_batch_execution,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
