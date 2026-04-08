"""
test_all_chapter_examples.py — Chapter 4: EVERY code block tested
DSPy: The Mostly Harmless Guide

Tests every code example from the chapter markdown.
Runs all optimizers with live API calls.
"""

import os
import sys
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

import dspy
from pydantic import ValidationError

from ticket_classifier import (
    TicketClassification, ClassifyTicket, TicketClassifier,
    ticket_metric, TRAINSET, VALSET,
    TICKET_CATEGORIES, PRIORITY_LEVELS,
)

API_KEY = os.getenv("ANTHROPIC_API_KEY")


def setup_lm():
    lm = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7, max_tokens=1500)
    dspy.configure(lm=lm)
    return lm


# ============================================================
# EXAMPLE 1: Dataset structure
# ============================================================
def test_dataset():
    """Test that TRAINSET and VALSET are well-formed."""
    print("TEST: Dataset structure...")

    assert len(TRAINSET) == 10, f"Expected 10 train, got {len(TRAINSET)}"
    assert len(VALSET) == 8, f"Expected 8 val, got {len(VALSET)}"

    for ex in TRAINSET + VALSET:
        assert hasattr(ex, 'ticket'), "Missing ticket field"
        assert hasattr(ex, 'category'), "Missing category field"
        assert hasattr(ex, 'priority'), "Missing priority field"
        assert hasattr(ex, 'routing'), "Missing routing field"
        assert ex.category in TICKET_CATEGORIES, f"Bad category: {ex.category}"
        assert ex.priority in PRIORITY_LEVELS, f"Bad priority: {ex.priority}"

    # Check inputs are set correctly
    for ex in TRAINSET:
        assert 'ticket' in ex.inputs(), "ticket should be an input field"
        assert 'category' not in ex.inputs(), "category should NOT be an input"

    print(f"  TRAINSET: {len(TRAINSET)} examples, VALSET: {len(VALSET)} examples")
    print("  All examples well-formed")
    print("  PASSED")


# ============================================================
# EXAMPLE 2: Pydantic TicketClassification model
# ============================================================
def test_pydantic_model():
    """Test the Pydantic model validates correctly."""
    print("TEST: TicketClassification Pydantic model...")

    tc = TicketClassification(
        category="billing",
        priority="high",
        routing="billing_team",
        reasoning="Customer mentions charges and refund",
    )
    assert tc.category == "billing"
    assert tc.priority == "high"
    print(f"  Valid: category={tc.category}, priority={tc.priority}")

    # Invalid category should fail
    try:
        TicketClassification(
            category="invalid_category",
            priority="high",
            routing="test",
            reasoning="test",
        )
        print("  FAILED: Should have rejected invalid category")
        return False
    except ValidationError:
        print("  Literal validation: rejects invalid categories")

    # Invalid priority should fail
    try:
        TicketClassification(
            category="billing",
            priority="urgent",
            routing="test",
            reasoning="test",
        )
        print("  FAILED: Should have rejected invalid priority")
        return False
    except ValidationError:
        print("  Literal validation: rejects invalid priorities")

    print("  PASSED")


# ============================================================
# EXAMPLE 3: ClassifyTicket signature fields
# ============================================================
def test_signature():
    """Test the ClassifyTicket signature has correct fields."""
    print("TEST: ClassifyTicket signature...")

    assert "ticket" in ClassifyTicket.input_fields
    assert "classification" in ClassifyTicket.output_fields
    print("  ClassifyTicket: ticket → classification")
    print("  PASSED")


# ============================================================
# EXAMPLE 4: ticket_metric scoring
# ============================================================
def test_metric():
    """Test the metric function scores correctly."""
    print("TEST: ticket_metric...")

    example = dspy.Example(
        ticket="test",
        category="billing",
        priority="high",
        routing="billing_team",
    ).with_inputs("ticket")

    # Perfect match
    class PerfectPred:
        category = "billing"
        priority = "high"
        routing = "billing_team"

    score = ticket_metric(example, PerfectPred())
    assert score == 1.0, f"Perfect match should be 1.0, got {score}"
    print(f"  Perfect match: {score}")

    # Category only
    class CategoryOnly:
        category = "billing"
        priority = "low"
        routing = "wrong_team"

    score = ticket_metric(example, CategoryOnly())
    assert score == 0.5, f"Category-only should be 0.5, got {score}"
    print(f"  Category only: {score}")

    # Partial routing (team prefix match)
    class PartialRouting:
        category = "billing"
        priority = "high"
        routing = "billing_support"

    score = ticket_metric(example, PartialRouting())
    assert score == 0.9, f"Partial routing should be 0.9, got {score}"
    print(f"  Partial routing: {score}")

    # Complete miss
    class WrongAll:
        category = "technical"
        priority = "low"
        routing = "wrong_team"

    score = ticket_metric(example, WrongAll())
    assert score == 0.0, f"Complete miss should be 0.0, got {score}"
    print(f"  Complete miss: {score}")

    # Trace parameter accepted
    score = ticket_metric(example, PerfectPred(), trace="some_trace")
    assert score == 1.0
    print("  trace= parameter: accepted")

    print("  PASSED")


# ============================================================
# EXAMPLE 5: Baseline classifier (live LM)
# ============================================================
def test_baseline():
    """Test the unoptimized classifier with live API."""
    print("TEST: Baseline TicketClassifier...")

    setup_lm()
    classifier = TicketClassifier()

    result = classifier(ticket="I've been double-charged for my March subscription.")
    assert hasattr(result, 'category')
    assert hasattr(result, 'priority')
    assert hasattr(result, 'routing')
    assert hasattr(result, 'reasoning')
    assert result.category in TICKET_CATEGORIES
    assert result.priority in PRIORITY_LEVELS

    print(f"  Category: {result.category}")
    print(f"  Priority: {result.priority}")
    print(f"  Routing: {result.routing}")
    print(f"  Reasoning: {result.reasoning[:80]}...")
    print("  PASSED")


# ============================================================
# EXAMPLE 6: dspy.Evaluate with baseline
# ============================================================
def test_evaluate_baseline():
    """Test dspy.Evaluate on the baseline classifier."""
    print("TEST: Evaluate baseline...")

    setup_lm()
    classifier = TicketClassifier()

    evaluator = dspy.Evaluate(
        devset=VALSET[:3],  # 3 examples to save API calls
        metric=ticket_metric,
        num_threads=1,
        display_progress=True,
    )

    score = evaluator(classifier)
    print(f"  Baseline score (3 examples): {score}")
    print("  PASSED")


# ============================================================
# EXAMPLE 7: LabeledFewShot optimizer
# ============================================================
def test_labeled_few_shot():
    """Test LabeledFewShot optimizer."""
    print("TEST: LabeledFewShot optimizer...")

    setup_lm()
    from dspy.teleprompt import LabeledFewShot

    classifier = TicketClassifier()
    opt = LabeledFewShot(k=3)
    optimized = opt.compile(student=classifier, trainset=TRAINSET)

    # Verify demos were added
    for name, predictor in optimized.named_predictors():
        assert len(predictor.demos) > 0, f"{name} has no demos"
        print(f"  {name}: {len(predictor.demos)} demos added")

    # Run a prediction
    result = optimized(ticket="The API returns 500 errors since this morning.")
    assert result.category in TICKET_CATEGORIES
    assert result.priority in PRIORITY_LEVELS
    print(f"  Prediction: {result.category}/{result.priority}")

    # Evaluate on small valset
    evaluator = dspy.Evaluate(
        devset=VALSET[:3],
        metric=ticket_metric,
        num_threads=1,
        display_progress=False,
    )
    score = evaluator(optimized)
    print(f"  LabeledFewShot score (3 examples): {score}")
    print("  PASSED")


# ============================================================
# EXAMPLE 8: BootstrapFewShot optimizer
# ============================================================
def test_bootstrap_few_shot():
    """Test BootstrapFewShot optimizer."""
    print("TEST: BootstrapFewShot optimizer...")

    setup_lm()
    from dspy.teleprompt import BootstrapFewShot

    classifier = TicketClassifier()
    opt = BootstrapFewShot(
        metric=ticket_metric,
        metric_threshold=0.8,
        max_bootstrapped_demos=4,
        max_labeled_demos=2,
        max_rounds=1,  # 1 round to save API calls in tests
    )
    optimized = opt.compile(student=classifier, trainset=TRAINSET[:5])

    # Verify it produced something
    for name, predictor in optimized.named_predictors():
        print(f"  {name}: {len(predictor.demos)} demos")

    # Run a prediction
    result = optimized(ticket="How do I reset my password?")
    assert result.category in TICKET_CATEGORIES
    print(f"  Prediction: {result.category}/{result.priority}")

    # Evaluate
    evaluator = dspy.Evaluate(
        devset=VALSET[:3],
        metric=ticket_metric,
        num_threads=1,
        display_progress=False,
    )
    score = evaluator(optimized)
    print(f"  BootstrapFewShot score (3 examples): {score}")
    print("  PASSED")


# ============================================================
# EXAMPLE 9: MIPROv2 optimizer
# ============================================================
def test_mipro_v2():
    """Test MIPROv2 optimizer with auto=light."""
    print("TEST: MIPROv2 optimizer (auto=light)...")

    setup_lm()
    from dspy.teleprompt import MIPROv2

    classifier = TicketClassifier()
    opt = MIPROv2(metric=ticket_metric, auto="light", num_threads=1)
    optimized = opt.compile(
        student=classifier,
        trainset=TRAINSET[:5],
        valset=VALSET[:3],
    )

    # Inspect what MIPROv2 found
    for name, predictor in optimized.named_predictors():
        instruction = getattr(predictor.signature, 'instructions', '')
        print(f"  {name}:")
        if instruction:
            print(f"    Instruction: {str(instruction)[:100]}...")
        print(f"    Demos: {len(predictor.demos)}")

    # Run a prediction
    result = optimized(ticket="Your app crashes on my iPhone when I open settings.")
    assert result.category in TICKET_CATEGORIES
    print(f"  Prediction: {result.category}/{result.priority}")

    # Evaluate
    evaluator = dspy.Evaluate(
        devset=VALSET[:3],
        metric=ticket_metric,
        num_threads=1,
        display_progress=False,
    )
    score = evaluator(optimized)
    print(f"  MIPROv2 score (3 examples): {score}")
    print("  PASSED")


# ============================================================
# EXAMPLE 10: Save and load optimized program
# ============================================================
def test_save_load():
    """Test saving and loading an optimized classifier."""
    print("TEST: Save and Load optimized program...")

    setup_lm()
    from dspy.teleprompt import LabeledFewShot

    # Optimize
    classifier = TicketClassifier()
    opt = LabeledFewShot(k=3)
    optimized = opt.compile(student=classifier, trainset=TRAINSET)

    # Save
    save_path = os.path.join(tempfile.gettempdir(), "test_optimized_classifier.json")
    optimized.save(save_path)
    assert os.path.exists(save_path)
    print(f"  Saved to {save_path}")

    # Load into a fresh classifier
    loaded = TicketClassifier()
    loaded.load(save_path)

    # Verify demos survived the round-trip
    for name, predictor in loaded.named_predictors():
        assert len(predictor.demos) > 0, f"{name} lost its demos"
        print(f"  {name}: {len(predictor.demos)} demos loaded")

    # Run a prediction with the loaded model
    result = loaded(ticket="I need a refund for last month's charge.")
    assert result.category in TICKET_CATEGORIES
    print(f"  Loaded prediction: {result.category}/{result.priority}")

    # Clean up
    try:
        os.remove(save_path)
        print("  Cleaned up")
    except (PermissionError, FileNotFoundError):
        pass

    print("  PASSED")


# ============================================================
# EXAMPLE 11: SIMBA optimizer
# ============================================================
def test_simba():
    """Test SIMBA optimizer."""
    print("TEST: SIMBA optimizer...")

    setup_lm()
    from dspy.teleprompt import SIMBA

    classifier = TicketClassifier()
    opt = SIMBA(
        metric=ticket_metric,
        bsize=3,
        max_steps=1,  # 1 step to save API calls in tests
        max_demos=3,
        num_candidates=2,
        num_threads=1,
    )
    optimized = opt.compile(
        student=classifier,
        trainset=TRAINSET[:5],
    )

    # Run a prediction
    result = optimized(ticket="Please add a calendar integration feature.")
    assert result.category in TICKET_CATEGORIES
    print(f"  Prediction: {result.category}/{result.priority}")

    # Evaluate
    evaluator = dspy.Evaluate(
        devset=VALSET[:3],
        metric=ticket_metric,
        num_threads=1,
        display_progress=False,
    )
    score = evaluator(optimized)
    print(f"  SIMBA score (3 examples): {score}")
    print("  PASSED")


# ============================================================
# EXAMPLE 12: BootstrapFewShotWithRandomSearch
# ============================================================
def test_bootstrap_random_search():
    """Test BootstrapFewShotWithRandomSearch optimizer."""
    print("TEST: BootstrapFewShotWithRandomSearch...")

    setup_lm()
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch

    classifier = TicketClassifier()
    opt = BootstrapFewShotWithRandomSearch(
        metric=ticket_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        num_candidate_programs=2,  # Low count for test speed
        num_threads=1,
    )
    optimized = opt.compile(
        student=classifier,
        trainset=TRAINSET[:5],
        valset=VALSET[:3],
    )

    result = optimized(ticket="SSO is broken for our whole organization.")
    assert result.category in TICKET_CATEGORIES
    print(f"  Prediction: {result.category}/{result.priority}")

    evaluator = dspy.Evaluate(
        devset=VALSET[:3],
        metric=ticket_metric,
        num_threads=1,
        display_progress=False,
    )
    score = evaluator(optimized)
    print(f"  BootstrapRS score (3 examples): {score}")
    print("  PASSED")


# ============================================================
# MAIN
# ============================================================
def main():
    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print("=" * 60)
    print("Chapter 4: EVERY Code Example Tested")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print()

    tests = [
        test_dataset,
        test_pydantic_model,
        test_signature,
        test_metric,
        test_baseline,
        test_evaluate_baseline,
        test_labeled_few_shot,
        test_bootstrap_few_shot,
        test_mipro_v2,
        test_save_load,
        test_simba,
        test_bootstrap_random_search,
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
        print()

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
