"""
test_startup_roaster.py - Chapter 1: Don't Panic
DSPy: The Mostly Harmless Guide

Tests that verify our code samples are correct.
Run with: poetry run python test_startup_roaster.py

These tests validate:
1. Signature definitions are correct (fields, types, descriptions)
2. Module structure works (composition, forward method)
3. DSPy API patterns work as documented
4. LM calls produce structured output (requires API key)
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

import dspy

# Import our chapter code
from startup_roaster import RoastStartup, StartupRoaster


def test_signature_structure():
    """Verify the RoastStartup signature has correct fields."""
    print("TEST: Signature structure...")

    # Check input fields
    input_fields = RoastStartup.input_fields
    assert "pitch" in input_fields, "Missing input field: pitch"

    # Check output fields
    output_fields = RoastStartup.output_fields
    expected_outputs = ["roast", "viability_score", "strengths", "weaknesses", "verdict"]
    for field in expected_outputs:
        assert field in output_fields, f"Missing output field: {field}"

    # Check the docstring/instructions are set
    assert len(RoastStartup.__doc__) > 0, "Signature should have instructions (docstring)"

    print("  PASSED: All fields present and correct")


def test_module_structure():
    """Verify the StartupRoaster module is properly constructed."""
    print("TEST: Module structure...")

    roaster = StartupRoaster()

    # Check it has predictors
    predictors = roaster.predictors()
    assert len(predictors) == 1, f"Expected 1 predictor, got {len(predictors)}"

    # Check named predictors
    named = roaster.named_predictors()
    assert len(named) == 1, f"Expected 1 named predictor, got {len(named)}"
    name, predictor = named[0]
    assert name == "analyze", f"Expected predictor named 'analyze', got '{name}'"

    print("  PASSED: Module structure correct")


def test_inline_signature():
    """Verify inline signature syntax works as documented."""
    print("TEST: Inline signature syntax...")

    sig = dspy.Signature("question -> answer")
    assert "question" in sig.input_fields
    assert "answer" in sig.output_fields

    sig2 = dspy.Signature("question: str, context: str -> answer: str")
    assert "question" in sig2.input_fields
    assert "context" in sig2.input_fields
    assert "answer" in sig2.output_fields

    print("  PASSED: Inline signatures work")


def test_chain_of_thought():
    """Verify ChainOfThought adds reasoning field."""
    print("TEST: ChainOfThought structure...")

    class SimpleTask(dspy.Signature):
        """A simple task."""
        question: str = dspy.InputField(desc="A question")
        answer: str = dspy.OutputField(desc="The answer")

    cot = dspy.ChainOfThought(SimpleTask)

    # ChainOfThought should have the extended signature with reasoning
    extended_sig = cot.predict.signature
    assert "reasoning" in extended_sig.output_fields, (
        "ChainOfThought should add a 'reasoning' output field"
    )

    print("  PASSED: ChainOfThought adds reasoning field")


def test_example_creation():
    """Verify dspy.Example works as documented."""
    print("TEST: Example creation...")

    ex = dspy.Example(question="What is 2+2?", answer="4")
    assert ex.question == "What is 2+2?"
    assert ex["answer"] == "4"

    # Test with_inputs
    labeled = ex.with_inputs("question")
    inputs = labeled.inputs()
    assert "question" in inputs.keys()
    labels = labeled.labels()
    assert "answer" in labels.keys()

    print("  PASSED: Example creation and labeling works")


def test_lm_configuration():
    """Verify LM can be created and configured."""
    print("TEST: LM configuration...")

    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=1000,
    )
    assert lm.model == "anthropic/claude-sonnet-4-6"

    # Test configure
    dspy.configure(lm=lm)

    # Test context manager for temporary overrides
    alt_lm = dspy.LM("openai/gpt-5.4-mini", temperature=0.5)
    with dspy.context(lm=alt_lm):
        # Inside context, alt_lm is active
        pass
    # Outside context, original lm is active again

    print("  PASSED: LM configuration works")


def test_live_api_call():
    """Test actual API call if ANTHROPIC_API_KEY is set.

    This test is skipped if no API key is available.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("TEST: Live API call... SKIPPED (no ANTHROPIC_API_KEY)")
        return

    print("TEST: Live API call...")

    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)

    roaster = StartupRoaster()
    result = roaster(
        pitch="A social network exclusively for houseplants. "
              "Each plant gets a profile managed by its owner. "
              "Plants can 'follow' each other. $3/month per plant."
    )

    # Verify all output fields are populated
    assert result.roast is not None, "roast should not be None"
    assert isinstance(result.viability_score, int), (
        f"viability_score should be int, got {type(result.viability_score)}"
    )
    assert 1 <= result.viability_score <= 10, (
        f"viability_score should be 1-10, got {result.viability_score}"
    )
    assert isinstance(result.strengths, list), (
        f"strengths should be a list, got {type(result.strengths)}"
    )
    assert isinstance(result.weaknesses, list), (
        f"weaknesses should be a list, got {type(result.weaknesses)}"
    )
    assert result.verdict is not None, "verdict should not be None"

    print(f"  Roast: {result.roast[:80]}...")
    print(f"  Score: {result.viability_score}/10")
    print(f"  PASSED: Live API call returned structured output")

    # Also test inspect_history works
    print("\n  Inspecting last LM call:")
    dspy.inspect_history(n=1)


def main():
    print("=" * 60)
    print("Chapter 1 Tests: Don't Panic")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print()

    tests = [
        test_signature_structure,
        test_module_structure,
        test_inline_signature,
        test_chain_of_thought,
        test_example_creation,
        test_lm_configuration,
        test_live_api_call,
    ]

    passed = 0
    failed = 0
    skipped = 0

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

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
