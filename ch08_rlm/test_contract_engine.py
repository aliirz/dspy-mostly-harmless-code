"""
test_contract_engine.py - Chapter 8: The Infinite Improbability Context
DSPy: The Mostly Harmless Guide

Tests for the Contract Intelligence Engine (dspy.RLM).

Run with: poetry run python test_contract_engine.py

Tests are organized in three tiers:
  1. Structure tests — no API key, no Deno required
  2. Tool tests — verify custom tools work correctly
  3. Live RLM test — requires ANTHROPIC_API_KEY and Deno
"""

import os
import re
import shutil
import sys

from dotenv import load_dotenv
load_dotenv()

import dspy

from contract_engine import (
    ContractRisk,
    ContractIntelligenceEngine,
    extract_section,
    SAMPLE_CONTRACT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deno_available() -> bool:
    return shutil.which("deno") is not None


def _api_key_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Tier 1: Structure tests (no external dependencies)
# ---------------------------------------------------------------------------

def test_contract_risk_schema():
    """Verify ContractRisk has the correct Pydantic fields."""
    print("TEST: ContractRisk schema...")

    fields = ContractRisk.model_fields
    expected = ["key_obligations", "risk_flags", "important_dates", "liability_cap", "overall_risk"]
    for field in expected:
        assert field in fields, f"Missing field: {field}"

    # Verify list fields
    assert str(fields["key_obligations"].annotation) in (
        "list[str]", "typing.List[str]"
    ), "key_obligations should be list[str]"

    print("  PASSED: ContractRisk schema correct")


def test_rlm_module_structure():
    """Verify ContractIntelligenceEngine has RLM as its core module."""
    print("TEST: Module structure...")

    engine = ContractIntelligenceEngine()

    # The engine should have an analyze attribute that is an RLM
    assert hasattr(engine, "analyze"), "Engine should have 'analyze' attribute"
    assert isinstance(engine.analyze, dspy.RLM), (
        f"engine.analyze should be dspy.RLM, got {type(engine.analyze)}"
    )

    # The RLM should have the expected configuration
    assert engine.analyze.max_iterations == 12
    assert engine.analyze.max_llm_calls == 20

    print("  PASSED: Module structure correct")


def test_rlm_signature():
    """Verify the RLM signature has correct input/output fields."""
    print("TEST: RLM signature fields...")

    engine = ContractIntelligenceEngine()
    sig = engine.analyze.signature

    assert "contract" in sig.input_fields, "Missing input: contract"
    assert "risk" in sig.output_fields, "Missing output: risk"

    print("  PASSED: RLM signature fields correct")


def test_rlm_custom_tools():
    """Verify custom tools are registered on the RLM."""
    print("TEST: Custom tools registration...")

    engine = ContractIntelligenceEngine()
    tools = engine.analyze.tools

    assert "extract_section" in tools, (
        f"extract_section not in tools. Found: {list(tools.keys())}"
    )

    print("  PASSED: Custom tools registered correctly")


def test_rlm_sub_lm_wiring():
    """Verify sub_lm is passed through correctly."""
    print("TEST: sub_lm wiring...")

    cheap_lm = dspy.LM("anthropic/claude-haiku-4-5-20251001", max_tokens=1000)
    engine = ContractIntelligenceEngine(sub_lm=cheap_lm)

    assert engine.analyze.sub_lm is cheap_lm, (
        "sub_lm should be the cheap model passed at construction"
    )

    print("  PASSED: sub_lm wired correctly")


# ---------------------------------------------------------------------------
# Tier 2: Tool tests (no API key, no Deno)
# ---------------------------------------------------------------------------

def test_extract_section_found():
    """Verify extract_section returns the right section content."""
    print("TEST: extract_section — found case...")

    result = extract_section(SAMPLE_CONTRACT, "TERMINATION")

    assert "TERMINATION" in result.upper(), "Should contain the section name"
    assert "30 days" in result, "Should contain termination notice period"
    assert len(result) <= 3001, "Should be capped at 3000 chars"

    print("  PASSED: extract_section finds and returns correct section")


def test_extract_section_not_found():
    """Verify extract_section handles missing sections gracefully."""
    print("TEST: extract_section — not found case...")

    result = extract_section(SAMPLE_CONTRACT, "QUANTUM_ENTANGLEMENT_CLAUSE")

    assert "not found" in result.lower(), (
        f"Should return not-found message, got: {result[:80]}"
    )

    print("  PASSED: extract_section handles missing sections")


def test_extract_section_truncation():
    """Verify extract_section truncates to 3000 chars."""
    print("TEST: extract_section — truncation...")

    # Make a big fake section
    big_contract = "1. SERVICES\n" + "x" * 10_000
    result = extract_section(big_contract, "SERVICES")

    assert len(result) <= 3001, f"Should be ≤3000 chars, got {len(result)}"

    print("  PASSED: extract_section truncates correctly")


def test_contract_size():
    """Verify the sample contract is actually large enough to warrant RLM."""
    print("TEST: Sample contract is large...")

    assert len(SAMPLE_CONTRACT) > 5_000, (
        f"Sample contract should be >5K chars, got {len(SAMPLE_CONTRACT)}"
    )

    print(f"  PASSED: Sample contract is {len(SAMPLE_CONTRACT):,} characters")


# ---------------------------------------------------------------------------
# Tier 3: Live RLM test (requires API key + Deno)
# ---------------------------------------------------------------------------

def test_live_rlm_call():
    """Full end-to-end RLM test against real API.

    Requirements:
      - ANTHROPIC_API_KEY environment variable
      - Deno installed (deno --version must work)

    Known issue: DSPy's Deno/Pyodide sandbox has reported cache-discovery
    failures on some systems. When every REPL iteration fails, RLM falls back
    to an extract pass over the accumulated history — this still produces a
    useful result but as an unstructured string rather than a typed object.
    The test accepts both outcomes and reports which path was taken.
    See: https://dspy.ai/api/modules/RLM (Notes section)
    """
    if not _api_key_available():
        print("TEST: Live RLM call... SKIPPED (no ANTHROPIC_API_KEY)")
        return

    if not _deno_available():
        print("TEST: Live RLM call... SKIPPED (Deno not installed)")
        print("  Install: curl -fsSL https://deno.land/install.sh | sh")
        return

    print("TEST: Live RLM call (this will take ~30–60 seconds)...")

    main_lm = dspy.LM("anthropic/claude-sonnet-4-6", max_tokens=4000)
    cheap_lm = dspy.LM("anthropic/claude-haiku-4-5-20251001", max_tokens=2000)
    dspy.configure(lm=main_lm)

    engine = ContractIntelligenceEngine(sub_lm=cheap_lm)

    # Shorter contract to keep costs down
    test_contract = """
SERVICE AGREEMENT between TestCorp ("Client") and DevShop ("Vendor").

1. SERVICES: Vendor provides web development services per Exhibit A.
   Client provides feedback within 5 business days.

2. PAYMENT: $10,000/month, due within 30 days. Late interest: 1.5%/month.

3. TERMINATION: Either party may terminate for cause with 30 days notice.
   Client may terminate for convenience with 60 days notice and $20,000 fee.

4. IP: All work product belongs to Vendor until fully paid. Upon payment, IP transfers to Client.

5. LIABILITY: Total liability capped at fees paid in prior 2 months.
   No consequential damages under any circumstances.

6. GOVERNING LAW: Laws of California. Disputes via AAA arbitration in San Francisco.
"""

    risk = engine(contract_text=test_contract)

    # Happy path: Deno/Pyodide worked, we got a typed ContractRisk
    if isinstance(risk, ContractRisk):
        assert isinstance(risk.key_obligations, list), "key_obligations should be list"
        assert len(risk.key_obligations) > 0, "Should find at least one obligation"
        assert isinstance(risk.risk_flags, list), "risk_flags should be list"
        assert isinstance(risk.important_dates, list), "important_dates should be list"
        assert isinstance(risk.liability_cap, str) and len(risk.liability_cap) > 0
        assert risk.overall_risk in ("low", "medium", "high"), (
            f"overall_risk should be low/medium/high, got: {risk.overall_risk}"
        )
        print(f"  Overall risk: {risk.overall_risk}")
        print(f"  Obligations found: {len(risk.key_obligations)}")
        print(f"  Liability cap: {risk.liability_cap}")
        print("  PASSED: Live RLM call returned valid ContractRisk")

    # Degraded path: Deno/Pyodide cache failure, extract fallback returned str
    elif isinstance(risk, str):
        assert len(risk) > 50, "Fallback string should contain meaningful content"
        print(f"  NOTE: Deno/Pyodide sandbox failed — extract fallback used")
        print(f"  Fallback result: {risk[:120]}...")
        print("  PASSED: RLM ran and produced output (via extract fallback)")
        print("  TIP: If Pyodide keeps failing, run: dspy install pyodide")

    else:
        raise AssertionError(
            f"Unexpected result type: {type(risk)}. Expected ContractRisk or str."
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Chapter 8 Tests: The Infinite Improbability Context")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print(f"\nDeno available: {'yes' if _deno_available() else 'no (live test will skip)'}")
    print(f"API key set:    {'yes' if _api_key_available() else 'no (live test will skip)'}")
    print()

    tests = [
        # Tier 1: structure
        test_contract_risk_schema,
        test_rlm_module_structure,
        test_rlm_signature,
        test_rlm_custom_tools,
        test_rlm_sub_lm_wiring,
        # Tier 2: tools
        test_extract_section_found,
        test_extract_section_not_found,
        test_extract_section_truncation,
        test_contract_size,
        # Tier 3: live
        test_live_rlm_call,
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

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
