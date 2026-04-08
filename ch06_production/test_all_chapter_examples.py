"""
Tests for Chapter 6: Mostly Harmless (in Production)

Test categories:
- Tests 1-10: Structural tests (always pass, no API calls)
- Tests 11-18: Live API tests (need API key + internet)
- Tests 19-20: FastAPI endpoint tests (need API key + internet)

Run with: python -m pytest test_all_chapter_examples.py -v
"""

import asyncio
import json
import os
import sys
import tempfile
import time

import pytest

# Ensure the current directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Setup: Configure DSPy if API key is available
# ---------------------------------------------------------------------------

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
HAS_API_KEY = bool(API_KEY)

if HAS_API_KEY:
    import dspy
    lm = dspy.LM("anthropic/claude-sonnet-4-6", api_key=API_KEY, max_tokens=1024)
    dspy.configure(lm=lm)


# ===================================================================
# STRUCTURAL TESTS (1-10): No API calls needed
# ===================================================================

def test_01_imports():
    """Test that all module imports work."""
    from content_moderator import (
        ContentModerator,
        ModerateContent,
        ModerationCategory,
        ModerationDecision,
        ModerationLogger,
        BudgetAwareModerator,
        BudgetExceededError,
        FallbackModerator,
        DataFlywheelCallback,
    )
    assert ContentModerator is not None
    assert ModerateContent is not None
    assert ModerationCategory is not None


def test_02_moderation_category_enum():
    """Test the ModerationCategory enum has all expected values."""
    from content_moderator import ModerationCategory

    expected = {"safe", "spam", "toxic", "misinformation", "adult", "violence", "self_harm"}
    actual = {c.value for c in ModerationCategory}
    assert actual == expected


def test_03_moderation_decision_model():
    """Test that ModerationDecision validates correctly."""
    from content_moderator import ModerationDecision

    # Valid decision
    decision = ModerationDecision(
        category="safe",
        confidence=0.95,
        action="approve",
        explanation="Content is a positive product review with no violations.",
    )
    assert decision.category == "safe"
    assert decision.confidence == 0.95
    assert decision.action == "approve"

    # Invalid confidence (out of range)
    with pytest.raises(Exception):
        ModerationDecision(
            category="safe",
            confidence=1.5,  # > 1.0
            action="approve",
            explanation="test",
        )

    # Invalid category
    with pytest.raises(Exception):
        ModerationDecision(
            category="invalid_category",
            confidence=0.5,
            action="approve",
            explanation="test",
        )


def test_04_content_moderator_structure():
    """Test ContentModerator module structure."""
    from content_moderator import ContentModerator

    moderator = ContentModerator()
    assert hasattr(moderator, "moderate")
    assert hasattr(moderator, "forward")


def test_05_budget_moderator_structure():
    """Test BudgetAwareModerator initialization."""
    from content_moderator import BudgetAwareModerator

    moderator = BudgetAwareModerator(budget_limit_tokens=50_000)
    assert moderator.budget_limit == 50_000
    assert moderator.total_tokens_used == 0

    # Test reset
    moderator.total_tokens_used = 30_000
    moderator.reset_budget()
    assert moderator.total_tokens_used == 0


def test_06_budget_exceeded_error():
    """Test that BudgetExceededError is raised when budget is spent."""
    from content_moderator import BudgetAwareModerator, BudgetExceededError

    moderator = BudgetAwareModerator(budget_limit_tokens=100)
    moderator.total_tokens_used = 100  # Simulate exhausted budget

    with pytest.raises(BudgetExceededError, match="Token budget exceeded"):
        moderator(content="test content")


def test_07_moderation_logger_structure():
    """Test ModerationLogger callback structure."""
    from content_moderator import ModerationLogger

    logger = ModerationLogger()
    assert logger.events == []

    # Simulate some events
    logger.on_module_start("call-1", type("FakeModule", (), {"__class__": type("X", (), {"__name__": "FakeModule"})})(), {"content": "test"})
    logger.on_lm_start("call-2", type("FakeLM", (), {"model": "test-model"})(), {})
    logger.on_lm_end("call-2", {"usage": {"prompt_tokens": 100}})
    logger.on_module_end("call-1", {"decision": "safe"})

    summary = logger.get_summary()
    assert summary["total_events"] == 4
    assert summary["lm_calls"] == 1
    assert summary["errors"] == 0


def test_08_data_flywheel_callback():
    """Test DataFlywheelCallback trace collection."""
    from content_moderator import DataFlywheelCallback

    callback = DataFlywheelCallback(output_path="/tmp/test_traces.jsonl")
    assert callback.traces == []

    # Simulate a module call
    callback.on_module_start("call-1", type("FakeModule", (), {"__class__": type("X", (), {"__name__": "TestModule"})})(), {"content": "hello"})
    assert len(callback.traces) == 1
    assert callback.traces[0]["call_id"] == "call-1"

    callback.on_module_end("call-1", {"result": "safe"})
    assert callback.traces[0].get("outputs") is not None


def test_09_data_flywheel_flush():
    """Test DataFlywheelCallback flush to file."""
    from content_moderator import DataFlywheelCallback

    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "traces.jsonl")
    callback = DataFlywheelCallback(output_path=output_path)

    # Add some traces
    callback.on_module_start("call-1", type("M", (), {"__class__": type("X", (), {"__name__": "Mod"})})(), {"content": "test"})
    callback.on_module_end("call-1", {"result": "ok"})

    count = callback.flush()
    assert count == 1
    assert callback.traces == []  # Cleared after flush

    # Verify file contents
    with open(output_path) as f:
        lines = f.readlines()
    assert len(lines) == 1
    trace = json.loads(lines[0])
    assert trace["call_id"] == "call-1"


def test_10_fastapi_app_creation():
    """Test that the FastAPI app can be created."""
    if not HAS_API_KEY:
        pytest.skip("No API key — can't create FastAPI app (needs dspy.configure)")

    from content_moderator import create_app
    app = create_app()

    # Check routes exist
    routes = [route.path for route in app.routes]
    assert "/health" in routes
    assert "/moderate" in routes
    assert "/moderate/stream" in routes
    assert "/moderate/batch" in routes


# ===================================================================
# LIVE API TESTS (11-18): Require API key + internet
# ===================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_11_basic_moderation_safe_content():
    """Test basic moderation with clearly safe content."""
    from content_moderator import ContentModerator

    moderator = ContentModerator()
    result = moderator(content="I made the most amazing chocolate cake today! Here's the recipe.")

    assert hasattr(result, "decision")
    decision = result.decision
    assert decision.category == "safe"
    assert decision.action == "approve"
    assert decision.confidence > 0.5
    assert len(decision.explanation) > 10


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_12_basic_moderation_spam_content():
    """Test moderation catches obvious spam."""
    time.sleep(5)  # Small delay between API calls
    from content_moderator import ContentModerator

    moderator = ContentModerator()
    result = moderator(
        content="BUY CHEAP WATCHES NOW!!! Visit scam-watches.com for 90% OFF!!! "
                "Click here: bit.ly/totallynotascam FREE MONEY!!!",
        context="comment on a cooking blog",
    )

    decision = result.decision
    assert decision.category == "spam"
    assert decision.action in ("flag_for_review", "reject")


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_13_cost_tracking():
    """Test that dspy.track_usage() captures token counts."""
    time.sleep(5)
    from content_moderator import moderate_with_cost_tracking

    result = moderate_with_cost_tracking("Hello, this is a friendly message!")
    assert "decision" in result
    assert "usage_by_model" in result

    usage = result["usage_by_model"]
    # Should have at least one model's usage
    assert len(usage) > 0

    # Check that tokens were tracked
    for model_name, model_usage in usage.items():
        assert "prompt_tokens" in model_usage or "completion_tokens" in model_usage
        total = model_usage.get("prompt_tokens", 0) + model_usage.get("completion_tokens", 0)
        assert total > 0, f"Expected nonzero tokens for {model_name}"


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_14_callback_logging():
    """Test that callbacks capture events during moderation."""
    time.sleep(5)
    from content_moderator import moderate_with_logging

    result = moderate_with_logging("This is a normal product review. The quality was good.")

    assert "decision" in result
    assert "log_summary" in result

    summary = result["log_summary"]
    assert summary["total_events"] > 0
    assert summary["lm_calls"] >= 1  # At least one LLM call
    assert summary["errors"] == 0


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_15_cache_behavior():
    """Test that caching makes the second call faster."""
    time.sleep(5)
    from content_moderator import demonstrate_cache_behavior

    result = demonstrate_cache_behavior()

    assert result["results_match"] is True
    # Cache should be significantly faster (but we allow some variance)
    assert result["second_call_seconds"] <= result["first_call_seconds"]


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_16_per_request_config():
    """Test per-request model configuration with dspy.context()."""
    time.sleep(5)
    from content_moderator import moderate_with_custom_config

    decision = moderate_with_custom_config(
        content="Great weather today! Perfect for a walk.",
        model="anthropic/claude-sonnet-4-6",
    )

    assert decision.category == "safe"
    assert decision.action == "approve"


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_17_streaming():
    """Test that streamify produces chunks and a final prediction."""
    time.sleep(10)  # Longer delay before streaming test
    from content_moderator import demonstrate_streaming

    result = asyncio.run(demonstrate_streaming("What a beautiful sunset! Nature is amazing."))

    assert result["final_prediction"] is not None
    # Streaming should produce at least some chunks
    # (may be 0 if result was cached, which is fine)
    assert isinstance(result["num_chunks"], int)


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_18_save_and_load():
    """Test saving and loading a moderator."""
    time.sleep(5)
    from content_moderator import ContentModerator, save_moderator, load_moderator

    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, "test_moderator.json")

    # Create and save
    moderator = ContentModerator()
    saved = save_moderator(moderator, save_path)
    assert os.path.exists(saved)

    # Load and verify
    loaded = load_moderator(save_path)
    assert isinstance(loaded, ContentModerator)
    assert hasattr(loaded, "moderate")

    # Verify loaded moderator works
    result = loaded(content="Lovely garden flowers blooming this spring!")
    assert result.decision.category == "safe"


# ===================================================================
# FASTAPI TESTS (19-20): Require API key + internet
# ===================================================================

@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_19_fastapi_health_endpoint():
    """Test the /health endpoint."""
    time.sleep(5)
    from fastapi.testclient import TestClient
    from content_moderator import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "claude" in data["model"].lower() or "anthropic" in data["model"].lower()


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_20_fastapi_moderate_endpoint():
    """Test the /moderate endpoint with a real request."""
    time.sleep(10)  # Longer delay before heavy test
    from fastapi.testclient import TestClient
    from content_moderator import create_app

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/moderate",
        json={
            "content": "I just finished reading a great book about gardening!",
            "context": "book review forum",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert "decision" in data
    assert data["decision"]["category"] == "safe"
    assert data["decision"]["action"] == "approve"
    assert data["processing_time_ms"] > 0


# ===================================================================
# Run all tests
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
