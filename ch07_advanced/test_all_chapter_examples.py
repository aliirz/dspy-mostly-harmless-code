"""
Tests for Chapter 7: The Answer Is 42 (Tokens) — Advanced Patterns and Multimodal

Test categories:
- Tests 1-12: Structural tests (always pass, no API calls)
- Tests 13-20: Live API tests (need API key + internet)

Run with: python -m pytest test_all_chapter_examples.py -v
"""

import io
import os
import sys
import tempfile
import time

import pytest

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

    lm = dspy.LM("anthropic/claude-sonnet-4-6", api_key=API_KEY, max_tokens=2048)
    dspy.configure(lm=lm)


# ===================================================================
# STRUCTURAL TESTS (1-12): No API calls needed
# ===================================================================


def test_01_imports():
    """Test that all module imports work."""
    from multimodal_analyzer import (
        ProductImageAnalyzer,
        TextReviewAnalyzer,
        CascadingAnalyzer,
        EnsembleReviewAnalyzer,
        DeepReviewAnalyzer,
        MultimodalProductAnalyzer,
        VisualAnalysis,
        SentimentResult,
        ProductReport,
        DeepAnalysis,
    )
    assert ProductImageAnalyzer is not None
    assert TextReviewAnalyzer is not None
    assert CascadingAnalyzer is not None


def test_02_visual_analysis_model():
    """Test VisualAnalysis Pydantic model validates correctly."""
    from multimodal_analyzer import VisualAnalysis

    analysis = VisualAnalysis(
        product_visible=True,
        image_quality="high",
        matches_description=True,
        visual_notes="Clear product shot with good lighting",
    )
    assert analysis.product_visible is True
    assert analysis.image_quality == "high"

    with pytest.raises(Exception):
        VisualAnalysis(
            product_visible=True,
            image_quality="ultra",  # Not a valid literal
            matches_description=True,
            visual_notes="test",
        )


def test_03_sentiment_result_model():
    """Test SentimentResult Pydantic model validates correctly."""
    from multimodal_analyzer import SentimentResult

    result = SentimentResult(
        sentiment="positive",
        confidence=0.95,
        key_points=["Great battery", "Good build quality"],
        quality_score=4.5,
    )
    assert result.sentiment == "positive"
    assert result.confidence == 0.95
    assert len(result.key_points) == 2

    # Invalid quality_score (out of range)
    with pytest.raises(Exception):
        SentimentResult(
            sentiment="positive",
            confidence=0.9,
            key_points=["test"],
            quality_score=6.0,  # > 5.0
        )


def test_04_product_report_model():
    """Test ProductReport Pydantic model validates correctly."""
    from multimodal_analyzer import ProductReport

    report = ProductReport(
        overall_sentiment="positive",
        quality_score=4.2,
        image_text_agreement="strong_agreement",
        summary="Great product with matching visual evidence.",
        recommendation="recommended",
    )
    assert report.overall_sentiment == "positive"
    assert report.recommendation == "recommended"

    with pytest.raises(Exception):
        ProductReport(
            overall_sentiment="invalid",
            quality_score=4.0,
            image_text_agreement="strong_agreement",
            summary="test",
            recommendation="recommended",
        )


def test_05_deep_analysis_model():
    """Test DeepAnalysis Pydantic model validates correctly."""
    from multimodal_analyzer import DeepAnalysis

    analysis = DeepAnalysis(
        verdict="This is a genuine, positive review of a quality product.",
        confidence=0.88,
        reasoning_summary="Review language is natural, claims are reasonable, no red flags detected.",
    )
    assert analysis.confidence == 0.88
    assert "genuine" in analysis.verdict


def test_06_create_test_image():
    """Test that create_test_image produces a valid dspy.Image."""
    import dspy
    from multimodal_analyzer import create_test_image

    img = create_test_image(50, 50, "green")
    assert isinstance(img, dspy.Image)
    assert hasattr(img, "url")
    # Should be a base64 data URI
    assert img.url.startswith("data:image/")


def test_07_image_from_file():
    """Test creating dspy.Image from a local file."""
    import dspy
    from PIL import Image as PILImage
    from multimodal_analyzer import image_from_file

    # Create a temporary image file
    tmpdir = tempfile.mkdtemp()
    img_path = os.path.join(tmpdir, "test_product.png")
    PILImage.new("RGB", (100, 100), "red").save(img_path)

    dspy_img = image_from_file(img_path)
    assert isinstance(dspy_img, dspy.Image)
    assert dspy_img.url.startswith("data:image/png")


def test_08_module_structures():
    """Test that all modules have correct structure."""
    from multimodal_analyzer import (
        ProductImageAnalyzer,
        TextReviewAnalyzer,
        CascadingAnalyzer,
        EnsembleReviewAnalyzer,
        DeepReviewAnalyzer,
        MultimodalProductAnalyzer,
    )

    # ProductImageAnalyzer
    pia = ProductImageAnalyzer()
    assert hasattr(pia, "analyze")
    assert hasattr(pia, "forward")

    # TextReviewAnalyzer
    tra = TextReviewAnalyzer()
    assert hasattr(tra, "analyze")

    # CascadingAnalyzer
    ca = CascadingAnalyzer(confidence_threshold=0.9)
    assert ca.confidence_threshold == 0.9
    assert ca.cheap_count == 0
    assert ca.expensive_count == 0

    # EnsembleReviewAnalyzer
    era = EnsembleReviewAnalyzer(M=3)
    assert era.M == 3

    # DeepReviewAnalyzer
    dra = DeepReviewAnalyzer()
    assert hasattr(dra, "analyze")

    # MultimodalProductAnalyzer
    mpa = MultimodalProductAnalyzer()
    assert hasattr(mpa, "image_analyzer")
    assert hasattr(mpa, "text_analyzer")
    assert hasattr(mpa, "cross_reference")


def test_09_cascading_stats():
    """Test CascadingAnalyzer routing statistics tracking."""
    from multimodal_analyzer import CascadingAnalyzer

    cascade = CascadingAnalyzer(confidence_threshold=0.85)
    cascade.cheap_count = 7
    cascade.expensive_count = 3

    stats = cascade.get_routing_stats()
    assert stats["total_requests"] == 10
    assert stats["cheap_model_served"] == 7
    assert stats["expensive_model_served"] == 3
    assert stats["escalation_rate"] == 0.3


def test_10_advanced_optimizer_setup():
    """Test that advanced optimizer setup functions work structurally."""
    from multimodal_analyzer import (
        demonstrate_bootstrap_finetune_setup,
        demonstrate_grpo_setup,
        demonstrate_gepa_setup,
    )

    # BootstrapFinetune
    bf = demonstrate_bootstrap_finetune_setup()
    assert bf is not None

    # GRPO
    grpo = demonstrate_grpo_setup()
    assert grpo is not None

    # GEPA
    gepa = demonstrate_gepa_setup()
    assert gepa is not None


def test_11_better_together_setup():
    """Test BetterTogether setup (requires experimental flag)."""
    from multimodal_analyzer import demonstrate_better_together_setup

    bt = demonstrate_better_together_setup()
    assert bt is not None


def test_12_adapter_usage():
    """Test adapter documentation function."""
    from multimodal_analyzer import demonstrate_adapter_usage

    info = demonstrate_adapter_usage()
    assert len(info["available_adapters"]) == 3
    assert info["custom_adapter_base"] == "dspy.adapters.Adapter"
    assert "format" in info["required_methods"][0]
    assert "parse" in info["required_methods"][1]


# ===================================================================
# LIVE API TESTS (13-20): Require API key + internet
# ===================================================================


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_13_text_review_positive():
    """Test text review analysis with clearly positive content."""
    from multimodal_analyzer import TextReviewAnalyzer

    analyzer = TextReviewAnalyzer()
    result = analyzer(
        review_text="Absolutely love this laptop! The display is gorgeous, "
                    "performance is blazing fast, and the battery lasts all day. "
                    "Best purchase I've made this year. Highly recommend!",
        product_category="electronics",
    )

    assert result.result.sentiment == "positive"
    assert result.result.confidence > 0.7
    assert result.result.quality_score >= 4.0
    assert len(result.result.key_points) >= 2


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_14_text_review_negative():
    """Test text review analysis with clearly negative content."""
    time.sleep(5)
    from multimodal_analyzer import TextReviewAnalyzer

    analyzer = TextReviewAnalyzer()
    result = analyzer(
        review_text="Terrible product. Broke after two days. The screen has dead "
                    "pixels, the keyboard is mushy, and customer support was useless. "
                    "Complete waste of money. Do NOT buy this.",
        product_category="electronics",
    )

    assert result.result.sentiment == "negative"
    assert result.result.quality_score <= 2.5
    assert len(result.result.key_points) >= 2


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_15_image_analysis_with_test_image():
    """Test image analysis with a programmatically generated test image."""
    time.sleep(5)
    from multimodal_analyzer import ProductImageAnalyzer, create_test_image

    analyzer = ProductImageAnalyzer()
    test_image = create_test_image(200, 200, "blue")

    result = analyzer(
        product_image=test_image,
        review_text="The blue phone case is exactly as pictured. Great color!",
    )

    assert hasattr(result, "analysis")
    assert hasattr(result.analysis, "visual_notes")
    assert len(result.analysis.visual_notes) > 5


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_16_model_cascading():
    """Test model cascading routes based on confidence."""
    time.sleep(10)
    from multimodal_analyzer import CascadingAnalyzer

    # Use default LM for both (same model, but tests the routing logic)
    cascade = CascadingAnalyzer(confidence_threshold=0.85)

    # Easy case: very clear positive review
    result = cascade(
        review_text="Best headphones ever! 10/10 would buy again!",
        product_category="electronics",
    )

    assert hasattr(result, "result")
    assert result.result.sentiment in ("positive", "negative", "neutral", "mixed")
    # The cascade should have tracked this request
    stats = cascade.get_routing_stats()
    assert stats["total_requests"] == 1


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_17_deep_review_with_reasoning():
    """Test deep analysis that captures reasoning."""
    time.sleep(10)
    from multimodal_analyzer import DeepReviewAnalyzer

    analyzer = DeepReviewAnalyzer()
    result = analyzer(
        review_text="I've been using this standing desk for three months now. "
                    "The motor is quiet, height adjustment is smooth, and the "
                    "build quality feels premium. However, the cable management "
                    "tray is flimsy and the assembly instructions were confusing.",
        product_category="furniture",
    )

    assert hasattr(result, "reasoning")
    assert hasattr(result, "analysis")
    # Reasoning is str-like
    assert len(str(result.reasoning)) > 0
    assert result.analysis.confidence > 0
    assert len(result.analysis.verdict) > 10


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_18_multimodal_full_pipeline():
    """Test the full multimodal pipeline with image + text."""
    time.sleep(15)
    from multimodal_analyzer import MultimodalProductAnalyzer, create_test_image

    analyzer = MultimodalProductAnalyzer()
    test_image = create_test_image(200, 200, "silver")

    result = analyzer(
        product_image=test_image,
        review_text="This silver watch looks elegant in person. The craftsmanship "
                    "is excellent and it keeps perfect time. Worth every penny.",
        product_category="accessories",
    )

    assert hasattr(result, "image_analysis")
    assert hasattr(result, "text_analysis")
    assert hasattr(result, "report")
    assert result.report.overall_sentiment in ("positive", "negative", "neutral", "mixed")
    assert 1.0 <= result.report.quality_score <= 5.0
    assert result.report.recommendation in ("recommended", "not_recommended", "needs_more_info")


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_19_save_and_load():
    """Test saving and loading a pipeline."""
    time.sleep(5)
    from multimodal_analyzer import MultimodalProductAnalyzer, save_pipeline, load_pipeline

    tmpdir = tempfile.mkdtemp()
    original_dir = os.getcwd()

    try:
        os.chdir(tmpdir)
        pipeline = MultimodalProductAnalyzer()
        saved_path = save_pipeline(pipeline, version="test_v1")
        assert os.path.exists(saved_path)

        loaded = load_pipeline(version="test_v1")
        assert isinstance(loaded, MultimodalProductAnalyzer)
        assert hasattr(loaded, "image_analyzer")
        assert hasattr(loaded, "text_analyzer")
        assert hasattr(loaded, "cross_reference")
    finally:
        os.chdir(original_dir)


@pytest.mark.skipif(not HAS_API_KEY, reason="No API key available")
def test_20_ensemble_review():
    """Test MultiChainComparison ensemble analysis."""
    time.sleep(15)
    from multimodal_analyzer import EnsembleReviewAnalyzer

    analyzer = EnsembleReviewAnalyzer(M=3)
    result = analyzer(
        review_text="Decent coffee maker. Makes good coffee but the carafe drips "
                    "when pouring. Not bad for the price but not great either.",
        product_category="kitchen",
    )

    assert hasattr(result, "sentiment")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "rationale")
    # Rationale should contain the corrected reasoning
    assert len(str(result.rationale)) > 10


# ===================================================================
# Run all tests
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
