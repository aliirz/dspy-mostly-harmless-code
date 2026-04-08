"""
Chapter 7: The Answer Is 42 (Tokens) — Advanced Patterns and Multimodal

A Multimodal Product Review Analyzer that demonstrates:
- dspy.Image for vision-capable analysis
- dspy.Reasoning for extended thinking
- Model cascading (cheap → expensive routing)
- dspy.MultiChainComparison for ensemble reasoning
- dspy.Parallel for concurrent module execution
- Advanced optimizers: BootstrapFinetune, GRPO, BetterTogether, GEPA
- Custom adapters
- Save/Load for versioned deployment
"""

import io
import os
import time
from typing import Literal, Optional

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Part 1: Multimodal Inputs — dspy.Image
# ---------------------------------------------------------------------------
# DSPy treats images as first-class citizens. The Image type handles
# URLs, local files, PIL images, and raw bytes — all normalized to
# base64 data URIs behind the scenes.


def create_test_image(width: int = 100, height: int = 100, color: str = "red") -> "dspy.Image":
    """Create a simple test image using Pillow and wrap it in dspy.Image.

    This is how you'd pass any PIL image to DSPy — from a camera feed,
    a screenshot, an image processing pipeline, etc.
    """
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (width, height), color=color)
    return dspy.Image(img)


def image_from_url(url: str) -> "dspy.Image":
    """Create a dspy.Image from a URL.

    By default, DSPy passes the URL directly to the vision model.
    Set download=True to fetch the image and encode it as base64
    (useful when the URL might expire or require auth).
    """
    return dspy.Image(url)


def image_from_file(path: str) -> "dspy.Image":
    """Create a dspy.Image from a local file path.

    Supports PNG, JPEG, GIF, WebP, and other common formats.
    The image is automatically base64-encoded.
    """
    return dspy.Image(path)


# ---------------------------------------------------------------------------
# Part 2: The Visual Product Analyzer
# ---------------------------------------------------------------------------
# Our first signature uses dspy.Image as an input field. The LLM sees
# the image and analyzes it alongside the text review.


class VisualAnalysis(BaseModel):
    """What we can learn from looking at the product image."""
    product_visible: bool = Field(description="Whether the actual product is visible in the image")
    image_quality: Literal["high", "medium", "low"] = Field(description="Quality of the product photo")
    matches_description: bool = Field(description="Whether the image matches the text review")
    visual_notes: str = Field(description="Key observations from the image")


class AnalyzeProductImage(dspy.Signature):
    """Analyze a product image alongside its text review. Look for whether
    the image shows the actual product, assess image quality, and note
    whether visual evidence supports or contradicts the review text."""

    product_image: dspy.Image = dspy.InputField(desc="Photo of the product being reviewed")
    review_text: str = dspy.InputField(desc="The text of the product review")
    analysis: VisualAnalysis = dspy.OutputField(desc="Visual analysis of the product image")


class ProductImageAnalyzer(dspy.Module):
    """Analyzes product images with their accompanying reviews.

    This module demonstrates dspy.Image as a first-class input type.
    The vision-capable LLM receives both the image and text together.
    """

    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzeProductImage)

    def forward(self, product_image: dspy.Image, review_text: str) -> dspy.Prediction:
        result = self.analyze(product_image=product_image, review_text=review_text)
        return dspy.Prediction(analysis=result.analysis)


# ---------------------------------------------------------------------------
# Part 3: Text Review Analysis (for the cascade)
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    """Structured sentiment analysis output."""
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    key_points: list[str] = Field(description="Main points from the review (2-5 items)")
    quality_score: float = Field(ge=1.0, le=5.0, description="Overall product quality rating on 1-5 scale")


class AnalyzeReviewText(dspy.Signature):
    """Analyze a product review to extract sentiment, key points, and an
    overall quality score. Be calibrated: 5-star language → high score,
    mixed feelings → middle score, complaints → low score."""

    review_text: str = dspy.InputField(desc="The product review text to analyze")
    product_category: str = dspy.InputField(desc="Category of the product (e.g., 'electronics', 'clothing')")
    result: SentimentResult = dspy.OutputField(desc="Structured sentiment analysis")


class TextReviewAnalyzer(dspy.Module):
    """Analyzes the text portion of a product review."""

    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzeReviewText)

    def forward(self, review_text: str, product_category: str = "general") -> dspy.Prediction:
        result = self.analyze(review_text=review_text, product_category=product_category)
        return dspy.Prediction(result=result.result)


# ---------------------------------------------------------------------------
# Part 4: Model Cascading — Cheap Model First, Expensive When Needed
# ---------------------------------------------------------------------------
# The idea: use a fast, cheap model for easy cases. Route hard cases
# (low confidence, mixed sentiment) to a more capable (expensive) model.
# This can cut costs 50-70% with negligible quality loss.


class CascadingAnalyzer(dspy.Module):
    """A model cascade that routes reviews based on difficulty.

    Step 1: Cheap model does a quick classification.
    Step 2: If the cheap model is confident (> threshold), return its result.
    Step 3: If uncertain, re-analyze with the expensive model.

    This pattern is the single biggest cost optimization for production
    LLM systems. Most content is easy — don't waste your best model on it.
    """

    def __init__(
        self,
        cheap_lm: dspy.LM | None = None,
        expensive_lm: dspy.LM | None = None,
        confidence_threshold: float = 0.85,
    ):
        self.cheap_analyzer = dspy.ChainOfThought(AnalyzeReviewText)
        self.expensive_analyzer = dspy.ChainOfThought(AnalyzeReviewText)
        self.cheap_lm = cheap_lm
        self.expensive_lm = expensive_lm
        self.confidence_threshold = confidence_threshold

        # Tracking for monitoring
        self.cheap_count = 0
        self.expensive_count = 0

    def forward(self, review_text: str, product_category: str = "general") -> dspy.Prediction:
        # Step 1: Try the cheap model
        if self.cheap_lm:
            with dspy.context(lm=self.cheap_lm):
                cheap_result = self.cheap_analyzer(
                    review_text=review_text, product_category=product_category
                )
        else:
            cheap_result = self.cheap_analyzer(
                review_text=review_text, product_category=product_category
            )

        # Step 2: Check confidence
        if cheap_result.result.confidence >= self.confidence_threshold:
            self.cheap_count += 1
            return dspy.Prediction(
                result=cheap_result.result,
                model_tier="cheap",
                escalated=False,
            )

        # Step 3: Escalate to expensive model
        if self.expensive_lm:
            with dspy.context(lm=self.expensive_lm):
                expensive_result = self.expensive_analyzer(
                    review_text=review_text, product_category=product_category
                )
        else:
            expensive_result = self.expensive_analyzer(
                review_text=review_text, product_category=product_category
            )

        self.expensive_count += 1
        return dspy.Prediction(
            result=expensive_result.result,
            model_tier="expensive",
            escalated=True,
        )

    def get_routing_stats(self):
        total = self.cheap_count + self.expensive_count
        return {
            "total_requests": total,
            "cheap_model_served": self.cheap_count,
            "expensive_model_served": self.expensive_count,
            "escalation_rate": round(self.expensive_count / max(total, 1), 3),
        }


# ---------------------------------------------------------------------------
# Part 5: MultiChainComparison — Ensemble Reasoning
# ---------------------------------------------------------------------------
# Generate M independent reasoning attempts, then let the LLM compare
# them all and pick (or synthesize) the best answer. This is ensemble
# reasoning for LLMs — the language model equivalent of random forests.


class EnsembleReviewAnalyzer(dspy.Module):
    """Uses MultiChainComparison to ensemble M independent analyses.

    The workflow:
    1. Generate M independent review analyses (at temperature > 0)
    2. Feed all M attempts to MultiChainComparison
    3. MCC compares them and produces a corrected, synthesized result

    This consistently outperforms single-pass analysis on nuanced reviews
    where sentiment is mixed or context-dependent.
    """

    def __init__(self, M: int = 3):
        self.M = M
        self.generate = dspy.ChainOfThought(
            "review_text, product_category -> sentiment, quality_score"
        )
        self.compare = dspy.MultiChainComparison(
            "review_text, product_category -> sentiment, quality_score",
            M=M,
            temperature=0.7,
        )

    def forward(self, review_text: str, product_category: str = "general") -> dspy.Prediction:
        # Step 1: Generate M independent attempts with temperature
        completions = []
        for _ in range(self.M):
            attempt = self.generate(
                review_text=review_text,
                product_category=product_category,
                config={"temperature": 0.7},
            )
            completions.append({
                "rationale": attempt.rationale if hasattr(attempt, "rationale") else "",
                "quality_score": str(attempt.quality_score),
            })

        # Step 2: Compare all attempts
        result = self.compare(
            completions,
            review_text=review_text,
            product_category=product_category,
        )

        return dspy.Prediction(
            sentiment=result.sentiment,
            quality_score=result.quality_score,
            rationale=result.rationale,
        )


# ---------------------------------------------------------------------------
# Part 6: dspy.Reasoning — Extended Thinking
# ---------------------------------------------------------------------------
# For models that support extended thinking (OpenAI o1, o3, etc.),
# dspy.Reasoning captures the model's internal reasoning process.
# It's str-like, so you can use it just like a string in your code.


class DeepAnalysis(BaseModel):
    """A thorough analysis that benefits from extended thinking."""
    verdict: str = Field(description="Final verdict on the product")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(description="Summary of the reasoning process")


class DeepProductAnalysis(dspy.Signature):
    """Perform a deep, thorough analysis of this product review. Consider
    potential biases, verify claims against common knowledge, and provide
    a well-reasoned verdict."""

    review_text: str = dspy.InputField(desc="The product review to deeply analyze")
    product_category: str = dspy.InputField(desc="Product category for context")
    reasoning: dspy.Reasoning = dspy.OutputField(desc="Your extended reasoning process")
    analysis: DeepAnalysis = dspy.OutputField(desc="The final deep analysis")


class DeepReviewAnalyzer(dspy.Module):
    """Uses dspy.Reasoning to capture extended thinking from the LLM.

    When used with reasoning-capable models (o1, o3), DSPy automatically
    enables native reasoning. With other models, the Reasoning field
    acts like an explicit chain-of-thought that's returned alongside
    the structured output.

    The Reasoning object is str-like — you can call .strip(), .split(),
    len(), iterate over it, etc.
    """

    def __init__(self):
        self.analyze = dspy.Predict(DeepProductAnalysis)

    def forward(self, review_text: str, product_category: str = "general") -> dspy.Prediction:
        result = self.analyze(review_text=review_text, product_category=product_category)
        return dspy.Prediction(
            reasoning=result.reasoning,
            analysis=result.analysis,
        )


# ---------------------------------------------------------------------------
# Part 7: dspy.Parallel — Concurrent Module Execution
# ---------------------------------------------------------------------------
# When you have multiple independent analyses to run, Parallel handles
# the threading, error collection, and progress tracking.


def analyze_reviews_parallel(reviews: list[dict], num_threads: int = 4):
    """Run text analysis on multiple reviews concurrently using dspy.Parallel.

    dspy.Parallel takes a list of (module, input) pairs and runs them
    in parallel with automatic error handling and straggler detection.
    """
    analyzer = TextReviewAnalyzer()

    # Build (module, example) pairs
    exec_pairs = [
        (analyzer, dspy.Example(
            review_text=r["text"],
            product_category=r.get("category", "general"),
        ).with_inputs("review_text", "product_category"))
        for r in reviews
    ]

    parallel = dspy.Parallel(
        num_threads=num_threads,
        return_failed_examples=True,
        provide_traceback=True,
    )

    results, failed, errors = parallel(exec_pairs)

    return {
        "succeeded": len(results),
        "failed": len(failed),
        "results": results,
        "errors": [str(e) for e in errors],
    }


# ---------------------------------------------------------------------------
# Part 8: The Full Multimodal Pipeline
# ---------------------------------------------------------------------------
# Combines image analysis + text analysis + cross-referencing into
# a single pipeline. This is the capstone module.


class ProductReport(BaseModel):
    """The final cross-referenced product review report."""
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    quality_score: float = Field(ge=1.0, le=5.0)
    image_text_agreement: Literal["strong_agreement", "partial_agreement", "disagreement"]
    summary: str = Field(description="2-3 sentence summary of findings")
    recommendation: Literal["recommended", "not_recommended", "needs_more_info"]


class CrossReference(dspy.Signature):
    """Cross-reference visual analysis of a product image with the text
    review sentiment analysis. Determine whether the image supports the
    review text, and produce a final product quality report."""

    visual_notes: str = dspy.InputField(desc="Key observations from the product image")
    image_quality: str = dspy.InputField(desc="Quality assessment of the product photo")
    text_sentiment: str = dspy.InputField(desc="Sentiment from the text review")
    text_key_points: str = dspy.InputField(desc="Key points from the text review")
    text_quality_score: str = dspy.InputField(desc="Quality score from text analysis")
    report: ProductReport = dspy.OutputField(desc="Final cross-referenced product report")


class MultimodalProductAnalyzer(dspy.Module):
    """The full multimodal pipeline: image + text → cross-referenced report.

    Workflow:
    1. Analyze the product image (visual quality, product visibility)
    2. Analyze the review text (sentiment, key points, quality score)
    3. Cross-reference both analyses into a final report
    """

    def __init__(self):
        self.image_analyzer = ProductImageAnalyzer()
        self.text_analyzer = TextReviewAnalyzer()
        self.cross_reference = dspy.ChainOfThought(CrossReference)

    def forward(
        self,
        product_image: dspy.Image,
        review_text: str,
        product_category: str = "general",
    ) -> dspy.Prediction:
        # Step 1: Analyze image
        image_result = self.image_analyzer(
            product_image=product_image,
            review_text=review_text,
        )

        # Step 2: Analyze text
        text_result = self.text_analyzer(
            review_text=review_text,
            product_category=product_category,
        )

        # Step 3: Cross-reference
        cross_ref = self.cross_reference(
            visual_notes=image_result.analysis.visual_notes,
            image_quality=image_result.analysis.image_quality,
            text_sentiment=text_result.result.sentiment,
            text_key_points=", ".join(text_result.result.key_points),
            text_quality_score=str(text_result.result.quality_score),
        )

        return dspy.Prediction(
            image_analysis=image_result.analysis,
            text_analysis=text_result.result,
            report=cross_ref.report,
        )


# ---------------------------------------------------------------------------
# Part 9: Advanced Optimizers Reference
# ---------------------------------------------------------------------------
# These optimizers require fine-tuning infrastructure (API support for
# training jobs). We document them here with complete, correct signatures
# so you know exactly how to use them when you're ready.


def demonstrate_bootstrap_finetune_setup():
    """Show how to set up BootstrapFinetune (requires fine-tuning API support).

    BootstrapFinetune works by:
    1. Running a teacher model to generate high-quality traces
    2. Filtering traces by your metric
    3. Launching fine-tuning jobs on your target model
    4. Returning the student with fine-tuned LMs swapped in
    """
    from dspy.teleprompt import BootstrapFinetune

    # This is the setup — actual execution requires fine-tuning API support
    optimizer = BootstrapFinetune(
        metric=lambda example, prediction, trace=None: prediction.result.confidence > 0.8,
        multitask=True,       # One fine-tuning job for all LMs
        exclude_demos=False,  # Include few-shot demos in training data
        num_threads=4,
    )

    # To compile (requires fine-tuning-capable LM):
    # student = TextReviewAnalyzer()
    # optimized = optimizer.compile(student, trainset=training_examples)
    return optimizer


def demonstrate_grpo_setup():
    """Show how to set up GRPO (Group Reward Policy Optimization).

    GRPO uses reinforcement learning to optimize LM weights directly,
    without needing labeled output examples — just a reward function.
    """
    from dspy.teleprompt.grpo import GRPO

    optimizer = GRPO(
        metric=lambda example, prediction, trace=None: prediction.result.confidence > 0.8,
        multitask=True,
        exclude_demos=True,       # Required: must be True for GRPO
        num_threads=6,
        num_train_steps=100,      # Training budget
        num_rollouts_per_grpo_step=1,
        failure_score=0,          # Score for metric failures
        format_failure_score=-1,  # Score for parse failures (must be < failure_score)
    )

    # To compile (requires RL-capable LM):
    # student = TextReviewAnalyzer()
    # optimized = optimizer.compile(student, trainset=training_examples)
    return optimizer


def demonstrate_better_together_setup():
    """Show how to set up BetterTogether (prompt + weight optimization).

    BetterTogether alternates between prompt optimization and fine-tuning:
    - "p" phase: optimizes instructions and few-shot demos
    - "w" phase: fine-tunes model weights

    The strategy string controls the sequence: "p -> w -> p" means
    optimize prompts, then fine-tune, then optimize prompts again.
    """
    import dspy

    # BetterTogether requires experimental mode
    dspy.settings.experimental = True

    from dspy.teleprompt import BetterTogether

    optimizer = BetterTogether(
        metric=lambda example, prediction, trace=None: prediction.result.confidence > 0.8,
        seed=42,
        # Uses BootstrapFewShotWithRandomSearch for prompts
        # Uses BootstrapFinetune for weights
    )

    # To compile:
    # student = TextReviewAnalyzer()
    # optimized = optimizer.compile(
    #     student,
    #     trainset=training_examples,
    #     strategy="p -> w -> p",   # Prompt, then weight, then prompt again
    #     valset_ratio=0.1,
    # )

    # Reset experimental mode
    dspy.settings.experimental = False
    return optimizer


def demonstrate_gepa_setup():
    """Show how to set up GEPA (evolutionary prompt optimization).

    GEPA uses an evolutionary approach with reflection to evolve
    instructions and few-shot demos. It's the most sophisticated
    prompt-only optimizer — no fine-tuning needed.

    Auto budget presets:
    - "light": 6 full evaluations
    - "medium": 12 full evaluations
    - "heavy": 18 full evaluations
    """
    import dspy

    dspy.settings.experimental = True

    from dspy.teleprompt import GEPA

    # GEPA needs a special metric that can optionally return feedback
    def review_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        """GEPA metric with optional textual feedback."""
        score = 1.0 if prediction.result.confidence > 0.8 else 0.0
        return score

    # GEPA requires a reflection LM — a strong model used to analyze failures
    # and propose better instructions. This is typically your best available model.
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    reflection_lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        api_key=api_key,
        max_tokens=4096,
        temperature=1.0,
    )

    optimizer = GEPA(
        metric=review_metric,
        auto="light",              # Budget preset: light/medium/heavy
        reflection_lm=reflection_lm,  # Required: strong model for reflection
        num_threads=4,
        track_stats=True,          # Track optimization statistics
        seed=42,
    )

    # To compile:
    # student = TextReviewAnalyzer()
    # optimized = optimizer.compile(
    #     student,
    #     trainset=training_examples,
    #     valset=validation_examples,
    # )
    # # Access detailed results:
    # optimized.detailed_results.best_idx
    # optimized.detailed_results.val_aggregate_scores

    dspy.settings.experimental = False
    return optimizer


# ---------------------------------------------------------------------------
# Part 10: Custom Adapters
# ---------------------------------------------------------------------------
# DSPy ships with ChatAdapter, JSONAdapter, and XMLAdapter. You can
# create your own by subclassing dspy.adapters.Adapter and implementing
# format() and parse(). Here's a minimal example.


def demonstrate_adapter_usage():
    """Show how to use different built-in adapters.

    - ChatAdapter (default): Uses [[ ## field ## ]] delimiters. Most flexible.
    - JSONAdapter: Forces JSON output. Good for structured data.
    - XMLAdapter: Forces XML tags. Some models prefer this.

    You set the adapter globally or per-request.
    """
    # Global adapter switch
    # dspy.configure(adapter=dspy.JSONAdapter())

    # Per-request adapter
    # with dspy.context(adapter=dspy.XMLAdapter()):
    #     result = module(inputs)

    return {
        "available_adapters": [
            "dspy.ChatAdapter (default)",
            "dspy.JSONAdapter",
            "dspy.XMLAdapter",
        ],
        "custom_adapter_base": "dspy.adapters.Adapter",
        "required_methods": ["format(signature, demos, inputs)", "parse(signature, completion)"],
    }


# ---------------------------------------------------------------------------
# Part 11: Save, Load, and Version
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: dspy.Module, version: str = "v1"):
    """Save an optimized pipeline for deployment.

    Convention: name your saves with version numbers so you can
    A/B test different optimized versions in production.
    """
    path = f"product_analyzer_{version}.json"
    pipeline.save(path)
    return path


def load_pipeline(version: str = "v1") -> MultimodalProductAnalyzer:
    """Load a previously saved pipeline."""
    path = f"product_analyzer_{version}.json"
    pipeline = MultimodalProductAnalyzer()
    pipeline.load(path)
    return pipeline


# ---------------------------------------------------------------------------
# Main: Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set LLM_API_KEY or ANTHROPIC_API_KEY in your .env file")
        exit(1)

    lm = dspy.LM("anthropic/claude-sonnet-4-6", api_key=api_key, max_tokens=2048)
    dspy.configure(lm=lm)

    print("=" * 60)
    print("Multimodal Product Review Analyzer — Demo")
    print("=" * 60)

    # 1. Text-only analysis
    print("\n--- Text Review Analysis ---")
    text_analyzer = TextReviewAnalyzer()
    result = text_analyzer(
        review_text="This wireless keyboard is fantastic! Great battery life, "
                    "comfortable keys, and the Bluetooth connection is rock solid. "
                    "Only downside is it's a bit pricey at $89.",
        product_category="electronics",
    )
    print(f"Sentiment: {result.result.sentiment}")
    print(f"Quality: {result.result.quality_score}/5")
    print(f"Key points: {result.result.key_points}")

    # 2. Image analysis (with test image)
    print("\n--- Image Analysis ---")
    test_image = create_test_image(200, 200, "blue")
    image_analyzer = ProductImageAnalyzer()
    img_result = image_analyzer(
        product_image=test_image,
        review_text="The blue case looks exactly like the photos online.",
    )
    print(f"Analysis: {img_result.analysis}")

    # 3. Model cascading
    print("\n--- Model Cascading ---")
    cascade = CascadingAnalyzer(confidence_threshold=0.85)
    cascade_result = cascade(
        review_text="Amazing headphones, best I've ever owned!",
        product_category="electronics",
    )
    print(f"Routed to: {cascade_result.model_tier} model")
    print(f"Sentiment: {cascade_result.result.sentiment}")

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)
