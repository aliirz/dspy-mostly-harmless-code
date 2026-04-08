"""
Chapter 6: Mostly Harmless (in Production) — Scaling, Observability, and Battle Scars

A Production-Ready Content Moderation Pipeline that demonstrates:
- Caching strategies (DSPy's two-level cache)
- Cost tracking with dspy.track_usage()
- Custom callbacks for observability
- Fallback chains (primary → secondary model)
- Streaming with dspy.streamify()
- Async processing with dspy.asyncify()
- Batch processing with module.batch()
- Save/Load for deployment
- Per-request configuration with dspy.context()
- FastAPI deployment with streaming responses
"""

import asyncio
import json
import os
import time
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

import dspy
from dotenv import load_dotenv
from dspy.utils.callback import BaseCallback
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Part 1: The Base Content Moderation Pipeline
# ---------------------------------------------------------------------------
# Before we add any production armor, we need a pipeline worth protecting.
# Our moderator classifies user-generated content, explains its reasoning,
# and recommends an action — all in a single structured pass.

class ModerationCategory(str, Enum):
    """Categories of content that might need moderation."""
    SAFE = "safe"
    SPAM = "spam"
    TOXIC = "toxic"
    MISINFORMATION = "misinformation"
    ADULT = "adult"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"


class ModerationDecision(BaseModel):
    """Structured output for a moderation decision."""
    category: Literal["safe", "spam", "toxic", "misinformation", "adult", "violence", "self_harm"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0 to 1")
    action: Literal["approve", "flag_for_review", "reject"]
    explanation: str = Field(description="Brief explanation of the moderation decision")


class ModerateContent(dspy.Signature):
    """You are a content moderator for a social platform. Analyze the given
    user-generated content and determine whether it should be approved,
    flagged for human review, or rejected. Be fair and avoid over-censoring
    legitimate speech. Only reject content that clearly violates policies."""

    content: str = dspy.InputField(desc="The user-generated content to moderate")
    context: str = dspy.InputField(
        desc="Additional context about where this content appears (e.g., 'comment on a cooking blog')",
        default="general social media post",
    )
    decision: ModerationDecision = dspy.OutputField(desc="The structured moderation decision")


class ContentModerator(dspy.Module):
    """A basic content moderation pipeline.

    This is the module we'll progressively armor with production features
    throughout the chapter.
    """

    def __init__(self):
        self.moderate = dspy.ChainOfThought(ModerateContent)

    def forward(self, content: str, context: str = "general social media post") -> dspy.Prediction:
        result = self.moderate(content=content, context=context)
        return dspy.Prediction(decision=result.decision)


# ---------------------------------------------------------------------------
# Part 2: Caching — Because Calling an LLM Twice for the Same Input is a Sin
# ---------------------------------------------------------------------------
# DSPy has a two-level cache: in-memory (LRU) and on-disk (FanoutCache).
# By default, caching is ON. Here's how to configure it.

def configure_production_cache(
    disk_dir: str = os.path.expanduser("~/.dspy_cache"),
    disk_size_gb: float = 1.0,
    memory_entries: int = 10_000,
):
    """Configure DSPy's cache for production use.

    The defaults are sensible for most deployments. Adjust disk_size_gb
    based on your traffic volume — 1 GB holds roughly 500K cached responses.
    """
    from dspy.clients import configure_cache

    configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=disk_dir,
        disk_size_limit_bytes=int(disk_size_gb * 1e9),
        memory_max_entries=memory_entries,
    )


def demonstrate_cache_behavior():
    """Show that DSPy caches LLM calls by default.

    Call the same input twice and observe that the second call
    is served from cache (near-instant).
    """
    moderator = ContentModerator()

    # First call — hits the LLM
    start = time.time()
    result1 = moderator(content="This is a great product, highly recommend!")
    first_call_time = time.time() - start

    # Second call — same input, served from cache
    start = time.time()
    result2 = moderator(content="This is a great product, highly recommend!")
    second_call_time = time.time() - start

    return {
        "first_call_seconds": round(first_call_time, 3),
        "second_call_seconds": round(second_call_time, 3),
        "results_match": result1.decision == result2.decision,
        "speedup": round(first_call_time / max(second_call_time, 0.001), 1),
    }


# To disable caching for a specific LM (useful during development):
# lm = dspy.LM("anthropic/claude-sonnet-4-6", cache=False)

# To disable caching globally:
# dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-6", cache=False))


# ---------------------------------------------------------------------------
# Part 3: Cost Tracking — Know What You're Spending Before Finance Asks
# ---------------------------------------------------------------------------

def moderate_with_cost_tracking(content: str, context: str = "general social media post"):
    """Moderate content while tracking token usage and costs.

    dspy.track_usage() is a context manager that captures every LLM call
    made within its scope — including calls from nested modules.
    """
    moderator = ContentModerator()

    with dspy.track_usage() as tracker:
        result = moderator(content=content, context=context)

    # Get aggregated usage by model
    usage = tracker.get_total_tokens()

    return {
        "decision": result.decision,
        "usage_by_model": usage,
    }


class BudgetExceededError(Exception):
    """Raised when a cost budget is exceeded."""
    pass


class BudgetAwareModerator(dspy.Module):
    """A moderator that tracks cumulative cost and raises an alarm
    when spending exceeds a threshold.

    This pattern is essential for production: you set a per-hour or
    per-day budget, and the service degrades gracefully instead of
    silently burning through your API credits.
    """

    def __init__(self, budget_limit_tokens: int = 100_000):
        self.moderate = dspy.ChainOfThought(ModerateContent)
        self.budget_limit = budget_limit_tokens
        self.total_tokens_used = 0

    def forward(self, content: str, context: str = "general social media post") -> dspy.Prediction:
        if self.total_tokens_used >= self.budget_limit:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.total_tokens_used}/{self.budget_limit} tokens used. "
                f"Refusing new requests until budget is reset."
            )

        with dspy.track_usage() as tracker:
            result = self.moderate(content=content, context=context)

        # Accumulate tokens
        usage = tracker.get_total_tokens()
        for model_usage in usage.values():
            self.total_tokens_used += model_usage.get("prompt_tokens", 0)
            self.total_tokens_used += model_usage.get("completion_tokens", 0)

        return dspy.Prediction(
            decision=result.decision,
            tokens_used=self.total_tokens_used,
            budget_remaining=self.budget_limit - self.total_tokens_used,
        )

    def reset_budget(self):
        """Reset the token counter. Call this on a schedule (e.g., hourly)."""
        self.total_tokens_used = 0


# ---------------------------------------------------------------------------
# Part 4: Callbacks — Your Pipeline's Black Box Recorder
# ---------------------------------------------------------------------------

class ModerationLogger(BaseCallback):
    """A callback that logs every LLM call and module execution.

    In production, you'd send these events to your observability stack
    (Datadog, Prometheus, OpenTelemetry, etc.). Here we keep it simple
    with an in-memory log.
    """

    def __init__(self):
        self.events = []

    def on_module_start(self, call_id, instance, inputs):
        self.events.append({
            "type": "module_start",
            "call_id": call_id,
            "module": instance.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "inputs": {k: str(v)[:100] for k, v in inputs.items()},  # Truncate for safety
        })

    def on_module_end(self, call_id, outputs, exception=None):
        self.events.append({
            "type": "module_end",
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "success": exception is None,
            "error": str(exception) if exception else None,
        })

    def on_lm_start(self, call_id, instance, inputs):
        self.events.append({
            "type": "lm_start",
            "call_id": call_id,
            "model": getattr(instance, "model", "unknown"),
            "timestamp": datetime.now().isoformat(),
        })

    def on_lm_end(self, call_id, outputs, exception=None):
        event = {
            "type": "lm_end",
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "success": exception is None,
        }
        if outputs and "usage" in outputs:
            event["usage"] = outputs["usage"]
        if exception:
            event["error"] = str(exception)
        self.events.append(event)

    def get_summary(self):
        """Get a summary of all logged events."""
        lm_calls = [e for e in self.events if e["type"] == "lm_start"]
        errors = [e for e in self.events if e.get("error")]
        return {
            "total_events": len(self.events),
            "lm_calls": len(lm_calls),
            "errors": len(errors),
            "error_details": [e["error"] for e in errors],
        }


def moderate_with_logging(content: str):
    """Demonstrate callback-based observability."""
    logger = ModerationLogger()

    # Callbacks can be set globally...
    # dspy.configure(callbacks=[logger])

    # ...or per-module via dspy.context()
    moderator = ContentModerator()
    with dspy.context(callbacks=[logger]):
        result = moderator(content=content)

    return {
        "decision": result.decision,
        "log_summary": logger.get_summary(),
        "events": logger.events,
    }


# ---------------------------------------------------------------------------
# Part 5: Fallback Chains — Because Even the Best Models Have Bad Days
# ---------------------------------------------------------------------------

def create_fallback_chain():
    """Create a list of LM instances ordered by preference.

    The idea: try your best model first. If it fails (rate limit, timeout,
    outage), fall through to the next one. Your users see degraded quality,
    not an error page.
    """
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    # Primary: Claude Sonnet (best quality)
    primary = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        api_key=api_key,
        max_tokens=1024,
        num_retries=2,  # Retry twice before falling through
    )

    # Secondary: Claude Haiku (faster, cheaper, still good)
    secondary = dspy.LM(
        "anthropic/claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=1024,
        num_retries=2,
    )

    return [primary, secondary]


class FallbackModerator(dspy.Module):
    """A moderator that tries multiple models in order.

    This is the production pattern for high-availability services.
    If Model A fails, fall through to Model B, then Model C.
    Log which model actually served each request for monitoring.
    """

    def __init__(self, models: list[dspy.LM] | None = None):
        self.moderate = dspy.ChainOfThought(ModerateContent)
        self.models = models or create_fallback_chain()

    def forward(self, content: str, context: str = "general social media post") -> dspy.Prediction:
        last_error = None

        for i, model in enumerate(self.models):
            try:
                with dspy.context(lm=model):
                    result = self.moderate(content=content, context=context)
                return dspy.Prediction(
                    decision=result.decision,
                    model_used=model.model,
                    fallback_level=i,
                )
            except Exception as e:
                last_error = e
                # Log the failure but keep trying
                print(f"Model {model.model} failed: {e}. Trying next...")
                continue

        # All models failed — this is a real production emergency
        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Part 6: Streaming — Real-Time Responses for Impatient Humans
# ---------------------------------------------------------------------------

def create_streaming_moderator():
    """Wrap a moderator with dspy.streamify() for incremental output.

    streamify() returns a function that produces an async generator.
    Each yielded value is either a StatusMessage, a ModelResponseStream
    chunk, or the final Prediction.
    """
    moderator = ContentModerator()
    return dspy.streamify(moderator)


async def demonstrate_streaming(content: str):
    """Show how streaming works with a content moderator.

    In production, you'd pipe these chunks directly to a WebSocket
    or Server-Sent Events (SSE) endpoint.
    """
    streaming_moderator = create_streaming_moderator()

    chunks = []
    final_prediction = None

    output = streaming_moderator(content=content)
    async for value in output:
        if isinstance(value, dspy.Prediction):
            final_prediction = value
        else:
            chunks.append(str(value))

    return {
        "num_chunks": len(chunks),
        "final_prediction": final_prediction,
    }


# ---------------------------------------------------------------------------
# Part 7: Async — Because Blocking is So Last Century
# ---------------------------------------------------------------------------

async def moderate_batch_async(contents: list[str]):
    """Process multiple moderation requests concurrently using asyncify.

    dspy.asyncify() wraps a synchronous DSPy module for async execution
    in a thread pool. This is the simplest path to concurrent processing.
    """
    moderator = ContentModerator()
    async_moderator = dspy.asyncify(moderator)

    # Fire all requests concurrently
    tasks = [async_moderator(content=c) for c in contents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        r.decision if isinstance(r, dspy.Prediction) else {"error": str(r)}
        for r in results
    ]


# ---------------------------------------------------------------------------
# Part 8: Batch Processing — The module.batch() Way
# ---------------------------------------------------------------------------

def moderate_batch_sync(contents: list[str], num_threads: int = 4):
    """Process a batch of content using DSPy's built-in batch() method.

    module.batch() handles threading, error collection, progress bars,
    and straggler detection automatically. It's the recommended way
    to process bulk workloads.
    """
    moderator = ContentModerator()

    # Create dspy.Example objects for batch processing
    examples = [dspy.Example(content=c).with_inputs("content") for c in contents]

    results, failed, errors = moderator.batch(
        examples,
        num_threads=num_threads,
        return_failed_examples=True,
        provide_traceback=True,
    )

    return {
        "succeeded": len(results),
        "failed": len(failed),
        "results": results,
        "errors": [str(e) for e in errors],
    }


# ---------------------------------------------------------------------------
# Part 9: Save and Load — Ship It Like a Real Artifact
# ---------------------------------------------------------------------------

def save_moderator(moderator: ContentModerator, path: str = "moderator_v1.json"):
    """Save a moderator's state for deployment.

    Use .json for portable, inspectable state (demos and instructions).
    Use .pkl if you need to serialize complex Python objects.
    """
    moderator.save(path)
    return path


def load_moderator(path: str = "moderator_v1.json") -> ContentModerator:
    """Load a previously saved moderator.

    This is how you deploy an optimized module: optimize once,
    save the state, load it in your production service.
    """
    moderator = ContentModerator()
    moderator.load(path)
    return moderator


# For saving the entire program (including code):
# moderator.save("moderator_v1_full/", save_program=True)
# loaded = dspy.load("moderator_v1_full/")


# ---------------------------------------------------------------------------
# Part 10: Per-Request Configuration — Thread-Safe Customization
# ---------------------------------------------------------------------------

def moderate_with_custom_config(
    content: str,
    model: str = "anthropic/claude-sonnet-4-6",
    temperature: float = 0.0,
):
    """Use dspy.context() for per-request model configuration.

    This is thread-safe — each request can use a different model
    without affecting other concurrent requests. Essential for A/B
    testing models in production.
    """
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    request_lm = dspy.LM(model, api_key=api_key, max_tokens=1024, temperature=temperature)
    moderator = ContentModerator()

    with dspy.context(lm=request_lm):
        result = moderator(content=content)

    return result.decision


# ---------------------------------------------------------------------------
# Part 11: FastAPI Deployment — The Full Production Service
# ---------------------------------------------------------------------------

def create_app():
    """Create a FastAPI application wrapping our content moderator.

    This is the deployment pattern: a stateless HTTP service that
    loads an optimized module at startup, handles concurrent requests
    with proper error handling, and exposes health checks for
    your orchestrator (Kubernetes, ECS, etc.).
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(
        title="Content Moderation Service",
        description="Production-ready content moderation powered by DSPy",
        version="1.0.0",
    )

    # --- Startup: configure LM and load optimized module ---

    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    # The global LM — used unless overridden per-request
    primary_lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        api_key=api_key,
        max_tokens=1024,
        num_retries=3,
    )
    dspy.configure(lm=primary_lm)

    # The moderator instance — shared across requests (stateless forward pass)
    moderator = ContentModerator()

    # The streaming variant
    streaming_moderator = dspy.streamify(moderator)

    # The callback logger (in production, replace with your telemetry SDK)
    request_logger = ModerationLogger()

    # --- Request/Response Models ---

    class ModerationRequest(BaseModel):
        content: str = Field(min_length=1, max_length=10_000)
        context: str = Field(default="general social media post")

    class ModerationResponse(BaseModel):
        decision: ModerationDecision
        model_used: str = "anthropic/claude-sonnet-4-6"
        cached: bool = False
        processing_time_ms: float = 0.0

    class BatchRequest(BaseModel):
        items: list[ModerationRequest] = Field(min_length=1, max_length=100)

    class HealthResponse(BaseModel):
        status: str = "healthy"
        model: str = ""
        cache_enabled: bool = True

    # --- Endpoints ---

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check for load balancers and orchestrators."""
        return HealthResponse(
            status="healthy",
            model=primary_lm.model,
            cache_enabled=True,
        )

    @app.post("/moderate", response_model=ModerationResponse)
    async def moderate(request: ModerationRequest):
        """Moderate a single piece of content."""
        start = time.time()
        try:
            with dspy.context(callbacks=[request_logger]):
                with dspy.track_usage() as tracker:
                    result = moderator(
                        content=request.content,
                        context=request.context,
                    )

            elapsed_ms = (time.time() - start) * 1000
            return ModerationResponse(
                decision=result.decision,
                model_used=primary_lm.model,
                processing_time_ms=round(elapsed_ms, 2),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/moderate/stream")
    async def moderate_stream(request: ModerationRequest):
        """Moderate content with streaming response (SSE)."""

        async def event_generator():
            output = streaming_moderator(content=request.content, context=request.context)
            async for value in output:
                if isinstance(value, dspy.Prediction):
                    yield f"data: {json.dumps({'type': 'result', 'decision': value.decision.model_dump() if hasattr(value.decision, 'model_dump') else str(value.decision)})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': str(value)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    @app.post("/moderate/batch")
    async def moderate_batch_endpoint(request: BatchRequest):
        """Moderate multiple items in parallel."""
        examples = [
            dspy.Example(content=item.content, context=item.context).with_inputs("content", "context")
            for item in request.items
        ]

        results, failed, errors = moderator.batch(
            examples,
            num_threads=min(len(examples), 8),
            return_failed_examples=True,
        )

        return {
            "results": [
                {"decision": r.decision.model_dump() if hasattr(r.decision, "model_dump") else str(r.decision)}
                for r in results
            ],
            "failed_count": len(failed),
            "errors": [str(e) for e in errors],
        }

    return app


# ---------------------------------------------------------------------------
# Part 12: The Data Flywheel — Collecting Traces for Re-Optimization
# ---------------------------------------------------------------------------

class DataFlywheelCallback(BaseCallback):
    """Capture production traces that can be fed back into optimization.

    The flywheel: deploy → collect traces → label edge cases →
    re-optimize with new data → redeploy. This callback captures
    the raw data you need for the "collect traces" step.
    """

    def __init__(self, output_path: str = "production_traces.jsonl"):
        self.output_path = output_path
        self.traces = []

    def on_module_start(self, call_id, instance, inputs):
        self.traces.append({
            "call_id": call_id,
            "module": instance.__class__.__name__,
            "inputs": {k: str(v) for k, v in inputs.items()},
            "timestamp": datetime.now().isoformat(),
        })

    def on_module_end(self, call_id, outputs, exception=None):
        # Find the matching start trace
        for trace in reversed(self.traces):
            if trace.get("call_id") == call_id and "outputs" not in trace:
                if outputs and hasattr(outputs, "items"):
                    trace["outputs"] = {k: str(v) for k, v in outputs.items()}
                elif outputs:
                    trace["outputs"] = str(outputs)
                trace["error"] = str(exception) if exception else None
                break

    def flush(self):
        """Write accumulated traces to a JSONL file for later analysis."""
        with open(self.output_path, "a") as f:
            for trace in self.traces:
                f.write(json.dumps(trace) + "\n")
        count = len(self.traces)
        self.traces = []
        return count


# ---------------------------------------------------------------------------
# Main: Quick demo of the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set LLM_API_KEY or ANTHROPIC_API_KEY in your .env file")
        exit(1)

    lm = dspy.LM("anthropic/claude-sonnet-4-6", api_key=api_key, max_tokens=1024)
    dspy.configure(lm=lm)

    print("=" * 60)
    print("Content Moderation Pipeline — Production Demo")
    print("=" * 60)

    # 1. Basic moderation
    print("\n--- Basic Moderation ---")
    moderator = ContentModerator()
    result = moderator(content="I love this recipe! The chocolate cake turned out amazing.")
    print(f"Decision: {result.decision}")

    # 2. Cost tracking
    print("\n--- Cost Tracking ---")
    tracked = moderate_with_cost_tracking("Buy cheap watches at scamsite.com!!!")
    print(f"Decision: {tracked['decision']}")
    print(f"Usage: {tracked['usage_by_model']}")

    # 3. Callback logging
    print("\n--- Callback Logging ---")
    logged = moderate_with_logging("This product is terrible and the company is a fraud")
    print(f"Decision: {logged['decision']}")
    print(f"Log summary: {logged['log_summary']}")

    # 4. Cache demonstration
    print("\n--- Cache Behavior ---")
    cache_demo = demonstrate_cache_behavior()
    print(f"First call: {cache_demo['first_call_seconds']}s")
    print(f"Second call: {cache_demo['second_call_seconds']}s")
    print(f"Speedup: {cache_demo['speedup']}x")

    print("\n" + "=" * 60)
    print("All demos complete. Run 'uvicorn content_moderator:app --reload'")
    print("to start the FastAPI service.")
    print("=" * 60)

    # Create the FastAPI app instance for uvicorn
    app = create_app()
