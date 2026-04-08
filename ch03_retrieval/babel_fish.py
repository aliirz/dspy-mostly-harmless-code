"""
babel_fish.py — The Babel Fish Documentation Bot
Chapter 3: Life, the Universe, and Retrieval
DSPy: The Mostly Harmless Guide

A RAG-powered documentation Q&A system that retrieves relevant passages
and answers questions with citations.

Run with: poetry run python babel_fish.py
"""

import os

import numpy as np
import dspy
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# ===================================================================
# Knowledge Base
# ===================================================================

DOCS_CORPUS = [
    # Core Concepts
    "DSPy Signatures define the input and output fields for language model calls. "
    "A signature like 'question -> answer' tells DSPy what the LM should receive and produce. "
    "Class-based signatures allow adding descriptions, types, and constraints to each field.",

    "DSPy Modules are the building blocks of programs. Every module has a forward() method "
    "that defines its behavior. Modules can contain other modules, just like PyTorch nn.Module. "
    "The two most common modules are dspy.Predict and dspy.ChainOfThought.",

    "dspy.Predict is the simplest module. It takes a signature and calls the language model once. "
    "dspy.ChainOfThought extends Predict by asking the model to show its reasoning step-by-step "
    "before producing the final answer. ChainOfThought is better for complex reasoning tasks.",

    "dspy.ChainOfThought adds a 'reasoning' field to the output automatically. This field "
    "contains the model's step-by-step thought process. You don't need to define it in your "
    "signature — DSPy injects it for you. Access it via result.reasoning.",

    # Configuration
    "dspy.LM creates a language model connection. It accepts any model string supported by "
    "LiteLLM, like 'anthropic/claude-sonnet-4-6' or 'openai/gpt-5.4'. Parameters include "
    "temperature, max_tokens, and stop sequences. Configure globally with dspy.configure(lm=lm).",

    "dspy.configure() sets global defaults for LM, adapter, and retrieval model. "
    "Use dspy.context(lm=other_lm) as a context manager to temporarily switch models "
    "for specific pipeline steps. This is the recommended way to use different models "
    "for different steps in a pipeline.",

    # Adapters
    "DSPy adapters control how signatures are formatted into LM prompts. ChatAdapter is the "
    "default and works well with most models. JSONAdapter forces JSON output format, which "
    "is useful for models that support structured output natively. XMLAdapter uses XML tags.",

    "When ChatAdapter fails to parse output, it automatically falls back to JSONAdapter. "
    "You can disable this with adapter.use_json_adapter_fallback = False. This is useful "
    "when you want strict control over the output format.",

    # Pydantic Integration
    "DSPy supports Pydantic BaseModel as output types. Define a Pydantic model with typed "
    "fields and use it as an OutputField type. DSPy will instruct the LM to produce output "
    "matching your schema and validate it automatically. Use Literal types for constrained values.",

    "When using Pydantic models with DSPy, field descriptions are sent to the LM as part of "
    "the prompt. Good descriptions dramatically improve output quality. Always include desc= "
    "in your Field() definitions — it's not just documentation, it's prompt engineering.",

    # Retrieval
    "dspy.Embeddings is a retrieval module that uses dense vector search. Initialize it with "
    "a corpus (list of strings), an embedder function, and k (number of results). It computes "
    "embeddings for the corpus upfront and uses similarity search to find relevant passages.",

    "The embedder parameter in dspy.Embeddings can be a hosted model string like "
    "'openai/text-embedding-3-small' or a custom callable that takes a list of strings and "
    "returns a numpy array of embeddings. Custom embedders must produce consistent dimensions.",

    "dspy.Embedder wraps embedding functions into a consistent interface. It handles batching, "
    "normalization, and caching. Use dspy.Embedder(model_name) for hosted models or "
    "dspy.Embedder(your_function) for custom embedding functions.",

    # Evaluation
    "dspy.Evaluate runs a DSPy program against a dataset and measures performance with a metric. "
    "Initialize with devset (list of Examples), metric function, and num_threads. The metric "
    "function takes (example, prediction) and returns a score between 0 and 1.",

    "Built-in metrics include dspy.evaluate.metrics.answer_exact_match for exact string matching, "
    "and SemanticF1 for LLM-based semantic similarity. You can also write custom metrics that "
    "check for specific properties of the output.",

    # Optimizers (preview)
    "DSPy optimizers automatically improve your program by tuning prompts, selecting demonstrations, "
    "or fine-tuning weights. Common optimizers include BootstrapFewShot (self-taught demos), "
    "MIPROv2 (instruction optimization), and LabeledFewShot (from your labeled examples).",

    "To use an optimizer, you need three things: a DSPy program (Module), a training set "
    "(list of Examples), and a metric function. The optimizer runs your program many times, "
    "trying different configurations, and returns the best-performing version.",

    # Production Patterns
    "For production RAG systems, always evaluate retrieval quality separately from generation "
    "quality. A brilliant LM can't fix bad retrieval — if the right documents aren't in the "
    "context, the answer will be wrong or hallucinated, no matter how smart the model is.",

    "The save() and load() methods on dspy.Embeddings let you persist your index to disk. "
    "This avoids recomputing embeddings on every startup. Use save(path) after building "
    "your index and Embeddings.from_saved(path, embedder) to load it back.",

    # Common Mistakes
    "A common mistake in RAG systems is chunking documents too aggressively. If your chunks "
    "are too small, they lose context. If they're too large, retrieval becomes imprecise. "
    "For most use cases, 2-4 sentence chunks with 1-sentence overlap work well.",

    "Another common RAG mistake is not evaluating retrieval independently. Your pipeline might "
    "score well overall because the LM is good at guessing, but your retriever might be "
    "returning irrelevant documents. Always measure recall@k for your retriever.",
]


# ===================================================================
# Embedder
# ===================================================================

def tfidf_embedder(texts, dim=128):
    """Hash-based TF-IDF embedder with fixed dimensions.

    Uses word hashing to map tokens to dimension indices:
    - No vocabulary building needed (consistent dimensions)
    - Works for any text, any language
    - Fast enough for thousands of documents
    """
    embeddings = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            idx = hash(word) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)
    return embeddings


# ===================================================================
# Structured Output
# ===================================================================

class CitedAnswer(BaseModel):
    """An answer with source citations."""
    answer: str = Field(
        description="Clear, concise answer based on the provided context. "
        "Say honestly if the context doesn't contain enough information."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence based on context quality: 'high' = directly "
        "addressed, 'medium' = related but incomplete, 'low' = doesn't help."
    )
    source_quotes: list[str] = Field(
        description="1-3 direct quotes from the context supporting the answer."
    )


# ===================================================================
# Signatures
# ===================================================================

class AnswerWithCitations(dspy.Signature):
    """Answer a question using ONLY the provided context passages.
    Cite specific quotes from the context to support your answer.
    If the context doesn't contain the answer, say so — don't guess."""

    context: str = dspy.InputField(
        desc="Retrieved documentation passages, separated by blank lines"
    )
    question: str = dspy.InputField(desc="The user's question")
    response: CitedAnswer = dspy.OutputField(
        desc="A cited answer based on the context"
    )


# ===================================================================
# Pipeline Modules
# ===================================================================

class BabelFishQA(dspy.Module):
    """Simple RAG: retrieve → answer."""

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        results = self.retriever(question)
        context = "\n\n".join(results.passages)
        return self.answer(context=context, question=question)


class BabelFishBot(dspy.Module):
    """Full RAG pipeline: retrieve docs → answer with citations."""

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answer = dspy.ChainOfThought(AnswerWithCitations)

    def forward(self, question):
        results = self.retriever(question)
        context = "\n\n".join(results.passages)
        prediction = self.answer(context=context, question=question)

        return dspy.Prediction(
            response=prediction.response,
            retrieved_passages=results.passages,
            passage_indices=results.indices,
        )


# ===================================================================
# Metric
# ===================================================================

def answer_quality_metric(example, pred, trace=None):
    """Score an answer based on keyword overlap with the gold answer."""
    if hasattr(pred, 'response') and hasattr(pred.response, 'answer'):
        pred_text = pred.response.answer.lower()
    elif hasattr(pred, 'answer'):
        pred_text = pred.answer.lower()
    else:
        return 0.0

    gold_words = set(example.answer.lower().split())
    pred_words = set(pred_text.split())

    stop_words = {"a", "an", "the", "is", "it", "of", "to", "and", "in", "for", "that", "this"}
    gold_words -= stop_words
    pred_words -= stop_words

    if not gold_words:
        return 0.0

    overlap = len(gold_words & pred_words)
    return overlap / len(gold_words)


# ===================================================================
# Evaluation Dataset
# ===================================================================

QA_DATASET = [
    dspy.Example(
        question="What is a DSPy Signature?",
        answer="A Signature defines the input and output fields for language model "
        "calls. It tells DSPy what the LM should receive and produce."
    ).with_inputs("question"),

    dspy.Example(
        question="What's the difference between Predict and ChainOfThought?",
        answer="Predict calls the LM once. ChainOfThought extends Predict by asking "
        "the model to show step-by-step reasoning before the final answer."
    ).with_inputs("question"),

    dspy.Example(
        question="How do I use different LMs for different pipeline steps?",
        answer="Use dspy.context(lm=other_lm) as a context manager to temporarily "
        "switch models for specific steps. This is the recommended approach."
    ).with_inputs("question"),

    dspy.Example(
        question="What is dspy.Embeddings?",
        answer="dspy.Embeddings is a retrieval module that uses dense vector search. "
        "It takes a corpus, an embedder, and k, and builds a searchable index."
    ).with_inputs("question"),

    dspy.Example(
        question="How do I save and load a retrieval index?",
        answer="Use the save(path) method to persist an Embeddings index to disk, "
        "and Embeddings.from_saved(path, embedder) to load it back."
    ).with_inputs("question"),

    dspy.Example(
        question="What does the adapter do in DSPy?",
        answer="Adapters control how signatures are formatted into LM prompts. "
        "ChatAdapter is the default, JSONAdapter forces JSON output, "
        "and XMLAdapter uses XML tags."
    ).with_inputs("question"),
]


# ===================================================================
# Main
# ===================================================================

def main():
    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=1500,
    )
    dspy.configure(lm=lm)

    print("=" * 60)
    print("THE BABEL FISH DOCUMENTATION BOT")
    print("=" * 60)

    # Build retriever
    retriever = dspy.Embeddings(
        embedder=tfidf_embedder,
        corpus=DOCS_CORPUS,
        k=3,
    )

    # Build the bot
    bot = BabelFishBot(retriever=retriever)

    # Ask a question
    question = "What are DSPy adapters and when should I use JSONAdapter?"
    print(f"\nQuestion: {question}")
    print("-" * 40)

    result = bot(question=question)

    print(f"\nAnswer: {result.response.answer}")
    print(f"\nConfidence: {result.response.confidence}")
    print(f"\nSource quotes:")
    for q in result.response.source_quotes:
        print(f"  • \"{q}\"")
    print(f"\nRetrieved from corpus indices: {result.passage_indices}")

    # Evaluate
    print(f"\n{'=' * 60}")
    print("EVALUATION")
    print("=" * 60)

    evaluator = dspy.Evaluate(
        devset=QA_DATASET,
        metric=answer_quality_metric,
        num_threads=1,
        display_progress=True,
    )

    score = evaluator(bot)
    print(f"\nBabelFishBot score: {score}")


if __name__ == "__main__":
    main()
