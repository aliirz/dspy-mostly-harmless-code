"""
codebase_qa.py — The Codebase Q&A System
Chapter 3: Life, the Universe, and Retrieval
DSPy: The Mostly Harmless Guide

A RAG-powered codebase Q&A system that indexes a real GitHub repository
and answers questions about the code with file citations.

Run with: poetry run python codebase_qa.py
"""

import os

import numpy as np
import dspy
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# ===================================================================
# File Loading
# ===================================================================

def load_codebase(repo_path, extensions=(".py",), max_file_lines=500):
    """Load source files from a repository directory.

    Returns a list of dicts with 'path', 'content', and 'language'.
    Skips hidden dirs, __pycache__, and files over max_file_lines
    (which are usually auto-generated or vendored).
    """
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        for fname in filenames:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, repo_path)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                if len(lines) > max_file_lines or len(lines) < 3:
                    continue
                content = ''.join(lines)
                files.append({
                    'path': rel_path,
                    'content': content,
                    'language': fname.split('.')[-1],
                })
            except (IOError, UnicodeDecodeError):
                continue
    return files


def chunk_code_files(files, chunk_size=30, overlap=5):
    """Split source files into overlapping chunks.

    Each chunk includes the file path as header context.
    Small files become a single chunk; large files get
    a sliding window with overlap for continuity.
    """
    chunks = []
    for f in files:
        lines = f['content'].split('\n')
        path = f['path']

        if len(lines) <= chunk_size:
            chunk_text = f"File: {path}\n\n{f['content']}"
            chunks.append(chunk_text)
        else:
            for start in range(0, len(lines), chunk_size - overlap):
                end = min(start + chunk_size, len(lines))
                window = '\n'.join(lines[start:end])
                chunk_text = f"File: {path} (lines {start+1}-{end})\n\n{window}"
                chunks.append(chunk_text)
                if end == len(lines):
                    break
    return chunks


# ===================================================================
# Embedder
# ===================================================================

def get_embedder():
    """Create the embedder — hosted model by default, local fallback.

    Uses OpenAI text-embedding-3-small via LiteLLM for best retrieval
    quality. Falls back to a local hash-based embedder if no OpenAI
    key is available.
    """
    if os.getenv("OPENAI_API_KEY"):
        return dspy.Embedder(
            "openai/text-embedding-3-small",
            caching=True,
            batch_size=200,
        )
    else:
        print("⚠️  No OPENAI_API_KEY found — using local tfidf_embedder.")
        print("   Retrieval quality will be lower. Set OPENAI_API_KEY for best results.")
        return dspy.Embedder(tfidf_embedder)


def tfidf_embedder(texts, dim=256):
    """Hash-based TF-IDF embedder with fixed dimensions.

    Fallback embedder for environments without an OpenAI key.
    Maps each word to a dimension via hashing, then L2-normalizes.
    Free, fast, zero API calls — but lower retrieval quality.
    """
    embeddings = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        for word in text.lower().split():
            vec[hash(word) % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        embeddings.append(vec)
    return embeddings


# ===================================================================
# Structured Output
# ===================================================================

class CodeAnswer(BaseModel):
    """An answer about code with file citations."""
    answer: str = Field(
        description="A clear, technical answer to the question based on the "
        "retrieved source code. Reference specific classes, functions, "
        "and patterns you found. If the context doesn't contain enough "
        "information, say so honestly."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well the retrieved code addresses the question: "
        "'high' = found the exact relevant code, 'medium' = found "
        "related code but not the specific answer, 'low' = retrieved "
        "code isn't relevant to the question."
    )
    cited_files: list[str] = Field(
        description="File paths referenced in the answer, extracted from "
        "the 'File:' headers in the context passages."
    )
    key_snippets: list[str] = Field(
        description="1-3 short, relevant code snippets (under 5 lines each) "
        "from the context that directly support the answer."
    )


# ===================================================================
# Signatures
# ===================================================================

class AnswerFromCode(dspy.Signature):
    """Answer a question about a codebase using retrieved source code.
    Cite specific files and include relevant code snippets.
    If the code doesn't contain the answer, say so — don't guess."""

    context: str = dspy.InputField(
        desc="Retrieved source code chunks with file path headers"
    )
    question: str = dspy.InputField(desc="The user's question about the codebase")
    response: CodeAnswer = dspy.OutputField(desc="A cited answer with code snippets")


# ===================================================================
# Pipeline Modules
# ===================================================================

class CodebaseQA(dspy.Module):
    """Simple RAG: retrieve code chunks → answer the question."""

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        results = self.retriever(question)
        context = "\n\n---\n\n".join(results.passages)
        return self.answer(context=context, question=question)


class CodebaseExplorer(dspy.Module):
    """Full codebase Q&A with file citations and code snippets."""

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answer = dspy.ChainOfThought(AnswerFromCode)

    def forward(self, question):
        results = self.retriever(question)
        context = "\n\n---\n\n".join(results.passages)
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

    stop_words = {"a", "an", "the", "is", "it", "of", "to", "and", "in",
                  "for", "that", "this", "with", "as", "on", "by", "or"}
    gold_words -= stop_words
    pred_words -= stop_words

    if not gold_words:
        return 0.0

    overlap = len(gold_words & pred_words)
    return overlap / len(gold_words)


# ===================================================================
# Evaluation Dataset
# ===================================================================

CODEBASE_QA_DATASET = [
    dspy.Example(
        question="What class handles dense vector retrieval in DSPy?",
        answer="The Embeddings class in dspy.retrievers.embeddings handles "
        "dense vector retrieval. It takes a corpus, embedder, and k parameter."
    ).with_inputs("question"),

    dspy.Example(
        question="How does ChainOfThought differ from Predict?",
        answer="ChainOfThought extends Predict by adding a reasoning field. "
        "It uses an extended signature that includes step-by-step reasoning "
        "before the final answer."
    ).with_inputs("question"),

    dspy.Example(
        question="What does dspy.Evaluate do?",
        answer="dspy.Evaluate runs a DSPy program against a devset using "
        "a metric function. It returns an EvaluationResult with a score "
        "and per-example results."
    ).with_inputs("question"),

    dspy.Example(
        question="How are DSPy signatures defined?",
        answer="Signatures can be defined inline as strings like "
        "'question -> answer' or as classes inheriting from dspy.Signature "
        "with typed InputField and OutputField attributes."
    ).with_inputs("question"),

    dspy.Example(
        question="What is the role of adapters in DSPy?",
        answer="Adapters format signatures into LM prompts. ChatAdapter "
        "is the default. JSONAdapter forces JSON output. The adapter "
        "also handles parsing LM responses back into structured output."
    ).with_inputs("question"),

    dspy.Example(
        question="How does DSPy handle saving and loading programs?",
        answer="DSPy programs can be saved and loaded to preserve optimized "
        "state. The Embeddings class has save() and from_saved() methods "
        "for persisting retrieval indices to disk."
    ).with_inputs("question"),
]


# ===================================================================
# Main
# ===================================================================

def main():
    # --- Setup ---
    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=1500,
    )
    dspy.configure(lm=lm)

    # --- Load and Index the Codebase ---
    repo_path = os.environ.get(
        "DSPY_REPO_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "dspy", "dspy"),
    )

    print("=" * 60)
    print("THE CODEBASE Q&A SYSTEM")
    print("=" * 60)
    print(f"\nIndexing: {repo_path}")

    files = load_codebase(repo_path)
    print(f"Loaded {len(files)} files")

    chunks = chunk_code_files(files)
    print(f"Created {len(chunks)} chunks")

    # Use hosted embedder (text-embedding-3-small) by default
    embedder = get_embedder()

    retriever = dspy.Embeddings(
        embedder=embedder,
        corpus=chunks,
        k=5,
    )
    print("Retriever built!")

    # --- Ask a Question ---
    explorer = CodebaseExplorer(retriever=retriever)

    question = "How does dspy.ChainOfThought extend Predict?"
    print(f"\nQuestion: {question}")
    print("-" * 40)

    result = explorer(question=question)

    print(f"\nAnswer: {result.response.answer}")
    print(f"\nConfidence: {result.response.confidence}")
    print(f"\nCited files:")
    for f in result.response.cited_files:
        print(f"  📄 {f}")
    print(f"\nKey snippets:")
    for s in result.response.key_snippets:
        print(f"  ```\n  {s}\n  ```")

    # --- Evaluate ---
    print(f"\n{'=' * 60}")
    print("EVALUATION")
    print("=" * 60)

    evaluator = dspy.Evaluate(
        devset=CODEBASE_QA_DATASET,
        metric=answer_quality_metric,
        num_threads=1,
        display_progress=True,
    )

    score = evaluator(explorer)
    print(f"\nCodebaseExplorer score: {score}")


if __name__ == "__main__":
    main()
