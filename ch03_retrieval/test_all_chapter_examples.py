"""
test_all_chapter_examples.py - Chapter 3: EVERY code block tested
DSPy: The Mostly Harmless Guide

This file tests EVERY code example from the chapter markdown.
If it appears in the chapter, it gets run here.
"""

import os
import sys
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import dspy
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from codebase_qa import (
    load_codebase, chunk_code_files, tfidf_embedder, get_embedder,
    CodeAnswer, AnswerFromCode,
    CodebaseQA, CodebaseExplorer,
    answer_quality_metric, CODEBASE_QA_DATASET,
)

API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Path to the DSPy source code
REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dspy", "dspy")

# Global state: loaded once, reused across tests
_files = None
_chunks = None
_retriever = None
_openai_retriever = None


def get_files():
    global _files
    if _files is None:
        _files = load_codebase(REPO_PATH)
    return _files


def get_chunks():
    global _chunks
    if _chunks is None:
        _chunks = chunk_code_files(get_files())
    return _chunks


def get_tfidf_retriever():
    """Retriever using local tfidf embedder (always available)."""
    global _retriever
    if _retriever is None:
        _retriever = dspy.Embeddings(embedder=tfidf_embedder, corpus=get_chunks(), k=5)
    return _retriever


def get_openai_retriever():
    """Retriever using OpenAI text-embedding-3-small (requires OPENAI_API_KEY)."""
    global _openai_retriever
    if _openai_retriever is None:
        embedder = dspy.Embedder("openai/text-embedding-3-small", caching=True, batch_size=200)
        # Use a smaller corpus for speed in tests
        _openai_retriever = dspy.Embeddings(embedder=embedder, corpus=get_chunks()[:100], k=5)
    return _openai_retriever


def setup_lm():
    lm = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7, max_tokens=1500)
    dspy.configure(lm=lm)
    return lm


# ============================================================
# EXAMPLE 1: load_codebase — loads files from a repo
# ============================================================
def test_load_codebase():
    """Test loading source files from the DSPy repo."""
    print("TEST: load_codebase...")

    files = get_files()
    assert len(files) > 50, f"Expected >50 files, got {len(files)}"
    print(f"  Loaded {len(files)} files")

    # Each file has path, content, language
    for f in files[:3]:
        assert 'path' in f
        assert 'content' in f
        assert 'language' in f
        assert f['language'] == 'py'
        print(f"    {f['path']} ({len(f['content'].split(chr(10)))} lines)")

    # No files over max_file_lines
    for f in files:
        line_count = len(f['content'].split('\n'))
        assert line_count <= 500, f"{f['path']} has {line_count} lines"

    print("  PASSED")


# ============================================================
# EXAMPLE 2: chunk_code_files — splits files into chunks
# ============================================================
def test_chunk_code_files():
    """Test chunking source files into overlapping windows."""
    print("TEST: chunk_code_files...")

    chunks = get_chunks()
    assert len(chunks) > 100, f"Expected >100 chunks, got {len(chunks)}"
    print(f"  Created {len(chunks)} chunks from {len(get_files())} files")

    # Every chunk starts with "File: "
    for chunk in chunks[:10]:
        assert chunk.startswith("File: "), f"Chunk doesn't start with 'File: ': {chunk[:50]}"

    # Average chunk length is reasonable
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"  Avg chunk length: {avg_len:.0f} chars")
    assert 200 < avg_len < 5000

    # Small files should produce exactly one chunk
    small_file = {'path': 'test.py', 'content': 'x = 1\ny = 2\nz = 3\n', 'language': 'py'}
    small_chunks = chunk_code_files([small_file], chunk_size=30, overlap=5)
    assert len(small_chunks) == 1
    assert small_chunks[0].startswith("File: test.py")
    print("  Small file → 1 chunk: OK")

    # Large files should produce multiple chunks
    large_file = {'path': 'big.py', 'content': '\n'.join([f'line_{i}' for i in range(100)]), 'language': 'py'}
    large_chunks = chunk_code_files([large_file], chunk_size=30, overlap=5)
    assert len(large_chunks) > 1
    print(f"  Large file (100 lines) → {len(large_chunks)} chunks: OK")

    print("  PASSED")


# ============================================================
# EXAMPLE 3: tfidf_embedder — consistent dimensions
# ============================================================
def test_tfidf_embedder():
    """Test the custom embedder function."""
    print("TEST: tfidf_embedder...")

    result = tfidf_embedder(["Hello world"])
    assert len(result) == 1
    assert len(result[0]) == 256
    print(f"  Single text: shape ({len(result)}, {len(result[0])})")

    result = tfidf_embedder(["Hello world", "Goodbye world", "DSPy is great"])
    assert len(result) == 3
    assert all(len(v) == 256 for v in result)
    print(f"  Multiple texts: shape ({len(result)}, {len(result[0])})")

    # Consistency: same input → same output
    r1 = tfidf_embedder(["test input"])
    r2 = tfidf_embedder(["test input"])
    assert np.allclose(r1[0], r2[0])
    print("  Consistency: PASSED")

    # Normalization: vectors have unit norm
    norm = np.linalg.norm(r1[0])
    assert abs(norm - 1.0) < 1e-5
    print(f"  Normalization: norm = {norm:.5f}")

    print("  PASSED")


# ============================================================
# EXAMPLE 4: dspy.Embeddings — basic retrieval (tfidf)
# ============================================================
def test_embeddings_retrieval():
    """Test dspy.Embeddings with real codebase chunks."""
    print("TEST: dspy.Embeddings with codebase (tfidf)...")

    retriever = get_tfidf_retriever()
    result = retriever("How does the Signature class define fields?")

    assert hasattr(result, 'passages')
    assert hasattr(result, 'indices')
    assert len(result.passages) == 5
    assert all(isinstance(p, str) for p in result.passages)
    assert all(p.startswith("File: ") for p in result.passages)

    print(f"  Retrieved {len(result.passages)} code chunks")
    for i, p in enumerate(result.passages):
        first_line = p.split('\n')[0]
        print(f"    {i+1}. {first_line}")
    print("  PASSED")


# ============================================================
# EXAMPLE 5: OpenAI Embedder — hosted model retrieval
# ============================================================
def test_openai_embedder():
    """Test dspy.Embedder with OpenAI text-embedding-3-small."""
    print("TEST: OpenAI text-embedding-3-small embedder...")

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIPPED: No OPENAI_API_KEY set")
        return

    # Test the embedder directly
    embedder = dspy.Embedder("openai/text-embedding-3-small", caching=True, batch_size=200)
    vecs = embedder(["Hello world", "DSPy retrieval"])
    assert isinstance(vecs, np.ndarray), f"Expected ndarray, got {type(vecs)}"
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 100  # text-embedding-3-small produces 1536-dim vectors
    print(f"  Embedder returned: shape {vecs.shape}")

    # Test retrieval with OpenAI embedder
    retriever = get_openai_retriever()
    result = retriever("How does the Signature class define fields?")

    assert hasattr(result, 'passages')
    assert len(result.passages) == 5
    assert all(p.startswith("File: ") for p in result.passages)

    print(f"  Retrieved {len(result.passages)} code chunks")
    for i, p in enumerate(result.passages):
        first_line = p.split('\n')[0]
        print(f"    {i+1}. {first_line}")
    print("  PASSED")


# ============================================================
# EXAMPLE 6: dspy.Embedder wrapper
# ============================================================
def test_embedder_wrapper():
    """Test dspy.Embedder wrapping custom function."""
    print("TEST: dspy.Embedder wrapper...")

    embedder = dspy.Embedder(tfidf_embedder)
    vecs = embedder(["hello world", "goodbye world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 256)
    print(f"  Embedder returned: shape {vecs.shape}")

    # Use with Embeddings (small corpus for speed)
    small_corpus = get_chunks()[:20]
    retriever = dspy.Embeddings(
        embedder=dspy.Embedder(tfidf_embedder),
        corpus=small_corpus,
        k=3,
    )
    result = retriever("What is a module?")
    assert len(result.passages) == 3
    print(f"  Retriever with Embedder wrapper: {len(result.passages)} results")
    print("  PASSED")


# ============================================================
# EXAMPLE 7: Pydantic CodeAnswer model
# ============================================================
def test_pydantic_models():
    """Test the Pydantic models used in the chapter."""
    print("TEST: Pydantic CodeAnswer model...")

    ca = CodeAnswer(
        answer="The Embeddings class handles vector retrieval.",
        confidence="high",
        cited_files=["retrievers/embeddings.py"],
        key_snippets=["class Embeddings:"],
    )
    assert ca.confidence == "high"
    assert len(ca.cited_files) == 1
    assert len(ca.key_snippets) == 1
    print(f"  CodeAnswer: confidence={ca.confidence}, files={ca.cited_files}")

    # Invalid confidence should fail
    try:
        CodeAnswer(
            answer="test",
            confidence="very_high",
            cited_files=["test.py"],
            key_snippets=["x = 1"],
        )
        print("  FAILED: Should have rejected 'very_high'")
        return False
    except ValidationError:
        print("  Literal validation: rejects invalid values")

    print("  PASSED")


# ============================================================
# EXAMPLE 8: Signature fields
# ============================================================
def test_signature_fields():
    """Verify AnswerFromCode signature has correct fields."""
    print("TEST: AnswerFromCode signature fields...")

    assert "context" in AnswerFromCode.input_fields
    assert "question" in AnswerFromCode.input_fields
    assert "response" in AnswerFromCode.output_fields
    print("  AnswerFromCode: context, question → response")
    print("  PASSED")


# ============================================================
# EXAMPLE 9: CodebaseQA — simple RAG with live LM
# ============================================================
def test_codebase_qa():
    """Test the simple RAG module with live API."""
    print("TEST: CodebaseQA simple RAG...")

    setup_lm()
    qa = CodebaseQA(retriever=get_tfidf_retriever())

    result = qa(question="What modules are available in the retrievers package?")
    assert result.answer is not None
    assert len(result.answer) > 20
    print(f"  Answer: {result.answer[:150]}...")
    print("  PASSED")


# ============================================================
# EXAMPLE 10: CodebaseExplorer — full citations with live LM
# ============================================================
def test_codebase_explorer():
    """Test the full explorer with citations."""
    print("TEST: CodebaseExplorer with citations...")

    setup_lm()
    explorer = CodebaseExplorer(retriever=get_tfidf_retriever())

    result = explorer(question="How does dspy.ChainOfThought extend Predict?")

    assert isinstance(result.response, CodeAnswer)
    assert len(result.response.answer) > 20
    assert result.response.confidence in ["high", "medium", "low"]
    assert isinstance(result.response.cited_files, list)
    assert isinstance(result.response.key_snippets, list)
    assert result.retrieved_passages is not None
    assert len(result.retrieved_passages) == 5

    print(f"  Answer: {result.response.answer[:120]}...")
    print(f"  Confidence: {result.response.confidence}")
    print(f"  Cited files: {result.response.cited_files}")
    print(f"  Key snippets: {len(result.response.key_snippets)}")
    print("  PASSED")


# ============================================================
# EXAMPLE 11: CodebaseExplorer with OpenAI embedder
# ============================================================
def test_codebase_explorer_openai():
    """Test the full explorer using OpenAI embeddings."""
    print("TEST: CodebaseExplorer with OpenAI embedder...")

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIPPED: No OPENAI_API_KEY set")
        return

    setup_lm()
    explorer = CodebaseExplorer(retriever=get_openai_retriever())

    result = explorer(question="How does dspy.ChainOfThought extend Predict?")

    assert isinstance(result.response, CodeAnswer)
    assert len(result.response.answer) > 20
    assert result.response.confidence in ["high", "medium", "low"]
    assert isinstance(result.response.cited_files, list)

    print(f"  Answer: {result.response.answer[:120]}...")
    print(f"  Confidence: {result.response.confidence}")
    print(f"  Cited files: {result.response.cited_files}")
    print("  PASSED")


# ============================================================
# EXAMPLE 12: answer_quality_metric
# ============================================================
def test_metric():
    """Test the metric function works correctly."""
    print("TEST: answer_quality_metric...")

    example = dspy.Example(
        question="What is DSPy?",
        answer="DSPy is a framework for programming language models"
    ).with_inputs("question")

    # Test with response.answer
    class MockResponse:
        answer = "DSPy is a great framework for programming language models and more"

    class MockPred:
        response = MockResponse()

    score = answer_quality_metric(example, MockPred())
    assert 0 < score <= 1.0
    print(f"  Score with response.answer: {score:.3f}")

    # Test with direct answer
    class MockDirect:
        answer = "DSPy is a great framework for programming language models and more"

    score2 = answer_quality_metric(example, MockDirect())
    assert 0 < score2 <= 1.0
    print(f"  Score with direct answer: {score2:.3f}")

    # Test with bad answer
    class MockBad:
        answer = "completely unrelated text about cooking recipes"

    score3 = answer_quality_metric(example, MockBad())
    assert score3 < score
    print(f"  Score with bad answer: {score3:.3f}")

    print("  PASSED")


# ============================================================
# EXAMPLE 13: dspy.Evaluate
# ============================================================
def test_evaluate():
    """Test dspy.Evaluate with our metric and dataset."""
    print("TEST: dspy.Evaluate...")

    setup_lm()
    qa = CodebaseQA(retriever=get_tfidf_retriever())

    evaluator = dspy.Evaluate(
        devset=CODEBASE_QA_DATASET[:2],  # 2 examples to save API calls
        metric=answer_quality_metric,
        num_threads=1,
        display_progress=False,
    )

    score = evaluator(qa)
    print(f"  CodebaseQA score (2 examples): {score}")
    print("  PASSED")


# ============================================================
# EXAMPLE 14: Save and Load retriever
# ============================================================
def test_save_load():
    """Test saving and loading an Embeddings index."""
    print("TEST: Save and Load retriever...")

    save_path = os.path.join(tempfile.gettempdir(), "test_codebase_index")

    # Build with small corpus for speed
    small_corpus = get_chunks()[:50]
    retriever = dspy.Embeddings(embedder=tfidf_embedder, corpus=small_corpus, k=3)
    retriever.save(save_path)
    assert os.path.exists(os.path.join(save_path, "config.json"))
    assert os.path.exists(os.path.join(save_path, "corpus_embeddings.npy"))
    print("  Saved index to disk")

    # Load
    loaded = dspy.Embeddings.from_saved(save_path, embedder=tfidf_embedder)
    result = loaded("What is a Signature?")
    assert len(result.passages) == 3
    assert all(isinstance(p, str) for p in result.passages)
    assert all(len(p) > 10 for p in result.passages)
    print(f"  Loaded retriever found: {result.passages[0].split(chr(10))[0]}")

    # Clean up
    try:
        shutil.rmtree(save_path)
        print("  Cleaned up test files")
    except PermissionError:
        print("  (cleanup skipped — permission restricted)")
    print("  PASSED")


# ============================================================
# EXAMPLE 15: get_embedder — the smart default
# ============================================================
def test_get_embedder():
    """Test the get_embedder() function picks the right backend."""
    print("TEST: get_embedder()...")

    embedder = get_embedder()
    assert embedder is not None
    print(f"  get_embedder() returned: {type(embedder).__name__}")

    # It should work as an embedder
    vecs = embedder(["test embedding"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 1
    assert vecs.shape[1] > 0
    print(f"  Embedding shape: {vecs.shape}")

    if os.getenv("OPENAI_API_KEY"):
        print("  Using: OpenAI text-embedding-3-small (hosted)")
    else:
        print("  Using: tfidf_embedder (local fallback)")

    print("  PASSED")


# ============================================================
# MAIN
# ============================================================
def main():
    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set. Most tests require live API.")
        sys.exit(1)

    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: DSPy source not found at {REPO_PATH}")
        print("Clone the DSPy repo or set DSPY_REPO_PATH.")
        sys.exit(1)

    print("=" * 60)
    print("Chapter 3: EVERY Code Example Tested")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print(f"Repo path: {os.path.abspath(REPO_PATH)}")
    print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print()

    tests = [
        test_load_codebase,
        test_chunk_code_files,
        test_tfidf_embedder,
        test_embeddings_retrieval,
        test_openai_embedder,
        test_embedder_wrapper,
        test_pydantic_models,
        test_signature_fields,
        test_metric,
        test_get_embedder,
        test_codebase_qa,
        test_codebase_explorer,
        test_codebase_explorer_openai,
        test_evaluate,
        test_save_load,
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
