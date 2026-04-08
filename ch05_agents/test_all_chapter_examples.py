"""
test_all_chapter_examples.py — Chapter 5: EVERY code block tested
DSPy: The Mostly Harmless Guide

Tests every code example from the chapter markdown.
Two real agents: Fact-Checker (web search) and Code Reviewer (GitHub API).
"""

import os
import sys
import time
import tempfile
import shutil

from dotenv import load_dotenv
load_dotenv()

import dspy
from typing import Literal
from pydantic import BaseModel, Field

from research_agent import (
    # Fact-Checker tools
    search_web, fetch_webpage, get_wikipedia_summary,
    # Fact-Checker types & agent
    SourceEvidence, FactCheckVerdict, CheckFact, FactChecker,
    # Code Reviewer tools
    get_repo_info, list_repo_files, read_github_file,
    get_recent_pull_requests, get_pr_diff,
    # Code Reviewer types & agent
    ReviewItem, CodeReviewReport, ReviewCode, CodeReviewer,
    # Quality wrappers
    QuickAnswer, fact_check_reward, review_quality_reward, answer_quality_reward,
)

API_KEY = os.getenv("ANTHROPIC_API_KEY")


def setup_lm():
    lm = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7, max_tokens=4000)
    dspy.configure(lm=lm)
    return lm


def network_available():
    """Check if we can reach external APIs."""
    import requests
    try:
        requests.get("https://api.github.com", timeout=5)
        return True
    except Exception:
        return False


HAS_NETWORK = network_available()


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL TESTS (no network or API key needed)
# ═══════════════════════════════════════════════════════════════════════════

def test_dspy_tool_creation():
    """Verify dspy.Tool wraps our functions correctly."""
    print("TEST 1: dspy.Tool wrapping...")

    tool = dspy.Tool(search_web)
    assert tool.name == "search_web", f"Expected 'search_web', got '{tool.name}'"
    assert "query" in tool.args, "Should have 'query' argument"
    assert tool.desc is not None and len(tool.desc) > 10, "Should have a description"

    # Tool with custom name
    custom = dspy.Tool(fetch_webpage, name="page_reader", desc="Reads a page")
    assert custom.name == "page_reader"
    assert custom.desc == "Reads a page"

    # GitHub tools
    gh_tool = dspy.Tool(get_repo_info)
    assert gh_tool.name == "get_repo_info"
    assert "owner_and_repo" in gh_tool.args

    print("  PASSED: dspy.Tool wrapping works")


def test_tool_arg_inference():
    """Verify dspy.Tool infers argument types from type hints."""
    print("TEST 2: Tool argument inference...")

    def calculate(name: str, points: int, bonus: float = 0.0) -> str:
        """Calculate a score."""
        return f"{name}: {points + bonus}"

    tool = dspy.Tool(calculate)
    assert "name" in tool.args
    assert "points" in tool.args
    assert "bonus" in tool.args

    result = tool(name="Alice", points=100, bonus=5.5)
    assert "Alice" in result and "105.5" in result

    print("  PASSED: Tool arg inference works")


def test_pydantic_types():
    """Verify Pydantic output models work correctly."""
    print("TEST 3: Pydantic output types...")

    # Fact-checker types
    evidence = SourceEvidence(
        source_name="Wikipedia",
        quote_or_finding="Python is widely used",
        supports_claim="supports",
    )
    assert evidence.supports_claim == "supports"

    verdict = FactCheckVerdict(
        claim="Test claim",
        verdict="true",
        confidence="high",
        evidence=[evidence],
        explanation="The claim is supported by multiple sources.",
    )
    assert verdict.verdict == "true"
    assert len(verdict.evidence) == 1

    # Code reviewer types
    item = ReviewItem(
        category="bug",
        severity="critical",
        file_path="main.py",
        description="Null pointer",
        suggestion="Add null check",
    )
    assert item.category == "bug"

    report = CodeReviewReport(
        repo_name="test/repo",
        overall_assessment="The code is well structured with minor issues.",
        review_items=[item],
        strengths=["Good test coverage"],
        verdict="approve",
    )
    assert report.verdict == "approve"

    print("  PASSED: Pydantic types work correctly")


def test_signatures():
    """Verify signature definitions have correct fields."""
    print("TEST 4: Signature definitions...")

    assert "claim" in CheckFact.input_fields
    assert "verdict" in CheckFact.output_fields

    assert "repo" in ReviewCode.input_fields
    assert "focus" in ReviewCode.input_fields
    assert "review" in ReviewCode.output_fields

    assert "question" in QuickAnswer.input_fields
    assert "answer" in QuickAnswer.output_fields

    print("  PASSED: Signatures defined correctly")


def test_fact_checker_construction():
    """Verify FactChecker agent is constructed correctly."""
    print("TEST 5: FactChecker construction...")

    checker = FactChecker(max_iters=5)
    assert hasattr(checker, "react")
    tools = checker.react.tools
    assert "search_web" in tools, "Should have search_web tool"
    assert "fetch_webpage" in tools, "Should have fetch_webpage tool"
    assert "get_wikipedia_summary" in tools, "Should have get_wikipedia_summary tool"
    assert "finish" in tools, "Should have auto-added 'finish' tool"

    print("  PASSED: FactChecker constructed correctly")


def test_code_reviewer_construction():
    """Verify CodeReviewer agent is constructed correctly."""
    print("TEST 6: CodeReviewer construction...")

    reviewer = CodeReviewer(max_iters=7)
    assert hasattr(reviewer, "react")
    tools = reviewer.react.tools
    assert "get_repo_info" in tools
    assert "list_repo_files" in tools
    assert "read_github_file" in tools
    assert "get_recent_pull_requests" in tools
    assert "get_pr_diff" in tools
    assert "finish" in tools

    print("  PASSED: CodeReviewer constructed correctly")


def test_reward_functions():
    """Verify reward functions score correctly."""
    print("TEST 7: Reward functions...")

    # fact_check_reward
    class FP:
        def __init__(self, v):
            self.verdict = v

    good_verdict = FactCheckVerdict(
        claim="X", verdict="true", confidence="high",
        evidence=[
            SourceEvidence(source_name="A", quote_or_finding="yes", supports_claim="supports"),
            SourceEvidence(source_name="B", quote_or_finding="yes", supports_claim="supports"),
        ],
        explanation="The claim is well supported by multiple independent sources that consistently confirm the core statement through rigorous evidence and analysis.",
    )
    score = fact_check_reward({}, FP(good_verdict))
    assert score >= 0.8, f"Good verdict should score >= 0.8, got {score}"

    bad_verdict = FactCheckVerdict(
        claim="X", verdict="unverifiable", confidence="low",
        evidence=[], explanation="Dunno.",
    )
    low_score = fact_check_reward({}, FP(bad_verdict))
    assert low_score < 0.3, f"Bad verdict should score < 0.3, got {low_score}"

    # review_quality_reward
    class RP:
        def __init__(self, r):
            self.review = r

    good_review = CodeReviewReport(
        repo_name="x/y",
        overall_assessment="The codebase is well structured with clear separation of concerns and good test coverage throughout.",
        review_items=[
            ReviewItem(category="bug", severity="warning", file_path="a.py",
                       description="Issue", suggestion="Fix"),
            ReviewItem(category="style", severity="suggestion", file_path="b.py",
                       description="Style", suggestion="Refactor"),
            ReviewItem(category="readability", severity="suggestion", file_path="c.py",
                       description="Naming", suggestion="Rename"),
        ],
        strengths=["Good tests"],
        verdict="approve",
    )
    r_score = review_quality_reward({}, RP(good_review))
    assert r_score >= 0.8, f"Good review should score >= 0.8, got {r_score}"

    # answer_quality_reward
    class AP:
        def __init__(self, a):
            self.answer = a

    assert answer_quality_reward({}, AP("Short")) < 0.3
    medium_answer = "This is a detailed answer with substance " * 4  # ~20 words
    assert answer_quality_reward({}, AP(medium_answer)) >= 0.4
    long_answer = "This is a comprehensive detailed answer with lots of substance " * 7  # 56 words
    assert answer_quality_reward({}, AP(long_answer)) >= 1.0

    print(f"  Fact-check score: {score}, Code review score: {r_score}")
    print("  PASSED: Reward functions score correctly")


def test_save_load_agents():
    """Verify we can save and load both agents."""
    print("TEST 8: Save/Load agents...")

    tmp_dir = tempfile.mkdtemp()
    try:
        # FactChecker
        checker = FactChecker(max_iters=5)
        path1 = os.path.join(tmp_dir, "fact_checker.json")
        checker.save(path1)
        assert os.path.exists(path1)
        loaded = FactChecker(max_iters=5)
        loaded.load(path1)
        assert "search_web" in loaded.react.tools

        # CodeReviewer
        reviewer = CodeReviewer(max_iters=7)
        path2 = os.path.join(tmp_dir, "code_reviewer.json")
        reviewer.save(path2)
        assert os.path.exists(path2)
        loaded2 = CodeReviewer(max_iters=7)
        loaded2.load(path2)
        assert "get_repo_info" in loaded2.react.tools

        print("  PASSED: Save/Load works for both agents")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL TESTS (need network, no API key)
# ═══════════════════════════════════════════════════════════════════════════

def test_search_web_live():
    """Verify search_web returns real results."""
    if not HAS_NETWORK:
        print("TEST 9: search_web live... SKIPPED (no network)")
        return
    print("TEST 9: search_web live...")

    results = search_web("Python programming language")
    assert "Search failed" not in results, f"Search failed: {results}"
    assert len(results) > 50, "Should return substantive results"

    print(f"  Got {len(results)} chars of search results")
    print("  PASSED: search_web returns real results")


def test_fetch_webpage_live():
    """Verify fetch_webpage extracts text from a real URL."""
    if not HAS_NETWORK:
        print("TEST 10: fetch_webpage live... SKIPPED (no network)")
        return
    print("TEST 10: fetch_webpage live...")

    content = fetch_webpage("https://en.wikipedia.org/wiki/Python_(programming_language)")
    assert "Failed to fetch" not in content, f"Fetch failed: {content}"
    assert "Python" in content, "Should contain Python content"
    assert len(content) > 100, "Should have substantial content"

    print(f"  Got {len(content)} chars from Wikipedia page")
    print("  PASSED: fetch_webpage extracts real content")


def test_wikipedia_summary_live():
    """Verify get_wikipedia_summary returns real data."""
    if not HAS_NETWORK:
        print("TEST 11: get_wikipedia_summary live... SKIPPED (no network)")
        return
    print("TEST 11: get_wikipedia_summary live...")

    summary = get_wikipedia_summary("Python (programming language)")
    assert "lookup failed" not in summary, f"Lookup failed: {summary}"
    assert "Python" in summary
    assert len(summary) > 100

    print(f"  Got {len(summary)} chars summary")
    print("  PASSED: Wikipedia summary works")


def test_github_repo_info_live():
    """Verify get_repo_info returns real GitHub data."""
    if not HAS_NETWORK:
        print("TEST 12: get_repo_info live... SKIPPED (no network)")
        return
    print("TEST 12: get_repo_info live...")

    info = get_repo_info("stanfordnlp/dspy")
    assert "Failed" not in info, f"Failed: {info}"
    assert "stanfordnlp/dspy" in info
    assert "Stars:" in info

    print(f"  {info.splitlines()[0]}")
    print("  PASSED: GitHub repo info works")


def test_github_list_files_live():
    """Verify list_repo_files returns real directory listings."""
    if not HAS_NETWORK:
        print("TEST 13: list_repo_files live... SKIPPED (no network)")
        return
    print("TEST 13: list_repo_files live...")

    listing = list_repo_files("stanfordnlp/dspy", "dspy/predict")
    assert "Failed" not in listing, f"Failed: {listing}"
    assert "react.py" in listing, "Should list react.py"

    print(f"  Got {len(listing.splitlines())} entries")
    print("  PASSED: GitHub file listing works")


def test_github_read_file_live():
    """Verify read_github_file returns real file content."""
    if not HAS_NETWORK:
        print("TEST 14: read_github_file live... SKIPPED (no network)")
        return
    print("TEST 14: read_github_file live...")

    content = read_github_file("stanfordnlp/dspy", "dspy/predict/react.py")
    assert "Failed" not in content, f"Failed: {content}"
    assert "ReAct" in content, "Should contain ReAct class"

    print(f"  Got {len(content)} chars from react.py")
    print("  PASSED: GitHub file reading works")


def test_github_prs_live():
    """Verify get_recent_pull_requests returns real PRs."""
    if not HAS_NETWORK:
        print("TEST 15: get_recent_pull_requests live... SKIPPED (no network)")
        return
    print("TEST 15: get_recent_pull_requests live...")

    prs = get_recent_pull_requests("stanfordnlp/dspy")
    assert "Failed" not in prs, f"Failed: {prs}"
    assert "PR #" in prs, "Should list PR numbers"

    print(f"  Got {len(prs.splitlines())} lines of PR data")
    print("  PASSED: GitHub PR listing works")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE AGENT TESTS (need network + API key)
# ═══════════════════════════════════════════════════════════════════════════

def test_fact_checker_live():
    """Run the Fact-Checker agent with real web search and Claude."""
    if not API_KEY:
        print("TEST 16: FactChecker live... SKIPPED (no ANTHROPIC_API_KEY)")
        return
    if not HAS_NETWORK:
        print("TEST 16: FactChecker live... SKIPPED (no network)")
        return
    print("TEST 16: FactChecker live run (waiting 60s for rate limit)...")
    time.sleep(60)
    setup_lm()

    checker = FactChecker(max_iters=8)
    result = checker(claim="Python is the most popular programming language in 2024")

    v = result.verdict
    assert isinstance(v, FactCheckVerdict), f"Expected FactCheckVerdict, got {type(v)}"
    assert v.verdict in ("true", "mostly true", "mixed", "mostly false", "false", "unverifiable")
    assert len(v.evidence) >= 1, "Should have at least 1 source"
    assert len(v.explanation.split()) >= 5, "Explanation should be substantive"

    trajectory = result.trajectory
    assert len(trajectory) > 0, "Agent should have a trajectory"

    print(f"  Verdict: {v.verdict} (confidence: {v.confidence})")
    print(f"  Evidence sources: {len(v.evidence)}")
    print(f"  Trajectory steps: {len(trajectory)}")
    print("  PASSED: FactChecker produced valid verdict")


def test_code_reviewer_live():
    """Run the Code Reviewer agent with real GitHub API and Claude."""
    if not API_KEY:
        print("TEST 17: CodeReviewer live... SKIPPED (no ANTHROPIC_API_KEY)")
        return
    if not HAS_NETWORK:
        print("TEST 17: CodeReviewer live... SKIPPED (no network)")
        return
    print("TEST 17: CodeReviewer live run (waiting 60s for rate limit)...")
    time.sleep(60)  # Respect Anthropic rate limits between heavy agent runs
    setup_lm()

    reviewer = CodeReviewer(max_iters=10)
    result = reviewer(
        repo="stanfordnlp/dspy",
        focus="the predict/ module architecture",
    )

    r = result.review
    assert isinstance(r, CodeReviewReport), f"Expected CodeReviewReport, got {type(r)}"
    assert len(r.review_items) >= 1, "Should have at least 1 review item"
    assert len(r.strengths) >= 1, "Should identify at least 1 strength"
    assert r.verdict in ("approve", "request changes", "needs discussion")

    trajectory = result.trajectory
    assert len(trajectory) > 0

    print(f"  Repo: {r.repo_name}")
    print(f"  Verdict: {r.verdict}")
    print(f"  Review items: {len(r.review_items)}")
    print(f"  Trajectory steps: {len(trajectory)}")
    print("  PASSED: CodeReviewer produced valid review")


def test_best_of_n_live():
    """Test BestOfN with a live API call."""
    if not API_KEY:
        print("TEST 18: BestOfN... SKIPPED (no ANTHROPIC_API_KEY)")
        return
    print("TEST 18: BestOfN live run (waiting 30s for rate limit)...")
    time.sleep(30)
    setup_lm()

    qa = dspy.ChainOfThought(QuickAnswer)
    best_of_3 = dspy.BestOfN(
        module=qa,
        N=3,
        reward_fn=answer_quality_reward,
        threshold=0.7,
    )

    result = best_of_3(question="What makes DSPy different from LangChain?")
    assert result.answer is not None
    assert len(result.answer.split()) >= 5

    print(f"  Answer length: {len(result.answer.split())} words")
    print("  PASSED: BestOfN produced quality answer")


def test_refine_live():
    """Test Refine with a live API call."""
    if not API_KEY:
        print("TEST 19: Refine... SKIPPED (no ANTHROPIC_API_KEY)")
        return
    print("TEST 19: Refine live run (waiting 30s for rate limit)...")
    time.sleep(30)
    setup_lm()

    refined_qa = dspy.Refine(
        module=dspy.ChainOfThought(QuickAnswer),
        N=3,
        reward_fn=answer_quality_reward,
        threshold=1.0,
    )

    result = refined_qa(question="How does ReAct improve over basic chain-of-thought?")
    assert result.answer is not None
    assert len(result.answer.split()) >= 5

    print(f"  Answer length: {len(result.answer.split())} words")
    print("  PASSED: Refine produced quality answer")


def test_react_inline_signature_live():
    """Test ReAct with a simple inline signature and web tools."""
    if not API_KEY:
        print("TEST 20: ReAct inline... SKIPPED (no ANTHROPIC_API_KEY)")
        return
    if not HAS_NETWORK:
        print("TEST 20: ReAct inline... SKIPPED (no network)")
        return
    print("TEST 20: ReAct with inline signature (waiting 30s for rate limit)...")
    time.sleep(30)
    setup_lm()

    agent = dspy.ReAct(
        "question -> answer",
        tools=[search_web, get_wikipedia_summary],
        max_iters=5,
    )
    result = agent(question="Who created the Python programming language?")
    assert result.answer is not None
    assert len(result.answer) > 10

    print(f"  Answer: {result.answer[:100]}...")
    print("  PASSED: Inline ReAct works with real tools")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Chapter 5 Tests: So Long, and Thanks for All the Prompts")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print(f"  Network: {'available' if HAS_NETWORK else 'unavailable'}")
    print(f"  API Key: {'set' if API_KEY else 'not set'}")
    print()

    tests = [
        # Structural (always run)
        test_dspy_tool_creation,
        test_tool_arg_inference,
        test_pydantic_types,
        test_signatures,
        test_fact_checker_construction,
        test_code_reviewer_construction,
        test_reward_functions,
        test_save_load_agents,
        # Tool tests (need network)
        test_search_web_live,
        test_fetch_webpage_live,
        test_wikipedia_summary_live,
        test_github_repo_info_live,
        test_github_list_files_live,
        test_github_read_file_live,
        test_github_prs_live,
        # Agent tests (need network + API key)
        test_fact_checker_live,
        test_code_reviewer_live,
        test_best_of_n_live,
        test_refine_live,
        test_react_inline_signature_live,
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
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
