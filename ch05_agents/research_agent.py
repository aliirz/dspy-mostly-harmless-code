"""
research_agent.py — Chapter 5: So Long, and Thanks for All the Prompts
DSPy: The Mostly Harmless Guide

Two real-world agents:
  1. FactChecker  — searches the web, reads pages, cross-references, delivers a verdict
  2. CodeReviewer — hits the GitHub API, reads source files, reviews code quality

Demonstrates: dspy.Tool, dspy.ReAct, dspy.BestOfN, dspy.Refine
"""

import os
import json
import textwrap
from typing import Literal

import dspy
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — FACT-CHECKER AGENT
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Tools: Web Search & Page Reading
# ---------------------------------------------------------------------------

def search_web(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top results with titles,
    URLs, and snippets. Use this to find relevant sources for a claim."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(f"- {r['title']}\n  URL: {r['href']}\n  {r['body']}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search failed: {e}"


def fetch_webpage(url: str) -> str:
    """Fetch a webpage and extract its main text content. Returns up to
    3000 characters of readable text. Use this to read a full article."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (DSPy Research Agent)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > 3000:
            return text[:3000] + f"\n\n... [truncated, {len(text)} total chars]"
        return text
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def get_wikipedia_summary(topic: str) -> str:
    """Get a Wikipedia summary for a topic. Returns the first few paragraphs.
    Use this for quick factual background on well-known topics."""
    try:
        params = {
            "action": "query",
            "titles": topic,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
            "redirects": 1,
        }
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                if len(extract) > 2000:
                    return extract[:2000] + "..."
                return extract
        return f"No Wikipedia article found for '{topic}'."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


# ---------------------------------------------------------------------------
# Structured Output: Fact-Check Verdict
# ---------------------------------------------------------------------------

class SourceEvidence(BaseModel):
    """Evidence from a single source."""
    source_name: str = Field(description="Name or URL of the source")
    quote_or_finding: str = Field(description="Relevant quote or finding from this source")
    supports_claim: Literal["supports", "contradicts", "neutral"] = Field(
        description="Whether this source supports, contradicts, or is neutral on the claim"
    )


class FactCheckVerdict(BaseModel):
    """A structured fact-check verdict."""
    claim: str = Field(description="The original claim being checked")
    verdict: Literal["true", "mostly true", "mixed", "mostly false", "false", "unverifiable"] = Field(
        description="The overall verdict"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the verdict based on source quality"
    )
    evidence: list[SourceEvidence] = Field(
        description="Evidence gathered from multiple sources"
    )
    explanation: str = Field(
        description="Clear explanation of the verdict (2-3 sentences)"
    )


# ---------------------------------------------------------------------------
# Fact-Checker Signature & Agent
# ---------------------------------------------------------------------------

class CheckFact(dspy.Signature):
    """You are a rigorous fact-checker. Given a claim, search the web for
    evidence from multiple sources. Read relevant pages to verify details.
    Cross-reference findings and deliver a structured verdict with evidence."""

    claim: str = dspy.InputField(desc="A factual claim to verify")
    verdict: FactCheckVerdict = dspy.OutputField(desc="Structured fact-check verdict")


class FactChecker(dspy.Module):
    """An agent that verifies claims by searching the web and reading sources."""

    def __init__(self, max_iters=8):
        super().__init__()
        self.react = dspy.ReAct(
            CheckFact,
            tools=[search_web, fetch_webpage, get_wikipedia_summary],
            max_iters=max_iters,
        )

    def forward(self, claim: str) -> dspy.Prediction:
        return self.react(claim=claim)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — CODE REVIEWER AGENT
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Tools: GitHub API
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
GITHUB_HEADERS = {"Accept": "application/vnd.github.v3+json"}
# Optional: set GITHUB_TOKEN env var for higher rate limits (60/hr without)
if os.getenv("GITHUB_TOKEN"):
    GITHUB_HEADERS["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"


def get_repo_info(owner_and_repo: str) -> str:
    """Get basic info about a GitHub repository. Pass owner/repo format,
    e.g. 'stanfordnlp/dspy'. Returns stars, description, language, and recent activity."""
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{owner_and_repo}",
            headers=GITHUB_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()
        return (
            f"Repository: {d['full_name']}\n"
            f"Description: {d.get('description', 'N/A')}\n"
            f"Language: {d.get('language', 'N/A')}\n"
            f"Stars: {d['stargazers_count']} | Forks: {d['forks_count']}\n"
            f"Open Issues: {d['open_issues_count']}\n"
            f"Last Push: {d['pushed_at']}\n"
            f"Default Branch: {d['default_branch']}"
        )
    except Exception as e:
        return f"Failed to get repo info: {e}"


def list_repo_files(owner_and_repo: str, path: str = "") -> str:
    """List files and directories in a GitHub repository path.
    Pass owner/repo format (e.g. 'stanfordnlp/dspy') and an optional path
    (e.g. 'dspy/predict'). Returns file names, types, and sizes."""
    try:
        url = f"{GITHUB_API}/repos/{owner_and_repo}/contents/{path}"
        resp = requests.get(url, headers=GITHUB_HEADERS, timeout=10)
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            return f"Path '{path}' is a file, not a directory. Use read_github_file instead."
        output = []
        for item in items[:30]:  # Limit to 30 entries
            kind = "dir" if item["type"] == "dir" else f"file ({item.get('size', 0)} bytes)"
            output.append(f"  {item['name']} [{kind}]")
        header = f"Contents of {owner_and_repo}/{path or '(root)'}:"
        return header + "\n" + "\n".join(output)
    except Exception as e:
        return f"Failed to list files: {e}"


def read_github_file(owner_and_repo: str, filepath: str) -> str:
    """Read the raw content of a file from a GitHub repository.
    Pass owner/repo (e.g. 'stanfordnlp/dspy') and filepath
    (e.g. 'dspy/predict/react.py'). Returns the file content (max 5000 chars)."""
    try:
        url = f"https://raw.githubusercontent.com/{owner_and_repo}/main/{filepath}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            # Try master branch
            url = f"https://raw.githubusercontent.com/{owner_and_repo}/master/{filepath}"
            resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        content = resp.text
        if len(content) > 5000:
            return content[:5000] + f"\n\n... [truncated, {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Failed to read file: {e}"


def get_recent_pull_requests(owner_and_repo: str) -> str:
    """Get the 5 most recent pull requests for a repository.
    Pass owner/repo format (e.g. 'stanfordnlp/dspy').
    Returns PR titles, authors, states, and URLs."""
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{owner_and_repo}/pulls",
            headers=GITHUB_HEADERS,
            params={"state": "all", "per_page": 5, "sort": "updated"},
            timeout=10,
        )
        resp.raise_for_status()
        prs = resp.json()
        if not prs:
            return "No pull requests found."
        output = []
        for pr in prs:
            status = pr["state"]
            if pr.get("merged_at"):
                status = "merged"
            output.append(
                f"- PR #{pr['number']}: {pr['title']}\n"
                f"  Author: {pr['user']['login']} | Status: {status}\n"
                f"  URL: {pr['html_url']}"
            )
        return "\n\n".join(output)
    except Exception as e:
        return f"Failed to get PRs: {e}"


def get_pr_diff(owner_and_repo: str, pr_number: int) -> str:
    """Get the diff/changes for a specific pull request.
    Pass owner/repo and the PR number. Returns the file changes and patch content."""
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{owner_and_repo}/pulls/{pr_number}/files",
            headers=GITHUB_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        files = resp.json()
        output = []
        total_chars = 0
        for f in files:
            patch = f.get("patch", "(binary or too large)")
            entry = (
                f"--- {f['filename']} ---\n"
                f"Status: {f['status']} | +{f['additions']} -{f['deletions']}\n"
                f"{patch}"
            )
            total_chars += len(entry)
            if total_chars > 6000:
                output.append(f"... [{len(files) - len(output)} more files truncated]")
                break
            output.append(entry)
        return "\n\n".join(output)
    except Exception as e:
        return f"Failed to get PR diff: {e}"


# ---------------------------------------------------------------------------
# Structured Output: Code Review
# ---------------------------------------------------------------------------

class ReviewItem(BaseModel):
    """A single code review observation."""
    category: Literal["bug", "style", "performance", "security", "readability", "architecture"] = Field(
        description="Category of the observation"
    )
    severity: Literal["critical", "warning", "suggestion"] = Field(
        description="How serious the issue is"
    )
    file_path: str = Field(description="The file this observation applies to")
    description: str = Field(description="Clear description of the issue or suggestion")
    suggestion: str = Field(description="Specific recommendation for improvement")


class CodeReviewReport(BaseModel):
    """A structured code review report."""
    repo_name: str = Field(description="The repository being reviewed")
    overall_assessment: str = Field(
        description="High-level assessment of the code quality (2-3 sentences)"
    )
    review_items: list[ReviewItem] = Field(
        description="Specific observations and suggestions"
    )
    strengths: list[str] = Field(description="Things the code does well")
    verdict: Literal["approve", "request changes", "needs discussion"] = Field(
        description="Overall review verdict"
    )


# ---------------------------------------------------------------------------
# Code Reviewer Signature & Agent
# ---------------------------------------------------------------------------

class ReviewCode(dspy.Signature):
    """You are an expert code reviewer. Given a GitHub repository (owner/repo
    format) and optionally a PR number, examine the code structure, read key
    files, and produce a thorough code review. Focus on bugs, architecture,
    readability, and actionable improvements."""

    repo: str = dspy.InputField(desc="GitHub repository in owner/repo format, e.g. 'stanfordnlp/dspy'")
    focus: str = dspy.InputField(desc="What to focus the review on, e.g. 'PR #123' or 'overall architecture'")
    review: CodeReviewReport = dspy.OutputField(desc="Structured code review report")


class CodeReviewer(dspy.Module):
    """An agent that reviews GitHub repositories and PRs."""

    def __init__(self, max_iters=10):
        super().__init__()
        self.react = dspy.ReAct(
            ReviewCode,
            tools=[
                get_repo_info,
                list_repo_files,
                read_github_file,
                get_recent_pull_requests,
                get_pr_diff,
            ],
            max_iters=max_iters,
        )

    def forward(self, repo: str, focus: str) -> dspy.Prediction:
        return self.react(repo=repo, focus=focus)


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — QUALITY WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════

class QuickAnswer(dspy.Signature):
    """Answer a question concisely and factually."""
    question: str = dspy.InputField(desc="A question to answer")
    answer: str = dspy.OutputField(desc="A concise, factual answer")


def fact_check_reward(args, prediction):
    """Reward function for the fact-checker. Checks verdict completeness."""
    try:
        v = prediction.verdict
        if not isinstance(v, FactCheckVerdict):
            return 0.0
        score = 0.0
        if len(v.evidence) >= 2:
            score += 0.4
        if len(v.explanation.split()) >= 15:
            score += 0.3
        if v.confidence in ("high", "medium"):
            score += 0.2
        if v.verdict != "unverifiable":
            score += 0.1
        return score
    except Exception:
        return 0.0


def review_quality_reward(args, prediction):
    """Reward function for the code reviewer. Checks review thoroughness."""
    try:
        r = prediction.review
        if not isinstance(r, CodeReviewReport):
            return 0.0
        score = 0.0
        if len(r.review_items) >= 3:
            score += 0.4
        if len(r.strengths) >= 1:
            score += 0.2
        if len(r.overall_assessment.split()) >= 15:
            score += 0.2
        if r.verdict in ("approve", "request changes", "needs discussion"):
            score += 0.2
        return score
    except Exception:
        return 0.0


def answer_quality_reward(args, prediction):
    """Reward function for BestOfN/Refine demos — checks answer substance."""
    try:
        answer = prediction.answer
        if not answer or len(answer.split()) < 10:
            return 0.0
        word_count = len(answer.split())
        if word_count >= 50:
            return 1.0
        elif word_count >= 30:
            return 0.7
        elif word_count >= 15:
            return 0.4
        return 0.2
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=4000,
    )
    dspy.configure(lm=lm)

    print("=" * 60)
    print("Chapter 5: Agents — Fact-Checker & Code Reviewer")
    print("=" * 60)

    # --- Fact-Checker Agent ---
    print("\n--- FACT-CHECKER AGENT ---")
    checker = FactChecker(max_iters=8)
    result = checker(claim="Python is the most popular programming language in 2024")
    v = result.verdict

    print(f"\nClaim: {v.claim}")
    print(f"Verdict: {v.verdict} (confidence: {v.confidence})")
    print(f"Evidence ({len(v.evidence)} sources):")
    for e in v.evidence:
        print(f"  [{e.supports_claim}] {e.source_name}: {e.quote_or_finding[:80]}...")
    print(f"Explanation: {v.explanation}")

    # --- Code Reviewer Agent ---
    print("\n\n--- CODE REVIEWER AGENT ---")
    reviewer = CodeReviewer(max_iters=10)
    result = reviewer(
        repo="stanfordnlp/dspy",
        focus="overall architecture of the predict/ module",
    )
    r = result.review

    print(f"\nRepo: {r.repo_name}")
    print(f"Assessment: {r.overall_assessment}")
    print(f"Verdict: {r.verdict}")
    print(f"Review items ({len(r.review_items)}):")
    for item in r.review_items:
        print(f"  [{item.severity}] {item.category}: {item.description[:60]}...")
    print(f"Strengths: {', '.join(r.strengths)}")

    # --- BestOfN ---
    print("\n\n--- BestOfN: Pick the Best of 3 ---")
    qa = dspy.ChainOfThought(QuickAnswer)
    best_of_3 = dspy.BestOfN(
        module=qa,
        N=3,
        reward_fn=answer_quality_reward,
        threshold=0.7,
    )
    result = best_of_3(question="What makes DSPy different from LangChain?")
    print(f"Answer: {result.answer}")

    # --- Refine ---
    print("\n\n--- Refine: Iterative Improvement ---")
    refined_qa = dspy.Refine(
        module=dspy.ChainOfThought(QuickAnswer),
        N=3,
        reward_fn=answer_quality_reward,
        threshold=1.0,
    )
    result = refined_qa(question="How does ReAct improve over basic chain-of-thought?")
    print(f"Answer: {result.answer}")

    print("\n" + "=" * 60)
    print("All agent demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
