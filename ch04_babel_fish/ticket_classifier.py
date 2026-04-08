"""
ticket_classifier.py — The Customer Support Ticket Classifier
Chapter 4: The Babel Fish — Optimizers Demystified
DSPy: The Mostly Harmless Guide

A ticket classification system built to demonstrate DSPy's optimizer
ecosystem. Build naively, then progressively optimize.

Run with: poetry run python ticket_classifier.py
"""

import os
import dspy
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


# ===================================================================
# Data
# ===================================================================

TICKET_CATEGORIES = ["billing", "technical", "account",
                     "feature_request", "general"]
PRIORITY_LEVELS = ["high", "medium", "low"]

TRAINSET = [
    dspy.Example(
        ticket="I've been charged twice for my subscription this month. "
        "Order #12847. Please refund the duplicate charge immediately.",
        category="billing", priority="high", routing="billing_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="The export to PDF feature keeps crashing when I try to "
        "export reports with more than 50 pages. Error code: EXP-4012.",
        category="technical", priority="high", routing="engineering_tier2",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="How do I change my email address? I can't find the option "
        "in account settings anywhere.",
        category="account", priority="low", routing="support_tier1",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="It would be amazing if you could add dark mode. "
        "My eyes are burning during late-night sessions!",
        category="feature_request", priority="low", routing="product_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="I need to upgrade from the starter plan to enterprise. "
        "We have 200 users now. What's the process?",
        category="billing", priority="medium", routing="sales_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="The API is returning 500 errors intermittently. "
        "Started about 2 hours ago. Affecting our production system.",
        category="technical", priority="high", routing="engineering_tier2",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Can you add my colleague jane@company.com to our "
        "team account? She needs editor access.",
        category="account", priority="medium", routing="support_tier1",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Would love to see Slack integration for notifications. "
        "Right now I have to check the dashboard manually.",
        category="feature_request", priority="low", routing="product_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="My trial expired yesterday but I wasn't done evaluating. "
        "Can you extend it by a week?",
        category="billing", priority="medium", routing="sales_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="SSO login stopped working after your update last night. "
        "Our entire team is locked out.",
        category="technical", priority="high", routing="engineering_tier2",
    ).with_inputs("ticket"),
]

VALSET = [
    dspy.Example(
        ticket="I was charged $49 but my plan is supposed to be $29. "
        "Can someone look into this?",
        category="billing", priority="high", routing="billing_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="The mobile app crashes every time I open the dashboard. "
        "iPhone 15, iOS 17.4.",
        category="technical", priority="high", routing="engineering_tier2",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="How do I enable two-factor authentication on my account?",
        category="account", priority="medium", routing="support_tier1",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Can you add support for webhooks? We need real-time "
        "event notifications for our integration.",
        category="feature_request", priority="medium", routing="product_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Just wanted to say your support team is fantastic. "
        "Sarah resolved my issue in 5 minutes!",
        category="general", priority="low", routing="support_tier1",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="I need to cancel my subscription effective end of month. "
        "Please confirm no further charges.",
        category="billing", priority="medium", routing="billing_team",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Search is really slow when I have more than 10,000 "
        "records. Takes 30+ seconds to return results.",
        category="technical", priority="medium", routing="engineering_tier2",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="I accidentally deleted my project. Is there any way "
        "to recover it? It had months of work.",
        category="account", priority="high", routing="support_tier1",
    ).with_inputs("ticket"),
]


# ===================================================================
# Structured Output
# ===================================================================

class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "account",
                       "feature_request", "general"] = Field(
        description="The primary category of the support ticket"
    )
    priority: Literal["high", "medium", "low"] = Field(
        description="Urgency level based on business impact"
    )
    routing: str = Field(
        description="Team or queue to route the ticket to"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decision"
    )


class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket into a category, priority level,
    and routing destination. Consider the urgency, topic, and business
    impact of the ticket."""
    ticket: str = dspy.InputField(desc="The customer support ticket text")
    classification: TicketClassification = dspy.OutputField(
        desc="Structured classification with category, priority, and routing"
    )


# ===================================================================
# Module
# ===================================================================

class TicketClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyTicket)

    def forward(self, ticket):
        result = self.classify(ticket=ticket)
        return dspy.Prediction(
            category=result.classification.category,
            priority=result.classification.priority,
            routing=result.classification.routing,
            reasoning=result.classification.reasoning,
        )


# ===================================================================
# Metric
# ===================================================================

def ticket_metric(example, pred, trace=None):
    """Score ticket classification accuracy.

    Scoring:
    - Category match:  0.5 points (primary task)
    - Priority match:  0.3 points (routing importance)
    - Routing match:   0.2 points (partial credit available)
    """
    score = 0.0

    if hasattr(pred, 'category') and pred.category == example.category:
        score += 0.5

    if hasattr(pred, 'priority') and pred.priority == example.priority:
        score += 0.3

    if hasattr(pred, 'routing'):
        if pred.routing == example.routing:
            score += 0.2
        elif example.routing.split('_')[0] in pred.routing:
            score += 0.1

    return score


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

    evaluator = dspy.Evaluate(
        devset=VALSET,
        metric=ticket_metric,
        num_threads=1,
        display_progress=True,
    )

    print("=" * 60)
    print("THE TICKET CLASSIFIER — OPTIMIZER SHOWDOWN")
    print("=" * 60)

    # --- Baseline ---
    print("\n--- Baseline (no optimization) ---")
    classifier = TicketClassifier()
    baseline = evaluator(classifier)
    print(f"Baseline score: {baseline}")

    # --- LabeledFewShot ---
    print("\n--- LabeledFewShot (k=3) ---")
    from dspy.teleprompt import LabeledFewShot
    classifier = TicketClassifier()
    opt = LabeledFewShot(k=3)
    optimized = opt.compile(student=classifier, trainset=TRAINSET)
    lfs_score = evaluator(optimized)
    print(f"LabeledFewShot score: {lfs_score}")

    # --- BootstrapFewShot ---
    print("\n--- BootstrapFewShot ---")
    from dspy.teleprompt import BootstrapFewShot
    classifier = TicketClassifier()
    opt = BootstrapFewShot(
        metric=ticket_metric,
        metric_threshold=0.8,
        max_bootstrapped_demos=4,
        max_labeled_demos=2,
        max_rounds=2,
    )
    optimized = opt.compile(student=classifier, trainset=TRAINSET)
    bfs_score = evaluator(optimized)
    print(f"BootstrapFewShot score: {bfs_score}")

    # --- MIPROv2 ---
    print("\n--- MIPROv2 (auto=light) ---")
    from dspy.teleprompt import MIPROv2
    classifier = TicketClassifier()
    opt = MIPROv2(metric=ticket_metric, auto="light", num_threads=1)
    optimized = opt.compile(
        student=classifier, trainset=TRAINSET, valset=VALSET,
    )
    mipro_score = evaluator(optimized)
    print(f"MIPROv2 score: {mipro_score}")

    # Save the MIPROv2 result (usually best)
    optimized.save("./optimized_classifier.json")
    print("\nSaved MIPROv2 result to ./optimized_classifier.json")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline:          {baseline}")
    print(f"  LabeledFewShot:    {lfs_score}")
    print(f"  BootstrapFewShot:  {bfs_score}")
    print(f"  MIPROv2 (light):   {mipro_score}")


if __name__ == "__main__":
    main()
