"""
lead_engine.py - Chapter 2: The Restaurant at the End of the Pipeline
DSPy: The Mostly Harmless Guide

A multi-step lead intelligence pipeline that researches companies,
classifies prospect intent, scores leads, and generates outreach.
"""

import dspy
from pydantic import BaseModel, Field
from typing import Literal


# --- Pydantic models for structured data flow ---

class CompanyIntel(BaseModel):
    """Structured intelligence about a company."""
    name: str = Field(description="Official company name")
    industry: str = Field(description="Primary industry or sector")
    size_estimate: Literal["startup", "smb", "mid-market", "enterprise"] = Field(
        description="Estimated company size category"
    )
    recent_developments: list[str] = Field(
        description="Notable recent news, launches, or changes"
    )
    potential_pain_points: list[str] = Field(
        description="Business challenges this company likely faces"
    )
    tech_stack_signals: list[str] = Field(
        description="Any signals about their technology choices"
    )


class IntentClassification(BaseModel):
    """Classification of a prospect's buying intent."""
    intent_level: Literal["hot", "warm", "cold", "unknown"] = Field(
        description="Assessed buying intent level"
    )
    intent_signals: list[str] = Field(
        description="Specific signals that indicate this intent level"
    )
    buyer_persona: str = Field(
        description="Likely buyer persona (e.g., 'Technical Decision Maker', "
                    "'Budget Holder', 'End User', 'Champion')"
    )


# --- DSPy Signatures ---

class ResearchCompany(dspy.Signature):
    """Research a company and produce structured intelligence.
    Use your knowledge to infer likely details about the company,
    including industry, size, recent developments, pain points,
    and technology signals."""

    company_name: str = dspy.InputField(desc="Name of the company to research")
    prospect_role: str = dspy.InputField(
        desc="The prospect's role or title at the company"
    )
    intel: CompanyIntel = dspy.OutputField(
        desc="Structured intelligence about the company"
    )


class ClassifyIntent(dspy.Signature):
    """Based on company intelligence and prospect information,
    classify the prospect's likely buying intent. Consider their
    role, the company's pain points, and any engagement signals."""

    company_intel: CompanyIntel = dspy.InputField(
        desc="Structured intelligence about the prospect's company"
    )
    prospect_role: str = dspy.InputField(desc="The prospect's role or title")
    engagement_context: str = dspy.InputField(
        desc="How/where we encountered this prospect"
    )
    classification: IntentClassification = dspy.OutputField(
        desc="Intent classification with signals and buyer persona"
    )


class ScoreLead(dspy.Signature):
    """Score a lead from 1-100 based on all available intelligence.
    Consider: company fit (industry, size), intent signals,
    buyer persona match, and engagement quality.
    Be calibrated: 80+ is genuinely hot, 50-79 is worth pursuing,
    below 50 needs nurturing."""

    company_intel: CompanyIntel = dspy.InputField(
        desc="Structured company intelligence"
    )
    classification: IntentClassification = dspy.InputField(
        desc="Intent classification results"
    )
    lead_score: int = dspy.OutputField(desc="Lead quality score from 1-100")
    score_reasoning: str = dspy.OutputField(
        desc="Brief explanation of the score (2-3 sentences)"
    )
    recommended_action: Literal[
        "fast_track", "standard_nurture", "long_term_nurture", "disqualify"
    ] = dspy.OutputField(desc="Recommended next action based on score")


class GenerateOutreach(dspy.Signature):
    """Write a personalized outreach email for this prospect.
    The email should reference specific company details and pain points.
    Match the tone to the buyer persona. Keep it under 150 words.
    No generic templates — every line should feel researched."""

    company_intel: CompanyIntel = dspy.InputField(
        desc="Company intelligence for personalization"
    )
    classification: IntentClassification = dspy.InputField(
        desc="Intent classification for tone matching"
    )
    lead_score: int = dspy.InputField(desc="Lead score for urgency calibration")
    prospect_name: str = dspy.InputField(desc="The prospect's name")
    subject_line: str = dspy.OutputField(
        desc="Email subject line (compelling, not clickbait)"
    )
    email_body: str = dspy.OutputField(
        desc="Personalized email body (under 150 words)"
    )


# --- The Pipeline Module ---

class LeadIntelligenceEngine(dspy.Module):
    """A multi-step pipeline that researches a company, classifies
    prospect intent, scores the lead, and generates personalized outreach."""

    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought(ResearchCompany)
        self.classify = dspy.ChainOfThought(ClassifyIntent)
        self.score = dspy.Predict(ScoreLead)
        self.outreach = dspy.Predict(GenerateOutreach)

    def forward(
        self,
        company_name: str,
        prospect_name: str,
        prospect_role: str,
        engagement_context: str,
    ) -> dspy.Prediction:

        # Step 1: Research the company
        research_result = self.research(
            company_name=company_name,
            prospect_role=prospect_role,
        )

        # Step 2: Classify intent
        classify_result = self.classify(
            company_intel=research_result.intel,
            prospect_role=prospect_role,
            engagement_context=engagement_context,
        )

        # Step 3: Score the lead
        score_result = self.score(
            company_intel=research_result.intel,
            classification=classify_result.classification,
        )

        # Step 4: Generate outreach
        outreach_result = self.outreach(
            company_intel=research_result.intel,
            classification=classify_result.classification,
            lead_score=score_result.lead_score,
            prospect_name=prospect_name,
        )

        return dspy.Prediction(
            intel=research_result.intel,
            classification=classify_result.classification,
            lead_score=score_result.lead_score,
            score_reasoning=score_result.score_reasoning,
            recommended_action=score_result.recommended_action,
            subject_line=outreach_result.subject_line,
            email_body=outreach_result.email_body,
        )


def main():
    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)

    engine = LeadIntelligenceEngine()

    print("=" * 60)
    print("LEAD INTELLIGENCE ENGINE")
    print("=" * 60)

    result = engine(
        company_name="Stripe",
        prospect_name="Sarah Chen",
        prospect_role="VP of Engineering",
        engagement_context=(
            "Downloaded our API performance whitepaper and "
            "visited the enterprise pricing page twice this week"
        ),
    )

    print(f"\n--- COMPANY RESEARCH ---")
    print(f"Company: {result.intel.name}")
    print(f"Industry: {result.intel.industry}")
    print(f"Size: {result.intel.size_estimate}")
    print(f"Pain Points: {result.intel.potential_pain_points}")
    print(f"Tech Signals: {result.intel.tech_stack_signals}")

    print(f"\n--- INTENT CLASSIFICATION ---")
    print(f"Intent: {result.classification.intent_level}")
    print(f"Persona: {result.classification.buyer_persona}")
    print(f"Signals: {result.classification.intent_signals}")

    print(f"\n--- LEAD SCORE ---")
    print(f"Score: {result.lead_score}/100")
    print(f"Action: {result.recommended_action}")
    print(f"Reasoning: {result.score_reasoning}")

    print(f"\n--- OUTREACH ---")
    print(f"Subject: {result.subject_line}")
    print(f"Email:\n{result.email_body}")


if __name__ == "__main__":
    main()
