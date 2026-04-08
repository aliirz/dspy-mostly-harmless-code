"""
startup_roaster.py - Chapter 1: Don't Panic
DSPy: The Mostly Harmless Guide

Your first DSPy program: a startup pitch analyzer that's
honest, structured, and doesn't require a single prompt template.
"""

import dspy


# --- Step 1: Define the Signature ---

class RoastStartup(dspy.Signature):
    """You are a sharp, witty startup analyst known for cutting through
    hype with humor. Analyze the given startup pitch and provide an
    honest, entertaining evaluation. Be constructive but don't pull punches."""

    pitch: str = dspy.InputField(
        desc="The startup pitch or idea to analyze"
    )
    roast: str = dspy.OutputField(
        desc="A witty, honest roast of the startup idea (2-3 sentences)"
    )
    viability_score: int = dspy.OutputField(
        desc="Viability score from 1 (terrible) to 10 (brilliant)"
    )
    strengths: list[str] = dspy.OutputField(
        desc="List of genuine strengths (be specific)"
    )
    weaknesses: list[str] = dspy.OutputField(
        desc="List of weaknesses and risks (be specific)"
    )
    verdict: str = dspy.OutputField(
        desc="One-sentence final verdict: invest or run?"
    )


# --- Step 2: Build the Module ---

class StartupRoaster(dspy.Module):
    """A module that roasts startup pitches with structured analysis.

    Why a Module instead of bare Predict? Modules are composable,
    optimizable, and testable. Even for simple programs, wrapping
    your Predict in a Module is a habit worth building early.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.Predict(RoastStartup)

    def forward(self, pitch: str) -> dspy.Prediction:
        return self.analyze(pitch=pitch)


# --- Step 3: Configure and Run ---

def main():
    # Configure the LM
    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)

    # Create the roaster
    roaster = StartupRoaster()

    # Some pitches to analyze
    pitches = [
        (
            "An AI-powered toothbrush that writes poetry while you brush. "
            "Uses an LLM to generate haikus about dental hygiene and posts "
            "them to Twitter automatically. $50/month subscription."
        ),
        (
            "A platform that connects freelance developers with nonprofits "
            "who need technical help but can't afford market rates. "
            "Developers get tax deductions and portfolio pieces. "
            "Nonprofits get quality software. Revenue from premium matching "
            "and enterprise volunteer programs."
        ),
        (
            "Uber for dogs. Not dog walking — actual rideshare for dogs. "
            "Your dog needs to get to the vet but you're at work? "
            "Book a DogUber. We handle pickup, transport, and drop-off. "
            "GPS tracking and live video feed included."
        ),
    ]

    for i, pitch in enumerate(pitches, 1):
        print(f"\n{'='*60}")
        print(f"PITCH #{i}")
        print(f"{'='*60}")
        print(f"{pitch[:80]}...")
        print()

        result = roaster(pitch=pitch)

        print(f"ROAST: {result.roast}")
        print(f"VIABILITY: {result.viability_score}/10")
        print(f"STRENGTHS: {result.strengths}")
        print(f"WEAKNESSES: {result.weaknesses}")
        print(f"VERDICT: {result.verdict}")

    # Show what DSPy actually sent to the LLM
    print(f"\n{'='*60}")
    print("BEHIND THE CURTAIN: What DSPy sent to the LLM")
    print(f"{'='*60}")
    dspy.inspect_history(n=1)


if __name__ == "__main__":
    main()
