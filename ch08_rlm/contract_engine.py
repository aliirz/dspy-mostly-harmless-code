"""
contract_engine.py - Chapter 8: The Infinite Improbability Context
DSPy: The Mostly Harmless Guide

Demonstrates dspy.RLM (Recursive Language Model) for analyzing legal contracts
that are too large to fit in a single LLM context window.

Requirements:
  - Deno installed (https://deno.land/): curl -fsSL https://deno.land/install.sh | sh
  - Restart your shell after installing Deno
  - ANTHROPIC_API_KEY in .env

Run:
  poetry run python contract_engine.py
"""

import dspy
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# --- Data Model ---

class ContractRisk(BaseModel):
    """Structured risk analysis of a legal contract."""

    key_obligations: list[str] = Field(
        description="Critical obligations for each party — what they MUST do"
    )
    risk_flags: list[str] = Field(
        description="Unusual, one-sided, or potentially problematic clauses"
    )
    important_dates: list[str] = Field(
        description="Deadlines, notice periods, renewal windows with timeframes"
    )
    liability_cap: str = Field(
        description="Maximum liability limit stated in contract, or 'uncapped' if none"
    )
    overall_risk: str = Field(
        description="Overall risk level: 'low', 'medium', or 'high'"
    )


# --- Custom Tool ---

def extract_section(text: str, section_name: str) -> str:
    """Find and return a named section from a contract (first 3000 chars).

    Args:
        text: The full contract text.
        section_name: The section title to search for (e.g. 'TERMINATION').

    Returns:
        The section content, or a not-found message.
    """
    import re
    pattern = rf'(?i){re.escape(section_name)}.*?(?=\n\d+\.\s|\Z)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0)[:3000] if match else f"Section '{section_name}' not found"


# --- RLM Module ---

class ContractIntelligenceEngine(dspy.Module):
    """Analyze a legal contract of any length using dspy.RLM.

    The RLM explores the document programmatically — searching for specific
    clauses, extracting relevant sections, and using a sub-LLM for semantic
    analysis. The full contract text never enters the LLM's context window
    directly; only the portions the LLM chooses to examine are processed.
    """

    def __init__(self, sub_lm: dspy.LM | None = None):
        super().__init__()

        analysis_sig = dspy.Signature(
            {
                "contract": dspy.InputField(
                    desc="Full contract text — may be very long (100K+ characters)"
                ),
                "risk": dspy.OutputField(
                    desc="Structured risk analysis of the contract",
                    annotation=ContractRisk,
                ),
            },
            """You are a senior contract analyst. Analyze the contract systematically:
            1. Print the first 2000 chars to understand structure and identify parties
            2. Use extract_section() to pull key sections: TERMINATION, LIABILITY, IP, PAYMENT
            3. Use llm_query() to semantically analyze specific clauses you've extracted
            4. Identify obligations, risk flags, dates, and liability caps
            5. Assess overall risk before submitting""",
        )

        self.analyze = dspy.RLM(
            analysis_sig,
            max_iterations=12,
            max_llm_calls=20,
            sub_lm=sub_lm,
            tools=[extract_section],
            verbose=True,
        )

    def forward(self, contract_text: str) -> ContractRisk | str:
        result = self.analyze(contract=contract_text)
        # result.risk is ContractRisk when the REPL worked correctly.
        # When Deno/Pyodide fails on every iteration, the extract fallback
        # may return a plain string — still useful, just unstructured.
        return result.risk


# --- Sample Contract ---

SAMPLE_CONTRACT = """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 15, 2025
("Effective Date") between Acme Corporation, a Delaware corporation ("Client"),
and Vendor Solutions Ltd, a UK company ("Vendor").

1. SERVICES
Vendor shall provide software development and consulting services as detailed
in Exhibit A attached hereto. Services include: custom application development,
API integration, quality assurance testing, and technical documentation.
Client shall provide timely feedback within 5 business days of each deliverable.
Client must provide access to all necessary systems and data within 10 business
days of the Effective Date. Failure to provide access may delay the project
schedule without penalty to Vendor.

2. PAYMENT TERMS
Client shall pay Vendor $35,000 per month, due within 30 days of each invoice.
Late payments accrue interest at 1.5% per month (18% per annum).
Vendor may suspend services after 45 days of non-payment without liability.
All amounts are exclusive of applicable taxes.
Client is responsible for all withholding taxes in their jurisdiction.

3. TERM AND TERMINATION
This Agreement commences on the Effective Date and continues for 24 months
unless earlier terminated.
Either party may terminate for cause upon 30 days written notice if the other
party materially breaches this Agreement and fails to cure within the notice period.
Client may terminate for convenience upon 90 days written notice, subject to
payment of a termination fee equal to 3 months of fees ($105,000).
Upon termination, Vendor shall deliver all work product within 14 days.

4. INTELLECTUAL PROPERTY
All work product, inventions, and deliverables created under this Agreement
shall be considered work-for-hire and shall belong exclusively to Vendor
until full payment of all outstanding invoices is received.
Upon receipt of full payment, all intellectual property rights transfer
irrevocably to Client. Vendor retains the right to use general methodologies
and non-Client-specific know-how developed during this engagement.
Client grants Vendor a limited license to use Client's trademarks solely
for the purpose of providing the Services.

5. CONFIDENTIALITY
Each party agrees to maintain the confidentiality of the other party's
Confidential Information for a period of 5 years from disclosure.
Confidential Information excludes information that is publicly known,
independently developed, or required to be disclosed by law.

6. WARRANTIES AND REPRESENTATIONS
Vendor warrants that the Services will be performed in a professional and
workmanlike manner consistent with industry standards.
EXCEPT AS EXPRESSLY SET FORTH HEREIN, VENDOR MAKES NO WARRANTIES,
EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES
OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
ALL DELIVERABLES ARE PROVIDED "AS IS" AFTER FINAL ACCEPTANCE BY CLIENT.

7. INDEMNIFICATION
Each party shall indemnify and defend the other against third-party claims
arising from: (a) that party's breach of this Agreement; (b) that party's
gross negligence or willful misconduct; (c) infringement of third-party
intellectual property rights caused by that party.
Client shall indemnify Vendor against any claims arising from Client's
use of the deliverables in a manner not contemplated by this Agreement.

8-46. [Additional standard clauses regarding: force majeure, dispute resolution,
governing law, notices, assignment, entire agreement, modifications, waiver,
severability, counterparts, electronic signatures, audit rights, insurance
requirements, export compliance, anti-corruption compliance, data protection,
business continuity, subcontracting, key personnel, change management,
acceptance testing procedures, escrow arrangements, service levels,
performance metrics, benchmarking rights, technology refresh obligations,
third-party software licensing, open source compliance, background checks,
security requirements, incident response, business associate agreement terms,
GDPR data processing addendum, California CCPA compliance measures,
international data transfer mechanisms, records retention, disaster recovery,
and environmental compliance. Each clause is fully elaborated in the original
document. Total document length: 487,234 characters across 48 sections.]

47. LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE, OR CONSEQUENTIAL DAMAGES,
INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOSS OF BUSINESS, OR LOSS OF DATA,
EVEN IF SUCH PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
IN NO EVENT SHALL EITHER PARTY'S TOTAL LIABILITY TO THE OTHER PARTY EXCEED
THE FEES ACTUALLY PAID BY CLIENT TO VENDOR IN THE THREE (3) MONTH PERIOD
IMMEDIATELY PRECEDING THE EVENT GIVING RISE TO THE CLAIM.
THE FOREGOING LIMITATIONS SHALL NOT APPLY TO: (A) BREACHES OF CONFIDENTIALITY
OBLIGATIONS; (B) INDEMNIFICATION OBLIGATIONS; OR (C) GROSS NEGLIGENCE OR
WILLFUL MISCONDUCT.

48. GOVERNING LAW AND DISPUTE RESOLUTION
This Agreement shall be governed by the laws of the State of Delaware,
without regard to its conflict of law principles.
Any disputes shall first be subject to good-faith negotiation for 30 days.
If unresolved, disputes shall be submitted to binding arbitration under
the rules of the American Arbitration Association in New York, NY.
The prevailing party shall be entitled to recover reasonable attorneys' fees.
"""


# --- Main ---

def main():
    # Two-model strategy: Sonnet for orchestration, Haiku for extraction
    main_lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        max_tokens=4000,
    )
    cheap_lm = dspy.LM(
        "anthropic/claude-haiku-4-5-20251001",
        max_tokens=2000,
    )

    dspy.configure(lm=main_lm)

    print("=" * 60)
    print("Contract Intelligence Engine — Chapter 8")
    print("DSPy: The Mostly Harmless Guide")
    print("=" * 60)
    print(f"\nContract size: {len(SAMPLE_CONTRACT):,} characters")
    print("Analyzing with RLM (this will take 15–45 seconds)...\n")

    engine = ContractIntelligenceEngine(sub_lm=cheap_lm)

    try:
        risk = engine(contract_text=SAMPLE_CONTRACT)

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"\nOverall Risk: {risk.overall_risk.upper()}")

        print(f"\nKey Obligations ({len(risk.key_obligations)}):")
        for obligation in risk.key_obligations:
            print(f"  • {obligation}")

        print(f"\nRisk Flags ({len(risk.risk_flags)}):")
        for flag in risk.risk_flags:
            print(f"  ⚠  {flag}")

        print(f"\nImportant Dates/Deadlines ({len(risk.important_dates)}):")
        for date in risk.important_dates:
            print(f"  📅 {date}")

        print(f"\nLiability Cap: {risk.liability_cap}")

    except Exception as e:
        if "deno" in str(e).lower() or "pyodide" in str(e).lower():
            print(f"\nERROR: Deno is required for dspy.RLM.")
            print("Install it: curl -fsSL https://deno.land/install.sh | sh")
            print("Then restart your shell and try again.")
        else:
            raise


if __name__ == "__main__":
    main()
