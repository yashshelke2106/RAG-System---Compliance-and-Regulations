"""Generate a small, self-contained compliance corpus as PDFs.

These documents are ORIGINAL, illustrative policies written for this demo. They
are modeled on the *structure* of real regulatory guidance (numbered sections,
defined terms, page breaks) so that citations like "AML Policy §4.2, page 2"
resolve exactly — but they are not copies of any real regulation.

    python -m scripts.generate_sample_pdfs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib.pagesizes import LETTER          # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch                 # noqa: E402
from reportlab.platypus import (                     # noqa: E402
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H0", parent=ss["Title"], fontSize=18, spaceAfter=14))
    ss.add(ParagraphStyle("Sec", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Body2", parent=ss["BodyText"], fontSize=10.5, leading=15))
    return ss


def build_doc(filename: str, title: str, blocks: list[tuple[str, str]], page_breaks: set[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ss = _styles()
    story = [Paragraph(title, ss["H0"]), Spacer(1, 6)]
    for i, (heading, body) in enumerate(blocks):
        if heading:
            story.append(Paragraph(heading, ss["Sec"]))
        story.append(Paragraph(body, ss["Body2"]))
        story.append(Spacer(1, 6))
        if i in page_breaks:
            story.append(PageBreak())
    SimpleDocTemplate(
        str(OUT / filename), pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
    ).build(story)
    print(f"wrote {OUT / filename}")


# ---------------------------------------------------------------------------
# Document 1: Anti-Money-Laundering Policy
# ---------------------------------------------------------------------------
AML = [
    ("1 Purpose and Scope",
     "This Policy establishes the minimum standards for detecting, preventing, and "
     "reporting money laundering and terrorist financing across all business units. "
     "It applies to every employee, contractor, and affiliated agent."),
    ("2 Definitions",
     "\"Beneficial owner\" means any natural person who ultimately owns or controls "
     "25 percent or more of a legal entity customer. \"Politically Exposed Person\" "
     "(PEP) means an individual entrusted with a prominent public function."),
    ("3 Customer Due Diligence",
     "The firm shall verify the identity of each customer before establishing a "
     "business relationship. Standard Customer Due Diligence (CDD) requires "
     "collection of legal name, date of birth, address, and a government-issued "
     "identification number."),
    ("4 Enhanced Due Diligence",
     "Enhanced Due Diligence (EDD) applies to higher-risk relationships."),
    ("4.1 Risk Triggers",
     "EDD is required where the customer is a Politically Exposed Person, is "
     "domiciled in a high-risk jurisdiction, or conducts transactions that are "
     "unusually large or lack an apparent economic purpose."),
    ("4.2 EDD Measures",
     "For relationships subject to EDD, the firm shall obtain senior management "
     "approval before onboarding, establish the source of funds and source of "
     "wealth, and conduct enhanced ongoing monitoring with a review frequency of "
     "no less than every twelve months."),
    ("5 Suspicious Activity Reporting",
     "Employees must report suspicious activity to the Compliance Officer within "
     "24 hours of detection. The Compliance Officer shall determine whether a "
     "Suspicious Activity Report is warranted and, if so, file it within the "
     "regulatory deadline. Tipping off a customer that a report has been filed is "
     "strictly prohibited."),
    ("6 Record Keeping",
     "All CDD and EDD records shall be retained for a minimum of five years after "
     "the end of the business relationship."),
]

# ---------------------------------------------------------------------------
# Document 2: KYC Onboarding Standard
# ---------------------------------------------------------------------------
KYC = [
    ("1 Overview",
     "This Standard defines the Know Your Customer (KYC) requirements for onboarding "
     "new customers and periodically refreshing existing customer information."),
    ("2 Identity Verification",
     "Individual customers must provide one primary photographic identity document "
     "and one proof of address dated within the last three months. Documents must "
     "be validated against an independent electronic data source."),
    ("3 Entity Customers",
     "For legal entity customers, the firm shall collect certificate of "
     "incorporation, register of directors, and ownership structure sufficient to "
     "identify all beneficial owners holding 25 percent or more."),
    ("4 Risk Rating",
     "Every customer shall be assigned a risk rating of Low, Medium, or High at "
     "onboarding based on geography, product, channel, and customer type."),
    ("4.1 Periodic Review",
     "Low-risk customers are reviewed every 36 months, Medium-risk every 24 months, "
     "and High-risk every 12 months. A trigger event, such as a change in "
     "beneficial ownership, requires an immediate out-of-cycle review."),
    ("5 Sanctions Screening",
     "All customers and related parties shall be screened against applicable "
     "sanctions lists at onboarding and re-screened whenever the lists are updated. "
     "A confirmed match must be escalated to Compliance and the relationship frozen "
     "pending investigation."),
]

# ---------------------------------------------------------------------------
# Document 3: Data Retention & Privacy Policy
# ---------------------------------------------------------------------------
DATA = [
    ("1 Scope",
     "This Policy governs the retention, protection, and disposal of personal and "
     "transactional data held by the firm."),
    ("2 Retention Periods",
     "Transaction records shall be retained for seven years. Marketing consent "
     "records shall be retained for the duration of the relationship plus two years."),
    ("3 Data Subject Rights",
     "Data subjects may request access to, correction of, or erasure of their "
     "personal data. The firm shall respond to a verified request within 30 "
     "calendar days."),
    ("4 Breach Notification",
     "A personal data breach that poses a risk to individuals shall be reported to "
     "the supervisory authority within 72 hours of the firm becoming aware of it."),
    ("5 Cross-Border Transfers",
     "Personal data may only be transferred outside the home jurisdiction where an "
     "adequacy decision or approved contractual safeguard is in place."),
]


def main() -> None:
    build_doc("AML-Policy.pdf", "Anti-Money-Laundering Policy", AML, page_breaks={3})
    build_doc("KYC-Onboarding-Standard.pdf", "KYC Onboarding Standard", KYC, page_breaks={2})
    build_doc("Data-Retention-Privacy-Policy.pdf", "Data Retention and Privacy Policy", DATA, page_breaks={2})


if __name__ == "__main__":
    main()
