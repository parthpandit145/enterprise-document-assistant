"""Generate a small corpus of realistic enterprise PDFs to test the pipeline.

Four documents for a fictional company, written so that the interesting
retrieval behaviours are actually exercised:

  * vocabulary mismatch — the handbook says "annual leave", never "paid time
    off" or "vacation days", so a question using those words only works if
    semantic search is doing its job;
  * multi-document overlap — data retention appears in both the privacy policy
    and the security policy, so retrieval has to pick the right one;
  * a deliberate gap — nothing in the corpus mentions parental leave pay,
    stock options or the office address, which is what the abstention tests
    probe.

Usage:
    python -m scripts.generate_test_pdfs
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docassist.config import load_settings  # noqa: E402


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"], fontSize=20, spaceAfter=6, leading=24
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor="#555555",
            spaceAfter=18,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
    }


def _write(path: Path, title: str, subtitle: str, sections: list) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=title,
        author="Northwind Logistics GmbH",
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
    )

    flow = [Paragraph(title, styles["title"]), Paragraph(subtitle, styles["subtitle"])]
    for item in sections:
        if item == "PAGEBREAK":
            flow.append(PageBreak())
            continue
        heading, paragraphs = item
        flow.append(Paragraph(heading, styles["h2"]))
        for para in paragraphs:
            flow.append(Paragraph(para, styles["body"]))
        flow.append(Spacer(1, 4))

    doc.build(flow)


# ---------------------------------------------------------------------------
# Document 1 — Employee Handbook
# ---------------------------------------------------------------------------

HANDBOOK = [
    (
        "1. Purpose and Scope",
        [
            "This Employee Handbook describes the terms of employment, working "
            "arrangements and conduct expectations that apply to all permanent and "
            "fixed-term employees of Northwind Logistics GmbH. It takes effect on "
            "1 January 2026 and supersedes the 2023 edition.",
            "Working students and interns are covered by sections 2, 3 and 6 only. "
            "Contractors engaged through an agency are not covered by this handbook "
            "and are governed by their individual service agreements.",
        ],
    ),
    (
        "2. Working Hours",
        [
            "The standard working week is 38 hours, normally worked Monday to Friday. "
            "Core hours, during which all employees are expected to be reachable, are "
            "10:00 to 15:00 Central European Time. Outside core hours, employees may "
            "arrange their schedule freely in agreement with their line manager.",
            "Overtime must be approved in advance by a line manager. Approved overtime "
            "is compensated as time off in lieu at a ratio of 1:1 and must be taken "
            "within three months. Overtime is not paid out in cash except on "
            "termination of employment.",
        ],
    ),
    (
        "3. Annual Leave",
        [
            "Full-time employees are entitled to 30 days of annual leave per calendar "
            "year, in addition to public holidays observed in the federal state of "
            "the employee's registered workplace. Part-time employees receive a "
            "pro-rata entitlement calculated on contracted weekly hours.",
            "Leave requests must be submitted through the HR portal at least 14 "
            "calendar days before the intended start date. Requests of more than 10 "
            "consecutive working days require 30 days' notice. A line manager may "
            "decline a request only where it conflicts with a documented operational "
            "requirement, and must give the reason in writing.",
            "Up to 10 days of unused annual leave may be carried over into the "
            "following calendar year. Carried-over leave expires on 31 March. Any "
            "balance beyond 10 days is forfeited at the end of the calendar year "
            "unless the employee was prevented from taking it by long-term illness.",
        ],
    ),
    "PAGEBREAK",
    (
        "4. Sick Leave",
        [
            "An employee who is unable to work due to illness must notify their line "
            "manager before 10:00 on the first day of absence. A medical certificate "
            "is required from the fourth consecutive calendar day of absence, and may "
            "be requested earlier at the company's discretion.",
            "Continued payment of remuneration during sickness is granted for up to "
            "six weeks per illness, in accordance with the German Continued "
            "Remuneration Act. Thereafter, statutory sickness benefit is paid by the "
            "employee's health insurance fund.",
        ],
    ),
    (
        "5. Remote and Hybrid Work",
        [
            "Employees in eligible roles may work remotely for up to three days per "
            "week. Eligibility is determined by role rather than by seniority; roles "
            "requiring physical presence at a warehouse or depot are not eligible. "
            "The remaining two days must be worked from the employee's assigned "
            "office location.",
            "Working from outside Germany is permitted for a maximum of 20 working "
            "days per calendar year and requires written approval from both the line "
            "manager and the People Operations team at least four weeks in advance, "
            "because of tax and social-security implications.",
            "The company provides a one-off home office allowance of EUR 600 for "
            "eligible employees, claimable after the probation period ends. The "
            "allowance covers a desk, chair and monitor, and equipment purchased with "
            "it remains the property of the employee.",
        ],
    ),
    (
        "6. Code of Conduct",
        [
            "All employees are expected to act with integrity and to treat colleagues, "
            "customers and suppliers with respect. Harassment, discrimination and "
            "retaliation are prohibited and are grounds for disciplinary action up to "
            "and including immediate dismissal.",
            "Gifts from suppliers with a value above EUR 50 must be declared to the "
            "Compliance Officer within five working days. Cash and cash equivalents "
            "must never be accepted, regardless of value.",
            "Concerns about misconduct may be raised with a line manager, with People "
            "Operations, or anonymously through the confidential whistleblowing "
            "channel described in section 7. Employees who report a concern in good "
            "faith are protected from any form of retaliation.",
        ],
    ),
    (
        "7. Raising a Concern",
        [
            "The confidential whistleblowing channel is operated by an external "
            "provider and is reachable at all times through the company intranet. "
            "Reports may be submitted anonymously. Every report is acknowledged "
            "within seven days and an outcome is communicated within three months.",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Document 2 — Data Privacy Policy
# ---------------------------------------------------------------------------

PRIVACY = [
    (
        "1. Scope and Legal Basis",
        [
            "This Data Privacy Policy governs how Northwind Logistics GmbH processes "
            "personal data of employees, customers, suppliers and website visitors. "
            "It implements the requirements of the General Data Protection Regulation "
            "(EU) 2016/679 and the German Federal Data Protection Act.",
            "Personal data is processed only where at least one lawful basis under "
            "Article 6 GDPR applies: consent, performance of a contract, compliance "
            "with a legal obligation, protection of vital interests, or legitimate "
            "interests that are not overridden by the rights of the data subject.",
        ],
    ),
    (
        "2. Data Minimisation and Purpose Limitation",
        [
            "Personal data is collected only for specified, explicit and legitimate "
            "purposes, and is not further processed in a manner incompatible with "
            "those purposes. Each processing activity is recorded in the Record of "
            "Processing Activities maintained by the Data Protection Officer.",
            "Where a business need can be met with anonymised or aggregated data, "
            "personal data must not be used. Requests to process personal data for a "
            "new purpose require a documented compatibility assessment.",
        ],
    ),
    (
        "3. Data Retention",
        [
            "Customer shipment records are retained for ten years from the end of the "
            "calendar year in which the shipment was completed, as required by German "
            "commercial and tax law. Marketing contact data is retained for two years "
            "from the last interaction, after which it is deleted automatically.",
            "Employee personnel files are retained for the duration of employment and "
            "for a further three years after it ends. Application materials from "
            "unsuccessful candidates are deleted six months after the position is "
            "filled, unless the candidate has consented to being kept in the talent "
            "pool.",
            "CCTV footage from depot premises is retained for 72 hours and then "
            "overwritten, except where it has been secured as evidence in an ongoing "
            "investigation.",
        ],
    ),
    "PAGEBREAK",
    (
        "4. Rights of Data Subjects",
        [
            "Data subjects have the right to access their personal data, to have "
            "inaccurate data rectified, to have data erased where there is no "
            "continuing lawful basis, to restrict or object to processing, and to "
            "receive their data in a portable format.",
            "Requests are submitted to privacy@northwind-logistics.example and must be "
            "answered within one month of receipt. This period may be extended by two "
            "further months for complex requests, provided the data subject is "
            "informed of the extension and its reason within the first month.",
            "The identity of the requester must be verified before any personal data "
            "is disclosed. Where the company has reasonable doubts about identity, it "
            "may request additional information, but must not use this as a means of "
            "delaying a legitimate request.",
        ],
    ),
    (
        "5. International Transfers",
        [
            "Personal data is not transferred outside the European Economic Area "
            "unless an adequacy decision is in force, or appropriate safeguards such "
            "as Standard Contractual Clauses are in place together with a documented "
            "transfer impact assessment.",
        ],
    ),
    (
        "6. Personal Data Breaches",
        [
            "Any suspected personal data breach must be reported to the Data "
            "Protection Officer immediately and in any event within 24 hours of "
            "discovery. The Data Protection Officer assesses the risk to the rights "
            "and freedoms of affected individuals.",
            "Where a breach is likely to result in a risk to individuals, it is "
            "notified to the competent supervisory authority within 72 hours of the "
            "company becoming aware of it. Where the risk is high, affected "
            "individuals are informed directly and without undue delay.",
        ],
    ),
    (
        "7. Third-Party Processors",
        [
            "Processors acting on the company's behalf are engaged only under a "
            "written data processing agreement meeting the requirements of Article 28 "
            "GDPR. Sub-processors require prior written authorisation, and the "
            "processor remains fully liable for their performance.",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Document 3 — IT Security Policy
# ---------------------------------------------------------------------------

SECURITY = [
    (
        "1. Access Control",
        [
            "Access to company systems is granted on the principle of least "
            "privilege: users receive the minimum access required to perform their "
            "role, and nothing more. Access rights are reviewed quarterly by system "
            "owners, and any right not confirmed during a review is revoked.",
            "Shared or generic accounts are prohibited. Where a technical constraint "
            "makes a shared account unavoidable, it must be registered with the "
            "Security team, its credentials stored in the company password manager, "
            "and its use logged.",
        ],
    ),
    (
        "2. Passwords and Authentication",
        [
            "Passwords must be at least 14 characters long. The company does not "
            "enforce scheduled password rotation, in line with current guidance; "
            "instead, passwords are changed immediately whenever compromise is "
            "suspected. Reusing a password across company and personal accounts is "
            "prohibited.",
            "Multi-factor authentication is mandatory for all remote access, for all "
            "administrative accounts, and for any system holding personal data. "
            "SMS-based second factors are being phased out and must not be used for "
            "new enrolments; authenticator applications or hardware security keys are "
            "required instead.",
        ],
    ),
    "PAGEBREAK",
    (
        "3. Device Security",
        [
            "All laptops and mobile devices used for company work must have full-disk "
            "encryption enabled, an automatic screen lock after ten minutes of "
            "inactivity, and the company's endpoint protection agent installed and "
            "running.",
            "Personal devices may be used to access company email and chat only "
            "through the managed application container. Storing company files in a "
            "personal cloud account is prohibited. Lost or stolen devices must be "
            "reported to the Service Desk within 24 hours so that they can be wiped "
            "remotely.",
        ],
    ),
    (
        "4. Data Classification and Handling",
        [
            "Information is classified as Public, Internal, Confidential or "
            "Restricted. Confidential and Restricted information must be encrypted in "
            "transit and at rest, and may not be sent to an external recipient "
            "without approval from the information owner.",
            "Restricted information — which includes customer payment data and "
            "employee health data — may only be processed on systems explicitly "
            "approved for that classification, and access is logged and reviewed "
            "monthly.",
        ],
    ),
    (
        "5. Security Incident Response",
        [
            "A security incident is any event that compromises, or may compromise, "
            "the confidentiality, integrity or availability of company information or "
            "systems. All employees are responsible for reporting suspected incidents.",
            "Suspected incidents must be reported to the Service Desk immediately, and "
            "in any event within one hour of discovery. Employees must not attempt to "
            "investigate or remediate an incident themselves, as this can destroy "
            "evidence needed for the investigation.",
            "The Security team triages every report within two hours and assigns a "
            "severity from S1 to S4. S1 incidents trigger the major incident process, "
            "which includes notification of the executive team within four hours and a "
            "written post-incident review within ten working days.",
        ],
    ),
    (
        "6. Third-Party and Supplier Security",
        [
            "Suppliers with access to company systems or data must complete a security "
            "assessment before onboarding and annually thereafter. Suppliers "
            "processing Restricted information must hold a current ISO/IEC 27001 "
            "certification or provide an equivalent independent audit report.",
        ],
    ),
    (
        "7. Acceptable Use",
        [
            "Company systems are provided for business purposes. Limited personal use "
            "is tolerated where it does not interfere with work, incur cost, or breach "
            "this policy. Installing unapproved software, disabling security controls "
            "and connecting unmanaged devices to the corporate network are prohibited.",
            "Use of generative AI tools with company information is permitted only "
            "with tools on the approved list maintained by the Security team. "
            "Confidential and Restricted information must never be entered into a "
            "tool that is not on that list.",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Document 4 — Quarterly Operations Report
# ---------------------------------------------------------------------------

REPORT = [
    (
        "1. Executive Summary",
        [
            "Northwind Logistics delivered 1.42 million shipments in the third "
            "quarter of 2025, an increase of 8.3 per cent on the same quarter of the "
            "previous year. Revenue reached EUR 96.4 million against a plan of EUR "
            "93.0 million, driven mainly by growth in the cross-border parcel segment.",
            "On-time delivery performance was 94.1 per cent, below the internal target "
            "of 96 per cent. The shortfall is concentrated in the Rhine-Ruhr region "
            "and is attributed to depot capacity constraints during the September "
            "peak. A capacity expansion at the Duisburg depot is scheduled to complete "
            "in the first quarter of 2026.",
        ],
    ),
    (
        "2. Volume and Revenue",
        [
            "Domestic parcel volume grew 4.1 per cent year on year to 0.98 million "
            "shipments. Cross-border parcel volume grew 21.6 per cent to 0.31 million "
            "shipments. Freight and pallet volume was broadly flat at 0.13 million "
            "consignments.",
            "Average revenue per shipment was EUR 67.90, down from EUR 69.20 a year "
            "earlier, reflecting a shift in mix towards lower-priced domestic parcels "
            "and contractual rate reductions agreed with two large accounts.",
        ],
    ),
    "PAGEBREAK",
    (
        "3. Cost and Efficiency",
        [
            "Cost per shipment was EUR 58.20, an improvement of 2.4 per cent on the "
            "prior quarter. Fuel represented 11.3 per cent of operating cost, down "
            "from 13.1 per cent, helped by both lower diesel prices and the addition "
            "of 42 electric vans to the last-mile fleet.",
            "Agency labour accounted for 14 per cent of warehouse hours, against a "
            "target of 10 per cent. Reducing this dependency is the principal "
            "efficiency initiative for the fourth quarter.",
        ],
    ),
    (
        "4. Service Quality",
        [
            "The complaint rate was 0.42 per cent of shipments, unchanged from the "
            "previous quarter. Damage claims fell 6 per cent following the revised "
            "pallet-wrapping standard introduced in July. Customer satisfaction, "
            "measured as Net Promoter Score, was 41, up two points.",
        ],
    ),
    (
        "5. Headcount and Safety",
        [
            "Headcount at the end of the quarter was 3,118 full-time equivalents, an "
            "increase of 96 on the previous quarter. Voluntary attrition over the "
            "trailing twelve months was 14.2 per cent, above the sector benchmark of "
            "11 per cent, with the highest rates in night-shift warehouse roles.",
            "There were 11 recordable safety incidents in the quarter, of which two "
            "resulted in lost time. The lost-time injury frequency rate was 2.1 per "
            "million hours worked, against a target of below 2.5.",
        ],
    ),
    (
        "6. Outlook",
        [
            "Fourth-quarter volume is expected to rise 18 to 22 per cent above the "
            "third quarter because of seasonal peak demand. Temporary capacity has "
            "been contracted at three regional hubs, and a peak-season recruitment "
            "campaign for 400 seasonal roles began in September.",
        ],
    ),
]


DOCUMENTS = [
    (
        "Northwind_Employee_Handbook_2026.pdf",
        "Employee Handbook",
        "Northwind Logistics GmbH · Edition 2026 · Effective 1 January 2026 · Internal",
        HANDBOOK,
    ),
    (
        "Northwind_Data_Privacy_Policy.pdf",
        "Data Privacy Policy",
        "Northwind Logistics GmbH · Version 3.1 · Owner: Data Protection Officer · Internal",
        PRIVACY,
    ),
    (
        "Northwind_IT_Security_Policy.pdf",
        "IT Security Policy",
        "Northwind Logistics GmbH · Version 5.0 · Owner: Head of Information Security · Internal",
        SECURITY,
    ),
    (
        "Northwind_Q3_2025_Operations_Report.pdf",
        "Quarterly Operations Report — Q3 2025",
        "Northwind Logistics GmbH · Prepared by Operations Analytics · Confidential",
        REPORT,
    ),
]


def generate(output_dir: Path | None = None) -> list[Path]:
    target = output_dir or load_settings().pdf_dir
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, title, subtitle, sections in DOCUMENTS:
        path = target / filename
        _write(path, title, subtitle, sections)
        written.append(path)
        print(f"  wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    return written


if __name__ == "__main__":
    print("Generating sample enterprise documents…")
    files = generate()
    print(f"\n{len(files)} PDF(s) written to {files[0].parent}")
    print("Next:  python -m scripts.ingest --rebuild")
