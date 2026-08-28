"""
Generates the small synthetic PDF corpus used by the Phase 10 evaluation
harness (evaluation/run_eval.py).

Why synthetic rather than "grab some real PDFs": the harness needs exact,
verifiable ground truth (which document + page contains the answer to
each eval question) - that's only possible if we control the source
content. Four fictional-company documents across distinct topics (HR,
product warranty, finance, security) give enough variety for
cross-document questions and enough total pages (~18) for a meaningful
retrieval evaluation, while staying small enough to run quickly and to
read end-to-end in a few minutes if you want to sanity-check the ground
truth in evaluation/dataset/questions.json against the actual text.

Every page here is a short paragraph of concrete, specific facts (exact
numbers, dates, thresholds) - deliberately written so each fact maps
cleanly to one eval question, and so it's obvious when a
retrieval/answer is wrong (a real number either was or wasn't returned).

Run: python evaluation/dataset/generate_dataset.py
"""
import os

import pymupdf

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pdfs")

# --- Document content -------------------------------------------------
# Each document is a list of (page_title, page_body) tuples. Kept as plain
# paragraphs (no columns/tables) since Phase 10 is testing retrieval and
# generation quality, not OCR/layout parsing (that's Phase 9's concern).

DOCUMENTS = {
    "hr_policy.pdf": {
        "title": "Acme Corp — Employee Handbook: HR Policies",
        "pages": [
            (
                "Vacation Policy",
                "Full-time employees accrue 15 days of paid vacation per year. "
                "After 3 years of continuous service, the accrual rate increases "
                "to 20 days per year. Vacation requests must be submitted at "
                "least 2 weeks in advance through the HR portal.",
            ),
            (
                "Sick Leave",
                "Employees receive 10 paid sick days per calendar year. Unused "
                "sick days do not roll over into the next calendar year. A "
                "doctor's note is required for any sick absence longer than 3 "
                "consecutive days.",
            ),
            (
                "Remote Work Policy",
                "Employees may work remotely up to 3 days per week with manager "
                "approval. Fully remote arrangements require VP-level sign-off. "
                "Remote employees must be reachable during core hours of 10am "
                "to 4pm in their local time zone.",
            ),
            (
                "Termination and Notice Period",
                "Employees who resign must provide at least 30 days of written "
                "notice. The company may terminate employment with 2 weeks "
                "notice or pay in lieu of notice, except in cases of gross "
                "misconduct, which may result in immediate termination.",
            ),
            (
                "Code of Conduct",
                "Employees must not discriminate against or harass colleagues "
                "on the basis of any protected characteristic. Violations are "
                "reviewed by HR and may result in termination. All employees "
                "must complete annual ethics training by December 31st.",
            ),
        ],
    },
    "product_warranty.pdf": {
        "title": "Acme Widget Pro — Limited Warranty Terms",
        "pages": [
            (
                "Warranty Overview",
                "The Acme Widget Pro is covered by a limited warranty for 24 "
                "months from the original date of purchase. This warranty "
                "covers manufacturing defects in materials and workmanship "
                "under normal use.",
            ),
            (
                "Warranty Exclusions",
                "This warranty does not cover damage caused by misuse, "
                "unauthorized modification, water damage, or normal wear and "
                "tear. Accessories such as charging cables and power adapters "
                "are covered for 6 months only.",
            ),
            (
                "Filing a Claim",
                "To file a warranty claim, contact support@acmewidgets.example "
                "with your order number and a description of the defect. "
                "Approved claims are resolved via repair, replacement, or "
                "refund at Acme's discretion, typically within 10 business "
                "days of approval.",
            ),
            (
                "International Coverage",
                "Warranty coverage outside the United States and Canada is "
                "limited to 12 months from purchase and may be subject to "
                "additional local consumer protection laws. Contact your "
                "regional distributor for country-specific details.",
            ),
        ],
    },
    "q3_financial_report.pdf": {
        "title": "Acme Corp — Q3 2026 Financial Summary",
        "pages": [
            (
                "Revenue Overview",
                "Acme Corp reported Q3 2026 revenue of $42.3 million, up 18% "
                "year-over-year. Subscription revenue accounted for 71% of "
                "total revenue, with the remainder from professional services "
                "and one-time licensing.",
            ),
            (
                "Operating Expenses",
                "Total operating expenses for Q3 2026 were $31.7 million. This "
                "included $12.1 million in research and development and $9.4 "
                "million in sales and marketing.",
            ),
            (
                "Headcount",
                "Acme Corp ended Q3 2026 with 412 full-time employees, an "
                "increase of 34 from the previous quarter. Engineering was the "
                "largest department, at 168 employees.",
            ),
            (
                "Outlook",
                "Management expects Q4 2026 revenue to be between $45 million "
                "and $47 million, driven primarily by continued growth in the "
                "enterprise customer segment.",
            ),
        ],
    },
    "security_policy.pdf": {
        "title": "Acme Corp — Information Security Policy",
        "pages": [
            (
                "Password Requirements",
                "All employee passwords must be at least 12 characters long "
                "and include a mix of uppercase letters, lowercase letters, "
                "numbers, and symbols. Passwords must be rotated every 90 "
                "days.",
            ),
            (
                "Access Control",
                "Access to production systems requires multi-factor "
                "authentication. Access is granted on a least-privilege basis "
                "and is reviewed quarterly by the security team.",
            ),
            (
                "Incident Response",
                "Security incidents must be reported to the security team "
                "within 1 hour of discovery. The incident response team will "
                "triage and classify all reported incidents within 4 hours.",
            ),
            (
                "Data Retention",
                "Customer data is retained for 7 years after account closure "
                "to meet regulatory requirements, then securely deleted. "
                "System backup logs are retained for 90 days.",
            ),
        ],
    },
}


def _build_pdf(filename: str, title: str, pages: list[tuple[str, str]]) -> str:
    doc = pymupdf.open()
    for page_title, body in pages:
        page = doc.new_page(width=612, height=792)  # US Letter
        page.insert_text((72, 72), title, fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4))
        page.insert_text((72, 110), page_title, fontsize=16, fontname="helv")
        page.insert_textbox(
            pymupdf.Rect(72, 140, 540, 700), body, fontsize=11, fontname="helv"
        )
        page.insert_text((72, 760), f"{page_title}", fontsize=8, fontname="helv", color=(0.6, 0.6, 0.6))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    doc.save(out_path)
    doc.close()
    return out_path


def generate_dataset() -> list[str]:
    paths = []
    for filename, spec in DOCUMENTS.items():
        path = _build_pdf(filename, spec["title"], spec["pages"])
        paths.append(path)
        print(f"Generated {path} ({len(spec['pages'])} pages)")
    return paths


if __name__ == "__main__":
    generate_dataset()
