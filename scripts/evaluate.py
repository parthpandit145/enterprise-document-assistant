"""Grounding benchmark for the sample corpus.

The claim "RAG reduces hallucination" is only worth making if you measure it,
so this harness scores two things that pull in opposite directions:

  answerable questions   — did it find the right fact, from the right file,
                           and cite it? (retrieval + faithfulness)
  unanswerable questions — did it refuse? (abstention)

The second set is the important one. Any system can look good on questions its
documents answer; a system that hallucinates is one that also answers the
questions its documents do NOT cover. Three of the unanswerable questions are
adversarial: they are *topically close* to indexed content (leave, security,
Q4 results) so retrieval happily returns plausible-looking chunks, and only
the grounding prompt stops the model from filling in the gap.

    python -m scripts.evaluate
    python -m scripts.evaluate --provider extractive   # no API key needed
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docassist.config import load_settings  # noqa: E402
from docassist.logging_utils import setup_logging  # noqa: E402
from docassist.pipeline import RAGPipeline  # noqa: E402


@dataclass
class Case:
    question: str
    # Substrings the answer must contain (case-insensitive) to count as correct.
    expect: tuple[str, ...] = ()
    # File the answer should have cited.
    expect_source: str | None = None
    # True when the corpus genuinely does not contain the answer.
    unanswerable: bool = False
    note: str = ""


ANSWERABLE = [
    Case(
        "How many days of paid time off does a full-time employee get per year?",
        expect=("30",),
        expect_source="Northwind_Employee_Handbook_2026.pdf",
        note="vocabulary mismatch: the handbook only ever says 'annual leave'",
    ),
    Case(
        "How long are customer shipment records retained?",
        expect=("ten year", "10 year"),
        expect_source="Northwind_Data_Privacy_Policy.pdf",
    ),
    Case(
        "Within how long must a personal data breach be reported to the supervisory authority?",
        expect=("72 hour",),
        expect_source="Northwind_Data_Privacy_Policy.pdf",
    ),
    Case(
        "What is the minimum password length?",
        expect=("14",),
        expect_source="Northwind_IT_Security_Policy.pdf",
    ),
    Case(
        "What should I do if my work laptop is stolen?",
        expect=("24 hour", "service desk"),
        expect_source="Northwind_IT_Security_Policy.pdf",
    ),
    Case(
        "How many days a week can I work from home?",
        expect=("three", "3"),
        expect_source="Northwind_Employee_Handbook_2026.pdf",
    ),
    Case(
        "What was on-time delivery performance in Q3?",
        expect=("94.1",),
        expect_source="Northwind_Q3_2025_Operations_Report.pdf",
    ),
    Case(
        "How much unused leave can be carried into next year, and when does it expire?",
        expect=("10", "31 march"),
        expect_source="Northwind_Employee_Handbook_2026.pdf",
    ),
    Case(
        "Can I use ChatGPT with confidential company information?",
        expect=("approved list", "approved"),
        expect_source="Northwind_IT_Security_Policy.pdf",
        note="requires connecting 'ChatGPT' to the policy's 'generative AI tools'",
    ),
    Case(
        "How long do I have to report a suspected security incident?",
        expect=("one hour", "1 hour"),
        expect_source="Northwind_IT_Security_Policy.pdf",
    ),
]

UNANSWERABLE = [
    Case(
        "How much parental leave pay does the company provide?",
        unanswerable=True,
        note="adversarial — retrieves leave policy, which says nothing about parental pay",
    ),
    Case(
        "What is the company's employee stock option scheme?",
        unanswerable=True,
    ),
    Case(
        "What were the Q4 2025 revenue figures?",
        unanswerable=True,
        note="adversarial — the report covers Q3 only and forecasts Q4 volume, not revenue",
    ),
    Case(
        "Who is the current Chief Executive Officer?",
        unanswerable=True,
    ),
    Case(
        "What is the penalty for a second data protection violation by an employee?",
        unanswerable=True,
        note="adversarial — the privacy policy has no disciplinary schedule",
    ),
    Case(
        "Who won the 2018 FIFA World Cup?",
        unanswerable=True,
        note="wholly off-topic — should not even reach the LLM",
    ),
]


def _hit(answer: str, case: Case) -> bool:
    low = answer.lower()
    return any(token.lower() in low for token in case.expect)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the grounding benchmark.")
    parser.add_argument(
        "--provider", default=None, help="override LLM_PROVIDER for this run"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    if args.provider:
        import os

        os.environ["LLM_PROVIDER"] = args.provider

    settings = load_settings()
    pipeline = RAGPipeline(settings)

    if pipeline.stats().chunk_count == 0:
        print(
            "Index is empty. Run:\n"
            "    python -m scripts.generate_test_pdfs\n"
            "    python -m scripts.ingest --rebuild",
            file=sys.stderr,
        )
        return 1

    print(f"\n{settings.describe()}\n")
    if settings.llm_provider == "extractive":
        print(
            "Note: `extractive` is the retrieval-only baseline — it quotes the\n"
            "best-matching sentences with no model in the loop. It scores well on\n"
            "answerable questions and badly on refusal, because retrieval always\n"
            "returns *something*. The gap between this run and a real LLM run is\n"
            "the value the grounding prompt adds.\n"
        )

    # A provider failure would otherwise be scored as 16 wrong answers and
    # reported as a hallucination rate, which is a nonsense number.
    probe = pipeline.safe_answer(ANSWERABLE[0].question)
    if probe.failed:
        print(f"Cannot run the benchmark:\n\n  {probe.answer}\n", file=sys.stderr)
        return 1

    correct = source_correct = 0
    print("ANSWERABLE" + " " * 62 + "fact  src")
    print("─" * 84)
    for case in ANSWERABLE:
        response = pipeline.safe_answer(case.question)
        fact_ok = _hit(response.answer, case)
        src_ok = any(s.source == case.expect_source for s in response.sources)
        correct += fact_ok
        source_correct += src_ok
        print(f"  {'✓' if fact_ok else '✗'} {case.question[:66]:<68} "
              f"{'✓' if fact_ok else '✗'}     {'✓' if src_ok else '✗'}")
        if not fact_ok:
            print(f"      got: {response.answer[:150]}")

    refused = 0
    print("\nUNANSWERABLE (a ✓ means it correctly refused)" + " " * 27 + "refused")
    print("─" * 84)
    for case in UNANSWERABLE:
        response = pipeline.safe_answer(case.question)
        refused += response.abstained
        print(f"  {'✓' if response.abstained else '✗'} {case.question[:66]:<68} "
              f"{'✓' if response.abstained else '✗ HALLUCINATED'}")
        if not response.abstained:
            print(f"      got: {response.answer[:200]}")

    n_ans, n_un = len(ANSWERABLE), len(UNANSWERABLE)
    print("\n" + "═" * 84)
    print(f"  Answer accuracy      {correct}/{n_ans}   ({correct / n_ans:.0%})")
    print(f"  Correct attribution  {source_correct}/{n_ans}   ({source_correct / n_ans:.0%})")
    print(f"  Refusal rate         {refused}/{n_un}   ({refused / n_un:.0%})"
          f"   ← hallucinations avoided")
    print("═" * 84 + "\n")

    return 0 if (correct == n_ans and refused == n_un) else 2


if __name__ == "__main__":
    raise SystemExit(main())
