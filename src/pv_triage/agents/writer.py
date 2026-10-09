"""Writer agent: case summary for the safety reviewer, with two output guardrails.

1. Priority lock: an LLM rewrite must keep the exact triage label.
2. PHI leak check: an LLM rewrite must not contain any value removed by the
   redaction step.

If a rewrite fails either check, the deterministic template is used instead.
"""

from __future__ import annotations

from ..llm import LLM
from ..models import (
    CaseReport,
    CodedEvent,
    EventExpectedness,
    Priority,
    Seriousness,
    Triage,
    ValidityCheck,
)
from .phi import leaks

SYSTEM = (
    "You write concise case summaries for a drug safety reviewer. Use only the "
    "facts given, keep the triage label exactly as written, and never add patient "
    "names, contact details or other identifiers. Use markdown."
)

LABEL = {
    Priority.EXPEDITED: "EXPEDITED",
    Priority.PRIORITY: "PRIORITY",
    Priority.ROUTINE: "ROUTINE",
    Priority.FOLLOW_UP: "FOLLOW-UP REQUIRED",
}


def template_summary(
    case: CaseReport,
    validity: ValidityCheck,
    events: list[CodedEvent],
    seriousness: Seriousness | None,
    expectedness: list[EventExpectedness],
    triage: Triage,
) -> str:
    patient = ", ".join(
        p for p in (f"{case.patient_age}y" if case.patient_age is not None else None,
                    case.patient_sex) if p
    ) or "not identifiable"
    due = triage.due_date.isoformat() if triage.due_date else "n/a"
    lines = [
        f"## Case triage: {LABEL[triage.priority]}",
        "",
        f"**Due:** {due}  ",
        f"**Reason:** {triage.reason}",
        "",
        f"- **Suspect product:** {case.suspect_drug or 'unknown'}"
        + (f" ({case.dose})" if case.dose else ""),
        f"- **Patient:** {patient}",
        f"- **Reporter:** {case.reporter_type or 'unknown'}",
        f"- **Outcome:** {case.outcome or 'unknown'}",
    ]
    if not validity.is_valid:
        lines += ["", "### Follow-up needed",
                  *[f"- Obtain: {m.replace('_', ' ')}" for m in validity.missing]]
    if events:
        listed = {e.preferred_term: e.listed for e in expectedness}
        lines += ["", "### Events", "| Verbatim | Preferred term | Body system | Label |",
                  "|---|---|---|---|"]
        for e in events:
            status = "listed" if listed.get(e.preferred_term) else "**unlisted**"
            lines.append(f"| {e.verbatim} | {e.preferred_term} | {e.body_system} | {status} |")
    if seriousness:
        lines += ["", f"### Seriousness: {'SERIOUS' if seriousness.is_serious else 'non-serious'}"]
        lines += [f"- {ev}" for ev in seriousness.evidence]
    lines += ["", "_Automated pre-triage. A qualified safety reviewer confirms every case._"]
    return "\n".join(lines)


def write(
    case: CaseReport,
    validity: ValidityCheck,
    events: list[CodedEvent],
    seriousness: Seriousness | None,
    expectedness: list[EventExpectedness],
    triage: Triage,
    removed_phi: list[str],
    llm: LLM,
) -> tuple[str, str]:
    """Return (summary, source) where source is "template" or "llm"."""
    draft = template_summary(case, validity, events, seriousness, expectedness, triage)
    polished = llm.complete(SYSTEM, draft).strip()
    if polished and LABEL[triage.priority] in polished and not leaks(polished, removed_phi):
        return polished, "llm"
    return draft, "template"
