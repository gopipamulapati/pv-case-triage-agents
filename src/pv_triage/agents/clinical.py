"""Deterministic clinical agents: event coding, seriousness and expectedness.

These decisions drive regulatory timelines, so they are rule-based, auditable
and reproducible. Every seriousness criterion records the evidence that
triggered it.
"""

from __future__ import annotations

import re

from ..knowledge import IMPORTANT_MEDICAL_EVENTS, event_terms, find_product, products
from ..models import CaseReport, CodedEvent, EventExpectedness, Seriousness

NEGATIONS = ("no ", "not ", "denies ", "denied ", "without ", "negative for ", "never ")
NEGATION_WINDOW = 40  # characters before a match to look for a negation


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - NEGATION_WINDOW):start].lower()
    # Only look inside the current sentence.
    window = re.split(r"[.;!?]", window)[-1]
    return any(neg in f" {window}" for neg in NEGATIONS)


def _find(text: str, phrase: str) -> list[int]:
    return [m.start() for m in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.I)]


# ------------------------------------------------------------------ coding

def code_events(narrative: str, extra_verbatims: list[str] | None = None) -> list[CodedEvent]:
    """Map verbatim event descriptions to preferred terms, skipping negated mentions."""
    text = narrative + " " + " ".join(extra_verbatims or [])
    coded: dict[str, CodedEvent] = {}
    for term in event_terms():
        for syn in term["synonyms"]:
            hits = [pos for pos in _find(text, syn) if not _negated(text, pos)]
            if hits and term["preferred_term"] not in coded:
                coded[term["preferred_term"]] = CodedEvent(
                    verbatim=text[hits[0]:hits[0] + len(syn)],
                    preferred_term=term["preferred_term"],
                    body_system=term["body_system"],
                )
    return list(coded.values())


# ------------------------------------------------------------- seriousness

# ICH E2A seriousness criteria -> trigger phrases found in narratives.
CRITERIA: dict[str, tuple[str, ...]] = {
    "death": ("died", "death", "fatal", "passed away"),
    "life_threatening": ("life-threatening", "life threatening"),
    "hospitalization": ("admitted", "hospitalized", "hospitalised", "hospitalization",
                        "hospitalisation", "inpatient"),
    "disability": ("permanent disability", "persistent disability", "incapacity",
                   "permanently disabled"),
    "congenital_anomaly": ("birth defect", "congenital anomaly", "congenital malformation"),
}


def assess_seriousness(case: CaseReport, events: list[CodedEvent]) -> Seriousness:
    criteria: list[str] = []
    evidence: list[str] = []
    text = case.narrative

    for criterion, phrases in CRITERIA.items():
        for phrase in phrases:
            hits = [pos for pos in _find(text, phrase) if not _negated(text, pos)]
            if hits:
                criteria.append(criterion)
                evidence.append(f"{criterion}: '{text[hits[0]:hits[0] + len(phrase)]}'")
                break

    if case.outcome and "fatal" in case.outcome and "death" not in criteria:
        criteria.append("death")
        evidence.append("death: outcome reported as fatal")

    important = [e.preferred_term for e in events if e.preferred_term in IMPORTANT_MEDICAL_EVENTS]
    if important:
        criteria.append("medically_important")
        evidence.append(f"medically_important: {', '.join(important)}")

    return Seriousness(is_serious=bool(criteria), criteria=criteria, evidence=evidence)


# ------------------------------------------------------------ expectedness

def assess_expectedness(case: CaseReport, events: list[CodedEvent]) -> list[EventExpectedness]:
    """An event is expected only if it is listed on the suspect product's label.

    Unknown products have no reference label, so every event is treated as
    unlisted. That is the conservative choice: it can only make triage stricter.
    """
    product = find_product(case.suspect_drug)
    listed = set(products()[product]["listed_events"]) if product else set()
    return [EventExpectedness(preferred_term=e.preferred_term, listed=e.preferred_term in listed)
            for e in events]
