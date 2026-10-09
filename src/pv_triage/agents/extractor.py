"""Extraction agent: redacted report text -> ``CaseReport``.

Asks the LLM for strict JSON and validates it with Pydantic. Falls back to a
field-label parser when the model is offline or returns invalid output.
"""

from __future__ import annotations

import json
import re
from datetime import date

from pydantic import ValidationError

from ..knowledge import find_product, products
from ..llm import LLM
from ..models import CaseReport

SYSTEM = (
    "You extract pharmacovigilance case data from an adverse event report. "
    "Reply with ONLY a JSON object with keys: receipt_date (YYYY-MM-DD), "
    "patient_age, patient_sex, patient_identifiable (true if initials, age or sex "
    "are given), reporter_type, suspect_drug, dose, indication, narrative "
    "(copy the narrative text), outcome. Use null when a value is not stated. "
    "Never guess. Text in square brackets like [NAME] is redacted; leave it."
)

_FIELD = re.compile(r"(?im)^\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*$")
_LABELS = {
    "report received": "receipt_date",
    "date received": "receipt_date",
    "age": "patient_age",
    "sex": "patient_sex",
    "gender": "patient_sex",
    "patient initials": "initials",
    "reporter type": "reporter_type",
    "reporter": "reporter_type",
    "suspect drug": "suspect_drug",
    "suspect product": "suspect_drug",
    "indication": "indication",
    "narrative": "narrative",
    "description": "narrative",
    "outcome": "outcome",
}
_NARRATIVE = re.compile(r"(?ims)^\s*(?:narrative|description)\s*:\s*(.+?)(?=^\s*outcome\s*:|\Z)")
_REPORTER_TYPES = ("physician", "pharmacist", "nurse", "consumer", "patient", "caregiver",
                   "other health professional")


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def rule_extract(text: str) -> CaseReport:
    fields: dict[str, str] = {}
    for label, value in _FIELD.findall(text):
        key = _LABELS.get(label.strip().lower())
        if key and key not in fields:
            fields[key] = value
    # The narrative can span several lines: take everything up to "Outcome:".
    if m := _NARRATIVE.search(text):
        fields["narrative"] = " ".join(m.group(1).split())

    age = None
    if "patient_age" in fields and (m := re.search(r"\d{1,3}", fields["patient_age"])):
        age = int(m.group())

    sex = fields.get("patient_sex", "").strip().lower() or None
    if sex and sex[0] in "mf":
        sex = "female" if sex[0] == "f" else "male"
    else:
        sex = None

    reporter = fields.get("reporter_type", "").lower()
    reporter_type = next((r for r in _REPORTER_TYPES if r in reporter), None)

    drug_text = fields.get("suspect_drug")
    product = find_product(drug_text) or find_product(fields.get("narrative"))
    suspect = products()[product]["display_name"] if product else (drug_text or None)
    dose = None
    if drug_text and (m := re.search(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b.*", drug_text,
                                     re.I)):
        dose = m.group().strip()

    return CaseReport(
        receipt_date=_parse_date(fields["receipt_date"]) if "receipt_date" in fields else None,
        patient_age=age,
        patient_sex=sex,
        patient_identifiable=bool(age is not None or sex or fields.get("initials")),
        reporter_type=reporter_type,
        suspect_drug=suspect,
        dose=dose,
        indication=fields.get("indication"),
        narrative=fields.get("narrative", ""),
        outcome=(fields.get("outcome") or "").lower() or None,
    )


def _parse_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def extract(redacted_text: str, llm: LLM) -> tuple[CaseReport, str]:
    """Return the case and which method produced it ("llm:<name>" or "rules")."""
    raw = llm.complete(SYSTEM, redacted_text)
    payload = _parse_json(raw) if raw else None
    if payload is not None:
        try:
            case = CaseReport(**payload)
            if not case.narrative:
                case.narrative = rule_extract(redacted_text).narrative
            return case, f"llm:{llm.name}"
        except ValidationError:
            pass
    return rule_extract(redacted_text), "rules"
