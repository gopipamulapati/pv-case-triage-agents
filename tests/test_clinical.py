from datetime import date

import pytest

from pv_triage.agents.clinical import assess_expectedness, assess_seriousness, code_events
from pv_triage.agents.triage import triage
from pv_triage.graph import drug_identifiable
from pv_triage.models import CaseReport, Priority, ValidityCheck


def case(narrative: str, drug: str = "Cardiolex", outcome: str | None = None) -> CaseReport:
    return CaseReport(narrative=narrative, suspect_drug=drug, outcome=outcome)


def test_coding_maps_synonyms_to_preferred_terms():
    terms = {e.preferred_term for e in code_events("She threw up and felt lightheaded.")}
    assert terms == {"Vomiting", "Dizziness"}


@pytest.mark.parametrize("text", [
    "Patient denies chest pain.",
    "No chest pain was reported.",
    "Presented without chest pain.",
])
def test_negated_events_are_not_coded(text):
    assert code_events(text) == []


def test_negation_does_not_cross_sentences():
    assert [e.preferred_term for e in code_events("No fever. Chest pain began at noon.")] == [
        "Chest pain"]


@pytest.mark.parametrize("narrative,criterion", [
    ("The patient died two days later.", "death"),
    ("The reaction was life-threatening.", "life_threatening"),
    ("She was admitted for observation.", "hospitalization"),
    ("This resulted in permanent disability.", "disability"),
])
def test_seriousness_criteria(narrative, criterion):
    result = assess_seriousness(case(narrative), [])
    assert result.is_serious and criterion in result.criteria
    assert result.evidence  # every criterion records what triggered it


def test_emergency_visit_alone_is_not_hospitalization():
    s = assess_seriousness(case("Seen in the emergency department and discharged."), [])
    assert not s.is_serious


def test_negated_hospitalization_is_not_serious():
    s = assess_seriousness(case("She was not admitted. No hospitalization required."), [])
    assert not s.is_serious


def test_important_medical_event_makes_case_serious():
    events = code_events("Patient had a seizure.")
    s = assess_seriousness(case("Patient had a seizure."), events)
    assert s.criteria == ["medically_important"]


def test_fatal_outcome_field_counts_as_death():
    assert "death" in assess_seriousness(case("Unwell.", outcome="fatal"), []).criteria


def test_expectedness_uses_the_product_label():
    events = code_events("Persistent cough and a rash.")
    result = {e.preferred_term: e.listed for e in assess_expectedness(case(""), events)}
    assert result == {"Cough": True, "Rash": False}  # Cardiolex lists cough, not rash


def test_unknown_product_is_treated_as_unlisted():
    events = code_events("Headache.")
    assert not assess_expectedness(case("", drug="Novaxil"), events)[0].listed


@pytest.mark.parametrize("name,ok", [
    ("Novaxil 50 mg", True),
    ("Cardiolex", True),
    ('"my new blood pressure pill"', False),
    ("the tablet", False),
    (None, False),
])
def test_drug_must_be_named(name, ok):
    assert drug_identifiable(name) is ok


def _valid():
    return ValidityCheck(identifiable_patient=True, identifiable_reporter=True,
                         suspect_drug=True, adverse_event=True)


def test_triage_due_dates():
    received = date(2026, 9, 1)
    serious = assess_seriousness(case("She was admitted."), [])
    rash = assess_expectedness(case(""), code_events("rash"))
    cough = assess_expectedness(case(""), code_events("cough"))

    expedited = triage(_valid(), serious, rash, received)
    assert expedited.priority == Priority.EXPEDITED
    assert expedited.due_date == date(2026, 9, 16)  # 15 calendar days

    assert triage(_valid(), serious, cough, received).priority == Priority.PRIORITY
    not_serious = assess_seriousness(case("Mild."), [])
    assert triage(_valid(), not_serious, rash, received).priority == Priority.ROUTINE


def test_invalid_case_needs_follow_up():
    invalid = ValidityCheck(identifiable_patient=True, identifiable_reporter=True,
                            suspect_drug=False, adverse_event=True)
    t = triage(invalid, None, [], date(2026, 9, 1))
    assert t.priority == Priority.FOLLOW_UP and "suspect_drug" in t.reason
