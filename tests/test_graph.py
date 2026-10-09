import json
from pathlib import Path

import pytest

from pv_triage.agents.phi import redact
from pv_triage.graph import triage_report
from pv_triage.llm import OfflineLLM

GOLD = json.loads((Path(__file__).resolve().parent.parent / "samples" / "gold_labels.json")
                  .read_text())


@pytest.mark.parametrize("name", sorted(GOLD))
def test_samples_match_gold_labels(sample, name):
    report = triage_report(sample(name), OfflineLLM())
    want = GOLD[name]
    assert report.triage.priority.value == want["priority"]
    assert {e.preferred_term for e in report.events} == set(want["events"])


def test_invalid_case_skips_clinical_agents(sample):
    report = triage_report(sample("04_incomplete_consumer.txt"), OfflineLLM())
    assert [t["node"] for t in report.trace] == [
        "redact_phi", "extract", "validate", "triage", "write"]


def test_parallel_branches_both_run(sample):
    nodes = [t["node"] for t in triage_report(sample("01_angioedema_hospitalized.txt"),
                                              OfflineLLM()).trace]
    assert {"seriousness", "expectedness"} <= set(nodes)
    assert nodes.index("code_events") < nodes.index("triage")


def test_llm_never_sees_raw_phi(sample, scripted_llm):
    text = sample("02_liver_failure_unexpected.txt")
    _, _, removed = redact(text)
    llm = scripted_llm()
    triage_report(text, llm)
    assert llm.inputs, "the LLM should have been called"
    for seen in llm.inputs:
        for value in removed:
            assert value not in seen


def test_report_output_contains_no_phi(sample):
    text = sample("01_angioedema_hospitalized.txt")
    _, _, removed = redact(text)
    dumped = triage_report(text, OfflineLLM()).model_dump_json()
    assert not [v for v in removed if v in dumped]


def test_valid_llm_extraction_is_used(sample, scripted_llm):
    payload = {"receipt_date": "2026-09-14", "patient_age": 67, "patient_sex": "female",
               "patient_identifiable": True, "reporter_type": "physician",
               "suspect_drug": "Cardiolex", "narrative": "She developed hives."}
    llm = scripted_llm({"extract": json.dumps(payload)})
    report = triage_report(sample("01_angioedema_hospitalized.txt"), llm)
    assert [e.preferred_term for e in report.events] == ["Urticaria"]


def test_invalid_llm_extraction_falls_back_to_rules(sample, scripted_llm):
    llm = scripted_llm({"extract": '{"patient_age": 400}'})  # fails validation
    report = triage_report(sample("01_angioedema_hospitalized.txt"), llm)
    assert report.case.patient_age == 67


def test_writer_blocks_phi_leak(sample, scripted_llm):
    llm = scripted_llm({"case summaries": "PRIORITY case for Maria Delgado, MRN 00482913."})
    report = triage_report(sample("01_angioedema_hospitalized.txt"), llm)
    assert "Maria Delgado" not in report.summary
    assert report.summary.startswith("## Case triage: PRIORITY")  # template used


def test_writer_blocks_changed_priority(sample, scripted_llm):
    llm = scripted_llm({"case summaries": "Routine case, nothing to do."})
    report = triage_report(sample("02_liver_failure_unexpected.txt"), llm)
    assert "EXPEDITED" in report.summary


def test_writer_accepts_faithful_rewrite(sample, scripted_llm):
    llm = scripted_llm({"case summaries": "EXPEDITED: serious unlisted hepatic failure."})
    report = triage_report(sample("02_liver_failure_unexpected.txt"), llm)
    assert report.summary == "EXPEDITED: serious unlisted hepatic failure."
