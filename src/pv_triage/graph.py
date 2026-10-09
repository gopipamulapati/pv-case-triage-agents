"""LangGraph orchestration of the case triage agents.

    redact_phi -> extract -> validate --invalid--> triage -> write
                                 | valid
                                 v
                            code_events
                           /           \\
                  seriousness       expectedness     (parallel)
                           \\           /
                              triage -> write -> END
"""

from __future__ import annotations

import operator
import re
import time
from collections.abc import Callable
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import clinical, extractor, phi, writer
from .agents.triage import triage as triage_case
from .llm import LLM, get_llm
from .models import (
    CaseReport,
    CodedEvent,
    EventExpectedness,
    Seriousness,
    Triage,
    TriageReport,
    ValidityCheck,
)


class CaseState(TypedDict, total=False):
    raw_text: str
    redacted_text: str
    phi_counts: dict[str, int]
    phi_removed: list[str]  # kept in memory for the leak check; never returned
    case: CaseReport
    extraction_method: str
    validity: ValidityCheck
    events: list[CodedEvent]
    seriousness: Seriousness
    expectedness: list[EventExpectedness]
    triage: Triage
    summary: str
    summary_source: str
    trace: Annotated[list[dict], operator.add]


def _traced(name: str, fn: Callable[[CaseState], dict]) -> Callable[[CaseState], dict]:
    def node(state: CaseState) -> dict:
        start = time.perf_counter()
        update = fn(state)
        update["trace"] = [{"node": name, "ms": round((time.perf_counter() - start) * 1000, 2)}]
        return update

    return node


_REPORTER_LINE = re.compile(r"(?im)^\s*reporter\b")
_VAGUE_DRUG = re.compile(r"(?i)\b(my|the|new|some|pill|tablet|medicine|medication|drug)\b")


def drug_identifiable(name: str | None) -> bool:
    """A suspect product counts only if it is named, e.g. "Novaxil 50 mg",
    not described, e.g. "my new blood pressure pill"."""
    if not name:
        return False
    name = name.strip().strip('"\'')
    return bool(re.match(r"[A-Z][A-Za-z0-9-]{2,}", name)) and not _VAGUE_DRUG.search(name)


def build_graph(llm: LLM | None = None):
    llm = llm or get_llm()

    def redact_node(state: CaseState) -> dict:
        text, counts, removed = phi.redact(state["raw_text"])
        return {"redacted_text": text, "phi_counts": counts, "phi_removed": removed}

    def extract_node(state: CaseState) -> dict:
        case, method = extractor.extract(state["redacted_text"], llm)
        return {"case": case, "extraction_method": method}

    def validate_node(state: CaseState) -> dict:
        case = state["case"]
        events = clinical.code_events(case.narrative)
        validity = ValidityCheck(
            identifiable_patient=case.patient_identifiable,
            identifiable_reporter=bool(case.reporter_type)
            or bool(_REPORTER_LINE.search(state["redacted_text"])),
            suspect_drug=drug_identifiable(case.suspect_drug),
            adverse_event=bool(events),
        )
        return {"validity": validity}

    def code_node(state: CaseState) -> dict:
        return {"events": clinical.code_events(state["case"].narrative)}

    def seriousness_node(state: CaseState) -> dict:
        return {"seriousness": clinical.assess_seriousness(state["case"], state["events"])}

    def expectedness_node(state: CaseState) -> dict:
        return {"expectedness": clinical.assess_expectedness(state["case"], state["events"])}

    def triage_node(state: CaseState) -> dict:
        return {
            "triage": triage_case(
                state["validity"],
                state.get("seriousness"),
                state.get("expectedness", []),
                state["case"].receipt_date,
            )
        }

    def write_node(state: CaseState) -> dict:
        summary, source = writer.write(
            state["case"],
            state["validity"],
            state.get("events", []),
            state.get("seriousness"),
            state.get("expectedness", []),
            state["triage"],
            state["phi_removed"],
            llm,
        )
        return {"summary": summary, "summary_source": source}

    def route_after_validate(state: CaseState) -> str:
        return "code_events" if state["validity"].is_valid else "triage"

    g = StateGraph(CaseState)
    for name, fn in [
        ("redact_phi", redact_node),
        ("extract", extract_node),
        ("validate", validate_node),
        ("code_events", code_node),
        ("seriousness", seriousness_node),
        ("expectedness", expectedness_node),
        ("triage", triage_node),
        ("write", write_node),
    ]:
        g.add_node(name, _traced(name, fn))

    g.add_edge(START, "redact_phi")
    g.add_edge("redact_phi", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", route_after_validate, ["code_events", "triage"])
    g.add_edge("code_events", "seriousness")
    g.add_edge("code_events", "expectedness")
    g.add_edge(["seriousness", "expectedness"], "triage")
    g.add_edge("triage", "write")
    g.add_edge("write", END)
    return g.compile()


def triage_report(text: str, llm: LLM | None = None, graph=None) -> TriageReport:
    """Run one report through the graph. Pass a compiled ``graph`` to reuse it."""
    graph = graph or build_graph(llm)
    s = graph.invoke({"raw_text": text, "trace": []})
    return TriageReport(
        case=s["case"],
        phi_redacted=s["phi_counts"],
        validity=s["validity"],
        events=s.get("events", []),
        seriousness=s.get("seriousness"),
        expectedness=s.get("expectedness", []),
        triage=s["triage"],
        summary=s["summary"],
        trace=s["trace"],
    )
