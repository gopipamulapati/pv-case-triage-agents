"""Typed data models shared by the agents."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    EXPEDITED = "expedited"  # serious + unexpected
    PRIORITY = "priority"  # serious + expected
    ROUTINE = "routine"  # non-serious
    FOLLOW_UP = "follow_up_required"  # not a valid case yet


class CaseReport(BaseModel):
    """Structured fields pulled out of a free-text adverse-event report."""

    receipt_date: date | None = None
    patient_age: int | None = Field(default=None, ge=0, le=120)
    patient_sex: str | None = None
    patient_identifiable: bool = False  # initials, age or sex is enough
    reporter_type: str | None = None  # physician, pharmacist, nurse, consumer...
    suspect_drug: str | None = None
    dose: str | None = None
    indication: str | None = None
    narrative: str = ""
    outcome: str | None = None


class CodedEvent(BaseModel):
    verbatim: str
    preferred_term: str
    body_system: str


class Seriousness(BaseModel):
    is_serious: bool
    criteria: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class EventExpectedness(BaseModel):
    preferred_term: str
    listed: bool


class ValidityCheck(BaseModel):
    """The four minimum criteria for a valid individual case safety report."""

    identifiable_patient: bool
    identifiable_reporter: bool
    suspect_drug: bool
    adverse_event: bool

    @property
    def is_valid(self) -> bool:
        return all(
            (self.identifiable_patient, self.identifiable_reporter, self.suspect_drug,
             self.adverse_event)
        )

    @property
    def missing(self) -> list[str]:
        return [name for name, ok in self.model_dump().items() if not ok]


class Triage(BaseModel):
    priority: Priority
    due_date: date | None = None
    reason: str


class TriageReport(BaseModel):
    case: CaseReport
    phi_redacted: dict[str, int]
    validity: ValidityCheck
    events: list[CodedEvent] = Field(default_factory=list)
    seriousness: Seriousness | None = None
    expectedness: list[EventExpectedness] = Field(default_factory=list)
    triage: Triage
    summary: str
    trace: list[dict] = Field(default_factory=list)
