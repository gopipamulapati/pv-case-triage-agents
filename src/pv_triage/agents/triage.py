"""Triage agent: priority and due date from validity, seriousness and expectedness."""

from __future__ import annotations

from datetime import date, timedelta

from ..config import DEFAULT_TIMELINES, Timelines
from ..models import EventExpectedness, Priority, Seriousness, Triage, ValidityCheck


def triage(
    validity: ValidityCheck,
    seriousness: Seriousness | None,
    expectedness: list[EventExpectedness],
    receipt_date: date | None,
    timelines: Timelines = DEFAULT_TIMELINES,
) -> Triage:
    start = receipt_date or date.today()

    if not validity.is_valid:
        return Triage(
            priority=Priority.FOLLOW_UP,
            due_date=start + timedelta(days=timelines.follow_up_days),
            reason=f"Not a valid case yet; missing: {', '.join(validity.missing)}.",
        )

    unlisted = [e.preferred_term for e in expectedness if not e.listed]
    if seriousness and seriousness.is_serious and unlisted:
        return Triage(
            priority=Priority.EXPEDITED,
            due_date=start + timedelta(days=timelines.expedited_days),
            reason=f"Serious ({', '.join(seriousness.criteria)}) and unexpected "
                   f"({', '.join(unlisted)}).",
        )
    if seriousness and seriousness.is_serious:
        return Triage(
            priority=Priority.PRIORITY,
            due_date=start + timedelta(days=timelines.serious_expected_days),
            reason=f"Serious ({', '.join(seriousness.criteria)}) but all events are listed.",
        )
    return Triage(
        priority=Priority.ROUTINE,
        due_date=start + timedelta(days=timelines.non_serious_days),
        reason="Non-serious.",
    )
