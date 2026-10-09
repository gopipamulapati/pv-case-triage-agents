"""Processing timelines in calendar days from the receipt date.

The 15-day expedited window mirrors the common regulatory convention for
serious, unexpected reports. The other values are example internal targets.
This is a portfolio project, not regulatory guidance.
"""

from pydantic import BaseModel


class Timelines(BaseModel):
    expedited_days: int = 15
    serious_expected_days: int = 30
    non_serious_days: int = 90
    follow_up_days: int = 7


DEFAULT_TIMELINES = Timelines()
