"""Reference data: the event vocabulary and the fictional product labels."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache
def event_terms() -> list[dict]:
    return json.loads((DATA_DIR / "event_dictionary.json").read_text())["terms"]


@lru_cache
def products() -> dict[str, dict]:
    return json.loads((DATA_DIR / "product_labels.json").read_text())["products"]


def find_product(text: str | None) -> str | None:
    """Return the product key mentioned in ``text``, if any."""
    if not text:
        return None
    lowered = text.lower()
    for key in products():
        if key in lowered:
            return key
    return None


# Events a safety team treats as medically important even without another
# seriousness criterion. Companies maintain their own list; this one is a
# short illustrative example.
IMPORTANT_MEDICAL_EVENTS = frozenset(
    {
        "Anaphylactic reaction",
        "Stevens-Johnson syndrome",
        "Hepatic failure",
        "Seizure",
        "Acute kidney injury",
        "Lactic acidosis",
        "QT prolongation",
    }
)
