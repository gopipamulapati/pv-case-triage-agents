"""PHI redaction: runs first, so no model ever sees patient-identifying data.

Rule-based on purpose. Redaction has to be predictable and testable, and it
must run before any text is sent to a hosted LLM. Ages and sex are kept
because they are needed for the case and do not identify a person on their own.
"""

from __future__ import annotations

import re

# (label, pattern). Order matters: specific patterns run before general ones.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("MRN", re.compile(r"(?i)\b(?:MRN|medical record(?: number)?)[:#\s]*[A-Z0-9-]{5,}\b")),
    ("DOB", re.compile(r"(?i)\b(?:DOB|date of birth)[:\s]*\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("ADDRESS", re.compile(
        r"\b\d{1,5}[ \t]+(?:[A-Z][a-z]+[ \t]){1,3}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b"
    )),
    # Names after a field label, e.g. "Patient name: Jane Doe"
    ("NAME", re.compile(
        r"(?im)^(?P<label>(?:patient|reporter)(?: name)?:[ \t]*)"
        r"(?P<value>[A-Z][a-z]+(?:[ \t][A-Z][a-z.]+)+)"
    )),
    # Doctor names in free text, e.g. "Dr. Alan Smith"
    ("NAME", re.compile(r"\bDr\.?[ \t]+[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?")),
]


_LABEL_PREFIX = re.compile(r"(?i)^(?:MRN|medical record(?: number)?|DOB|date of birth)[:#\s]*")


def redact(text: str) -> tuple[str, dict[str, int], list[str]]:
    """Return (redacted text, count per PHI type, the original values removed).

    The removed values are kept only in memory so the output guardrail can
    check that none of them reappear in generated text.
    """
    counts: dict[str, int] = {}
    removed: list[str] = []

    for label, pattern in PATTERNS:
        def _sub(m: re.Match[str], label: str = label) -> str:
            counts[label] = counts.get(label, 0) + 1
            if "value" in m.groupdict() and m.group("value"):
                removed.append(m.group("value"))
                return f"{m.group('label')}[{label}]"
            # Keep the identifier itself (e.g. "00482913", not "MRN: 00482913")
            # so the leak check catches it even without its label.
            removed.append(_LABEL_PREFIX.sub("", m.group(0)))
            return f"[{label}]"

        text = pattern.sub(_sub, text)
    return text, counts, removed


def leaks(text: str, removed: list[str]) -> list[str]:
    """PHI values that appear in ``text``. Used to block leaky model output."""
    lowered = text.lower()
    return [v for v in removed if v.lower() in lowered]
