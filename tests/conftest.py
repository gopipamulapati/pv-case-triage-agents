from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def sample():
    def _load(name: str) -> str:
        return (SAMPLES / name).read_text(encoding="utf-8")

    return _load


class ScriptedLLM:
    """Returns canned replies by prompt keyword and records every input it sees."""

    name = "scripted"

    def __init__(self, replies: dict[str, str] | None = None):
        self.replies = replies or {}
        self.inputs: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.inputs.append(user)
        for key, reply in self.replies.items():
            if key in system:
                return reply
        return ""


@pytest.fixture
def scripted_llm():
    return ScriptedLLM
