"""FastAPI service exposing the triage graph."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .graph import build_graph, triage_report
from .llm import get_llm
from .models import TriageReport

app = FastAPI(
    title="PV Case Triage Agents",
    version=__version__,
    description="Multi-agent adverse event case triage built with LangGraph. "
    "Synthetic data only; not for clinical or regulatory use.",
)

_llm = get_llm()
_graph = build_graph(_llm)


class TriageRequest(BaseModel):
    text: str = Field(min_length=20, description="Raw adverse event report text")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": _llm.name, "version": __version__}


@app.post("/triage", response_model=TriageReport)
def triage_endpoint(req: TriageRequest) -> TriageReport:
    # Only redacted, structured data is returned; raw PHI never leaves the request.
    return triage_report(req.text, graph=_graph)
