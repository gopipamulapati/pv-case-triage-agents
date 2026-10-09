"""Command-line entry point: ``pv-triage samples/*.txt``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import build_graph, triage_report
from .llm import get_llm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Triage adverse event reports.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", help="Print the full JSON report")
    p.add_argument("--provider", help="offline | openai | bedrock (default: $LLM_PROVIDER)")
    args = p.parse_args(argv)

    graph = build_graph(get_llm(args.provider))
    for path in args.files:
        report = triage_report(path.read_text(encoding="utf-8"), graph=graph)
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2))
            continue
        print(f"\n=== {path.name} -> {report.triage.priority.value} ===\n")
        print(report.summary)
        if report.phi_redacted:
            redacted = ", ".join(f"{k}x{v}" for k, v in sorted(report.phi_redacted.items()))
            print(f"\nPHI redacted before processing: {redacted}")
        print("trace: " + " -> ".join(t["node"] for t in report.trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
