"""Score the pipeline against hand-labelled synthetic cases.

Usage:  python scripts/evaluate.py

samples/gold_labels.json holds the expected triage priority, seriousness and
coded events for each sample report. This is a small regression set written
alongside the rules, not an independent benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pv_triage.graph import build_graph, triage_report  # noqa: E402
from pv_triage.llm import OfflineLLM  # noqa: E402


def main() -> int:
    gold = json.loads((ROOT / "samples" / "gold_labels.json").read_text())
    graph = build_graph(OfflineLLM())
    tp = fp = fn = 0
    priority_ok = serious_ok = serious_n = 0
    print(f"{'case':<36}{'expected':<20}{'got':<20}")
    for name, want in gold.items():
        r = triage_report((ROOT / "samples" / name).read_text(), graph=graph)
        got = r.triage.priority.value
        priority_ok += got == want["priority"]
        if want["serious"] is not None:
            serious_n += 1
            serious_ok += bool(r.seriousness and r.seriousness.is_serious) == want["serious"]
        found = {e.preferred_term for e in r.events}
        expected = set(want["events"])
        tp += len(found & expected)
        fp += len(found - expected)
        fn += len(expected - found)
        mark = "" if got == want["priority"] else "  <-- mismatch"
        print(f"{name:<36}{want['priority']:<20}{got:<20}{mark}")

    n = len(gold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    print(f"\ntriage priority accuracy: {priority_ok}/{n}")
    print(f"seriousness accuracy:     {serious_ok}/{serious_n}")
    print(f"event coding precision:   {precision:.0%}   recall: {recall:.0%}")
    return 0 if priority_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
