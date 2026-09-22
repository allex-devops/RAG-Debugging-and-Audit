"""Per-request logs and a summary of them: the first place to look when users say answers got worse."""
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .profile import percentile


class RequestLog:
    """Writes one JSON line per pipeline stage, tagged with a request id so stages can be put back together."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def tracer(self, question: str):
        rid = uuid.uuid4().hex[:8]

        def trace(event: str, fields: dict) -> None:
            row = {"ts": round(time.time(), 3), "rid": rid, "event": event, **fields}
            if event == "retrieve":
                row["question"] = question[:200]
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        return trace


@dataclass
class Summary:
    requests: int
    declined: int
    near_misses: int  # declined, but with a top score just under the cutoff
    empty_index: int
    slowest: list[tuple[str, float]]  # (question, total ms)
    p95_ms: float
    notes: list[str]

    def report(self) -> str:
        rate = self.declined / self.requests if self.requests else 0
        lines = [
            f"{self.requests} requests, {self.declined} declined ({rate:.0%}), p95 {self.p95_ms:.0f} ms",
            f"declined just under the cutoff: {self.near_misses}; declined because the index was empty: {self.empty_index}",
        ]
        if self.slowest:
            lines.append("slowest: " + "; ".join(f"{q[:40]!r} {ms:.0f} ms" for q, ms in self.slowest))
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def summarize(path: Path, near: float = 0.1) -> Summary:
    by_request: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_request[row["rid"]].append(row)

    declined = near_misses = empty = 0
    timings = []
    for rid, rows in by_request.items():
        ev = {r["event"]: r for r in rows}
        total = sum(r.get(k, 0) for r in rows for k in ("embed_ms", "search_ms", "llm_ms"))
        timings.append((ev.get("retrieve", {}).get("question", rid), total))
        if "decline" in ev:
            declined += 1
            top, cutoff = ev.get("retrieve", {}).get("top_score"), ev["decline"]["min_score"]
            if top is None:
                empty += 1
            elif cutoff - top <= near:
                near_misses += 1

    n = len(by_request)
    notes = []
    if empty:
        notes.append(f"{empty} requests hit an empty index: ingest may have failed or pointed at the wrong folder")
    if declined and near_misses / declined >= 0.3:
        notes.append("many declined questions scored just under the cutoff: it may be too strict, so look at those questions")
    if n and declined / n >= 0.5:
        notes.append("more than half of requests were declined: check the cutoff and that the index matches the embedding model")
    if n >= 20 and declined == 0:
        notes.append("nothing was ever declined: with unrelated questions in the mix that suggests the cutoff is too loose")
    return Summary(
        n, declined, near_misses, empty,
        sorted(timings, key=lambda t: -t[1])[:3],
        percentile([t for _, t in timings], 95),
        notes,
    )
