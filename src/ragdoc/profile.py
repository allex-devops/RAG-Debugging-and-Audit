"""Where does the time go? Runs questions through a RAG and splits each request into stages."""
from dataclasses import dataclass
from statistics import mean, quantiles
from typing import Callable

STAGES = ("embed_ms", "search_ms", "llm_ms")


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    cuts = quantiles(values, n=100, method="inclusive")
    return cuts[min(max(pct, 1), 99) - 1]


@dataclass
class StageStats:
    name: str
    mean: float
    p50: float
    p95: float
    share: float  # fraction of all time spent in this stage


@dataclass
class Profile:
    requests: int
    declined: int
    stages: list[StageStats]

    @property
    def bottleneck(self) -> StageStats:
        return max(self.stages, key=lambda s: s.share)

    def report(self) -> str:
        lines = [f"{self.requests} requests, {self.declined} declined without calling the model", ""]
        lines.append(f"{'stage':10} {'mean':>9} {'p50':>9} {'p95':>9} {'share':>7}")
        for s in self.stages:
            lines.append(f"{s.name.removesuffix('_ms'):10} {s.mean:8.1f}m {s.p50:8.1f}m {s.p95:8.1f}m {s.share:6.0%}")
        b = self.bottleneck
        lines += ["", f"bottleneck: {b.name.removesuffix('_ms')} ({b.share:.0%} of request time)"]
        return "\n".join(lines)


def profile(ask: Callable, questions: list[str], repeat: int = 1) -> Profile:
    """`ask(question, trace=...)` is anything shaped like ragchat.rag.answer."""
    samples = {s: [] for s in STAGES}
    declined = 0
    for q in questions * repeat:
        seen: dict = {}

        def trace(event, fields, seen=seen):
            seen.setdefault("declined", False)
            if event == "decline":
                seen["declined"] = True
            seen.update({k: v for k, v in fields.items() if k in STAGES})

        ask(q, trace=trace)
        declined += bool(seen.get("declined"))
        for stage in STAGES:
            samples[stage].append(seen.get(stage, 0.0))

    total = sum(sum(v) for v in samples.values()) or 1.0
    stats = [StageStats(s, mean(v), percentile(v, 50), percentile(v, 95), sum(v) / total) for s, v in samples.items()]
    return Profile(len(questions) * repeat, declined, stats)
