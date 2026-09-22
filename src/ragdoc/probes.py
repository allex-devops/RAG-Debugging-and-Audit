"""Checks that each look at one thing a RAG can get wrong. A failing probe points at a cause."""
import time
from dataclasses import dataclass
from statistics import quantiles

from .corpus import ANSWERABLE, UNANSWERABLE
from .rig import Rig

PASS_RATE = 0.9
# generous on purpose: this runs on a laptop that may be busy with a model, and a flaky probe is worse than none
LATENCY_BUDGET_MS = 60.0  # the default for Config.latency_budget_ms


@dataclass
class ProbeResult:
    name: str
    passed: bool
    detail: str


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags)


def hit_rate(rig: Rig) -> ProbeResult:
    """Does retrieval put the right document among the results?"""
    found = [any(h.source == src for h in rig.retrieve(q)) for q, src, _ in ANSWERABLE]
    return ProbeResult("hit_rate", _rate(found) >= PASS_RATE, f"{sum(found)}/{len(found)} questions retrieved their source")


def answers_when_it_should(rig: Rig) -> ProbeResult:
    grounded = [rig.ask(q).grounded for q, _, _ in ANSWERABLE]
    return ProbeResult(
        "answers_when_it_should", _rate(grounded) >= PASS_RATE, f"{sum(grounded)}/{len(grounded)} answerable questions were answered"
    )


def declines_when_it_should(rig: Rig) -> ProbeResult:
    """Unrelated questions must get "I don't know" without the model being called."""
    declined = []
    for q in UNANSWERABLE:
        before = len(rig.model_spy.prompts)
        a = rig.ask(q)
        declined.append(not a.grounded and len(rig.model_spy.prompts) == before)
    return ProbeResult(
        "declines_when_it_should", _rate(declined) >= PASS_RATE, f"{sum(declined)}/{len(declined)} unrelated questions were declined"
    )


def no_duplicates(rig: Rig) -> ProbeResult:
    """Repeated chunks crowd real results out of the top k."""
    repeats = 0
    for q, _, _ in ANSWERABLE:
        texts = [h.text for h in rig.retrieve(q)]
        repeats += len(texts) - len(set(texts))
    return ProbeResult("no_duplicates", repeats == 0, f"{repeats} repeated chunks across the top-{rig.cfg.k} results")


def query_kind(rig: Rig) -> ProbeResult:
    """Questions should be embedded as queries. Embedding them as documents quietly hurts retrieval."""
    rig.embed_spy.calls.clear()
    for q, _, _ in ANSWERABLE:
        rig.ask(q)
    kinds = sorted(set(rig.embed_spy.calls))
    return ProbeResult("query_kind", kinds == ["query"], f"questions were embedded as: {', '.join(kinds)}")


def context_intact(rig: Rig) -> ProbeResult:
    """The passage holding the answer has to reach the model whole."""
    intact = []
    for q, _, fact in ANSWERABLE:
        before = len(rig.model_spy.prompts)
        rig.ask(q)
        sent = rig.model_spy.prompts[before:]
        intact.append(bool(sent) and fact in sent[-1][-1]["content"])
    return ProbeResult(
        "context_intact", _rate(intact) >= PASS_RATE, f"the answer text reached the model for {sum(intact)}/{len(intact)} questions"
    )


def latency_budget(rig: Rig, budget_ms: float | None = None) -> ProbeResult:
    """p95 time to answer, and which stage is eating it."""
    budget_ms = budget_ms or rig.cfg.latency_budget_ms
    totals, stages = [], {"embed_ms": [], "search_ms": [], "llm_ms": []}
    for q, _, _ in ANSWERABLE * 3:
        seen: dict = {}
        t0 = time.perf_counter()
        rig.ask(q, trace=lambda event, fields: seen.update({k: v for k, v in fields.items() if k in stages}))
        totals.append((time.perf_counter() - t0) * 1000)
        for name in stages:
            stages[name].append(seen.get(name, 0.0))
    p95 = quantiles(totals, n=20)[-1]
    slowest = max(stages, key=lambda s: sum(stages[s]))
    return ProbeResult("latency_budget", p95 <= budget_ms, f"p95 {p95:.0f} ms against a {budget_ms:.0f} ms budget, mostly in {slowest}")


ALL_PROBES = [hit_rate, answers_when_it_should, declines_when_it_should, no_duplicates, query_kind, context_intact, latency_budget]

# what a failing probe usually means, in the order to look
LIKELY_CAUSES = {
    "hit_rate": "retrieval is missing the right documents: wrong embedding model or prefix, bad chunking, or an index built with different settings",
    "answers_when_it_should": "the similarity cutoff is too strict, or scores are being computed the wrong way round",
    "declines_when_it_should": "the cutoff is too loose or scores are inverted, so the model gets irrelevant context and will make things up",
    "no_duplicates": "the ingest job is not idempotent: chunk ids change between runs, so re-ingesting adds copies",
    "query_kind": "questions are embedded with the document prefix; with models that use prefixes this lowers retrieval quality",
    "context_intact": "the prompt is being cut short, or chunks are so small that an answer is split across them",
    "latency_budget": "one stage dominates; see the profile for which",
}


def run_all(rig: Rig) -> list[ProbeResult]:
    return [probe(rig) for probe in ALL_PROBES]
