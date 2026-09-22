import json

import pytest

from ragdoc.logs import RequestLog, summarize
from ragdoc.profile import percentile, profile
from ragdoc.rig import Config, Rig


def fake_ask(timings):
    """timings: list of (embed, search, llm) per call; llm None means the request was declined."""
    it = iter(timings)

    def ask(question, trace):
        embed, search, llm = next(it)
        trace("retrieve", {"embed_ms": embed, "search_ms": search})
        if llm is None:
            trace("decline", {"reason": "x", "min_score": 0.4})
        else:
            trace("generate", {"llm_ms": llm})

    return ask


def test_profile_splits_time_by_stage_and_names_the_bottleneck():
    p = profile(fake_ask([(10, 5, 85), (10, 5, 85)]), ["q1", "q2"])
    shares = {s.name: round(s.share, 2) for s in p.stages}
    assert shares == {"embed_ms": 0.1, "search_ms": 0.05, "llm_ms": 0.85}
    assert p.bottleneck.name == "llm_ms" and p.requests == 2 and p.declined == 0


def test_declined_requests_count_and_contribute_no_llm_time():
    p = profile(fake_ask([(10, 5, None), (10, 5, 100)]), ["a", "b"])
    assert p.declined == 1
    assert next(s for s in p.stages if s.name == "llm_ms").mean == 50


def test_repeat_multiplies_the_requests():
    p = profile(fake_ask([(1, 1, 1)] * 6), ["a", "b"], repeat=3)
    assert p.requests == 6


def test_report_is_readable():
    text = profile(fake_ask([(10, 5, 85)]), ["q"]).report()
    assert "bottleneck: llm (85% of request time)" in text and "1 requests" in text


def test_percentiles():
    assert percentile([], 95) == 0.0
    assert percentile([7], 95) == 7
    assert percentile(list(range(1, 101)), 50) == pytest.approx(50.5)
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def write_log(path, requests):
    """requests: list of dicts with top_score, declined, ms"""
    with open(path, "w") as f:
        for i, r in enumerate(requests):
            rid = f"r{i}"
            f.write(json.dumps({"rid": rid, "event": "retrieve", "question": f"question {i}", "top_score": r["top"], "embed_ms": r["ms"], "search_ms": 0}) + "\n")
            if r["declined"]:
                f.write(json.dumps({"rid": rid, "event": "decline", "reason": "x", "min_score": 0.4}) + "\n")
            else:
                f.write(json.dumps({"rid": rid, "event": "generate", "llm_ms": 0}) + "\n")


def test_summary_counts_declines_and_near_misses(tmp_path):
    write_log(tmp_path / "l.jsonl", [
        {"top": 0.9, "declined": False, "ms": 10},
        {"top": 0.35, "declined": True, "ms": 10},   # just under 0.4: a near miss
        {"top": 0.05, "declined": True, "ms": 10},   # clearly unrelated
        {"top": None, "declined": True, "ms": 10},   # nothing in the index
    ])
    s = summarize(tmp_path / "l.jsonl")
    assert (s.requests, s.declined, s.near_misses, s.empty_index) == (4, 3, 1, 1)
    assert any("empty index" in n for n in s.notes)


def test_summary_flags_a_cutoff_that_is_probably_too_strict(tmp_path):
    write_log(tmp_path / "l.jsonl", [{"top": 0.36, "declined": True, "ms": 5}] * 4 + [{"top": 0.9, "declined": False, "ms": 5}] * 2)
    notes = summarize(tmp_path / "l.jsonl").notes
    assert any("just under the cutoff" in n for n in notes)
    assert any("more than half" in n for n in notes)


def test_summary_finds_the_slowest_requests(tmp_path):
    write_log(tmp_path / "l.jsonl", [{"top": 0.9, "declined": False, "ms": m} for m in (10, 900, 30, 400)])
    s = summarize(tmp_path / "l.jsonl")
    assert [q for q, _ in s.slowest] == ["question 1", "question 3", "question 2"]


def test_a_healthy_log_has_no_complaints(tmp_path):
    write_log(tmp_path / "l.jsonl", [{"top": 0.9, "declined": False, "ms": 5}] * 5 + [{"top": 0.02, "declined": True, "ms": 5}] * 3)
    assert summarize(tmp_path / "l.jsonl").notes == []


def test_real_requests_leave_a_log_that_summarises(tmp_path):
    rig = Rig(Config(), tmp_path / "rig")
    log = RequestLog(tmp_path / "logs" / "requests.jsonl")
    for q in ("what frequency do cats purr at", "who won the football world cup"):
        rig.ask(q, trace=log.tracer(q))
    s = summarize(log.path)
    assert (s.requests, s.declined) == (2, 1)
    lines = [json.loads(l) for l in log.path.read_text().splitlines()]
    assert sum("question" in l for l in lines) == 2  # only the retrieve line carries the question
