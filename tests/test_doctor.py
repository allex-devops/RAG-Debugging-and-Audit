import pytest

from ragdoc.cli import main
from ragdoc.faults import BY_NAME, FAULTS
from ragdoc.probes import ALL_PROBES, LIKELY_CAUSES, latency_budget, run_all
from ragdoc.rig import Config, Rig


def rig_with(tmp_path, *fault_names):
    cfg = Config()
    for name in fault_names:
        BY_NAME[name].configure(cfg)
    return Rig(cfg, tmp_path)


def test_a_healthy_rag_passes_every_probe(tmp_path):
    results = run_all(rig_with(tmp_path))
    assert [r.name for r in results if not r.passed] == []


@pytest.mark.parametrize("fault", FAULTS, ids=lambda f: f.name)
def test_each_planted_bug_is_caught_by_the_probe_built_for_it(tmp_path, fault):
    rig = rig_with(tmp_path, fault.name)
    if fault.caught_by == "latency_budget":
        results = [latency_budget(rig)]  # the slow fault makes every other probe slow too, so run just this one
    else:
        # the timing probe is left out here: it depends on how busy the machine is, not on the bug
        results = [p(rig) for p in ALL_PROBES if p is not latency_budget]
    failed = {r.name for r in results if not r.passed}
    assert fault.caught_by in failed, f"{fault.name} slipped past {fault.caught_by}: {results}"


def test_every_probe_has_an_explanation_and_every_fault_names_a_real_probe():
    names = {p.__name__ for p in ALL_PROBES}
    assert set(LIKELY_CAUSES) == names
    assert {f.caught_by for f in FAULTS} <= names


def test_a_fault_changes_the_symptom_not_just_the_label(tmp_path):
    good = rig_with(tmp_path / "a")
    bad = rig_with(tmp_path / "b", "inverted_scores")
    q = "who won the football world cup"
    assert not good.ask(q).grounded
    assert bad.ask(q).grounded  # the bug: an unrelated question now gets a confident answer


def test_the_prompt_truncation_bug_really_cuts_what_the_model_sees(tmp_path):
    rig = rig_with(tmp_path, "prompt_truncated")
    rig.ask("what frequency do cats purr at")
    assert len(rig.model_spy.prompts[-1][-1]["content"]) == 60


def test_duplicates_bug_really_doubles_the_index(tmp_path):
    q = "how do honeybees share food locations"
    good = [h.text for h in rig_with(tmp_path / "a").retrieve(q)]
    bad = [h.text for h in rig_with(tmp_path / "b", "duplicate_ingest").retrieve(q)]
    assert len(set(good)) == len(good)
    assert len(set(bad)) < len(bad)


def test_cli_exit_code_reflects_health(tmp_path, capsys):
    with pytest.raises(SystemExit) as ok:
        main(["doctor"])
    assert ok.value.code == 0
    with pytest.raises(SystemExit) as bad:
        main(["doctor", "--fault", "inverted_scores"])
    assert bad.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL declines_when_it_should" in out and "likely:" in out
