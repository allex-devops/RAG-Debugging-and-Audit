import pytest
from ragchat.rag import answer
from ragchat.store import EmbeddingMismatch, Store

from ragdoc.incident import bow_embed_v2, measure, reproduce, small_embed
from ragdoc.rig import bow_embed


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return reproduce(tmp_path_factory.mktemp("incident"))


def test_a_healthy_index_answers_everything_and_ranks_the_right_document_first(result):
    assert result["healthy"] == {"questions": 8, "right_document_first": 8, "answered": 8}


def test_swapping_models_without_reindexing_degrades_to_roughly_chance_without_any_error(result):
    after = result["after_model_swap"]
    # eight documents, so a random pick gets about one right; what matters is that nothing raised
    assert after["right_document_first"] <= 3
    assert after["answered"] <= 3


def test_the_two_models_really_do_share_a_vector_size():
    assert bow_embed(["cats purr"]).shape == bow_embed_v2(["cats purr"]).shape == (1, 256)
    assert small_embed(["cats purr"]).shape == (1, 128)


def test_a_different_vector_size_fails_loudly_in_the_database(result):
    assert "dimension" in result["raw_dimension_error"]


def test_the_guard_refuses_the_other_model_and_says_what_to_do(result):
    assert "built with 'model-a'" in result["guard"] and "re-index" in result["guard"]


def test_the_healthy_measurement_notices_when_things_go_wrong(tmp_path):
    # sanity check on measure() itself: it has to be able to report a bad result
    from ragdoc.rig import Config, Rig

    Rig(Config(), tmp_path)
    good = measure(tmp_path / "index", bow_embed)
    bad = measure(tmp_path / "index", bow_embed_v2)
    assert good["right_document_first"] > bad["right_document_first"]
