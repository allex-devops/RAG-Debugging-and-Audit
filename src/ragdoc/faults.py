"""Bugs planted on purpose, each one a mistake that really happens in RAG code."""
import time
from dataclasses import dataclass
from typing import Callable

from .rig import Config


@dataclass
class Fault:
    name: str
    story: str  # how it happens in real life
    symptom: str  # what a user or a log would show
    caught_by: str  # the probe that should notice
    configure: Callable[[Config], None]


def _query_as_document(cfg: Config) -> None:
    # someone reuses the document-embedding call for questions
    cfg.embed_wrappers.append(lambda inner: (lambda texts, kind="document": inner(texts, "document")))


def _inverted_scores(cfg: Config) -> None:
    # a distance gets used where a similarity is expected
    cfg.score_map = lambda s: 1.0 - s


def _duplicate_ingest(cfg: Config) -> None:
    cfg.ingest_twice = True


def _prompt_truncated(cfg: Config) -> None:
    # a "keep the prompt small" guard that cuts the context instead of choosing less of it
    def wrap(inner):
        class Truncating:
            def chat(self, messages, **kw):
                cut = [dict(m) for m in messages]
                cut[-1]["content"] = cut[-1]["content"][:60]
                return inner.chat(cut, **kw)

        return Truncating()

    cfg.llm_wrappers.append(wrap)


def _tiny_chunks(cfg: Config) -> None:
    cfg.chunk_size, cfg.overlap = 30, 0


def _threshold_too_high(cfg: Config) -> None:
    cfg.min_score = 0.95


def _threshold_too_low(cfg: Config) -> None:
    cfg.min_score = 0.0  # "never say I don't know"


def _slow_embedding(cfg: Config) -> None:
    def wrap(inner):
        def slow(texts, kind="document"):
            time.sleep(0.12)  # a network hop to a remote embedding service
            return inner(texts, kind)

        return slow

    cfg.embed_wrappers.append(wrap)


FAULTS = [
    Fault("query_embedded_as_document", "the question goes through the same call as the documents", "answers get a bit worse everywhere, nothing errors", "query_kind", _query_as_document),
    Fault("inverted_scores", "distance used as similarity", "unrelated questions get confident answers", "declines_when_it_should", _inverted_scores),
    Fault("duplicate_ingest", "re-running ingest adds a second copy of every chunk", "answers repeat themselves, fewer distinct sources", "no_duplicates", _duplicate_ingest),
    Fault("prompt_truncated", "prompt cut to a fixed length", "answers ignore the retrieved text or say it isn't there", "context_intact", _prompt_truncated),
    Fault("tiny_chunks", "chunk size set in words but read as characters", "facts are split across chunks, answers are partial", "context_intact", _tiny_chunks),
    Fault("threshold_too_high", "cutoff copied from a different embedding model", "nearly everything is 'I don't know'", "answers_when_it_should", _threshold_too_high),
    Fault("threshold_too_low", "cutoff removed to stop 'I don't know' complaints", "the model answers off-topic questions from irrelevant context", "declines_when_it_should", _threshold_too_low),
    Fault("slow_embedding", "embedding moved behind a network call", "every request takes longer, quality unchanged", "latency_budget", _slow_embedding),
]

BY_NAME = {f.name: f for f in FAULTS}
