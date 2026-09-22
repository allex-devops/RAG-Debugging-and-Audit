"""A small RAG built from ragchat's own parts, with hooks so faults can be planted and probes can look inside."""
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from ragchat.ingest import Chunk
from ragchat.rag import Answer, answer
from ragchat.store import Hit, Store
from shared.docs import chunk_text

from .corpus import DOCS

STOPWORDS = set("a an and are as at be by do does for how in is it of on or the to what when who why with that this from".split())


def bow_embed(texts: list[str], kind: str = "document") -> np.ndarray:
    """Bag of words hashed to 256 dimensions. No model needed, and texts sharing rare words land close."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOPWORDS:
                out[row, zlib.crc32(word.encode()) % 256] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


class StubModel:
    def chat(self, messages, **kw):
        return {"content": "According to the context [1]."}


@dataclass
class Config:
    k: int = 5
    min_score: float = 0.25
    chunk_size: int = 800
    overlap: int = 120
    ingest_twice: bool = False
    embed_wrappers: list[Callable] = field(default_factory=list)
    llm_wrappers: list[Callable] = field(default_factory=list)
    score_map: Callable[[float], float] | None = None
    latency_budget_ms: float = 60.0  # for the toy setup; a real model needs seconds, see cli.build_rig


class EmbedSpy:
    """Sits at the bottom of the embedding path and records what actually reaches the embedder."""

    def __init__(self, embed):
        self.embed, self.calls = embed, []

    def __call__(self, texts, kind="document"):
        self.calls.append(kind)
        return self.embed(texts, kind)


class ModelSpy:
    """Sits at the bottom of the model path and records the prompts the model really received."""

    def __init__(self, model):
        self.model, self.prompts = model, []

    def chat(self, messages, **kw):
        self.prompts.append(messages)
        return self.model.chat(messages, **kw)


class ScoredStore:
    def __init__(self, store: Store, score_map: Callable[[float], float] | None):
        self.store, self.score_map = store, score_map

    def query(self, vector, k=5, owner=None) -> list[Hit]:
        hits = self.store.query(vector, k=k, owner=owner)
        if self.score_map is None:
            return hits
        return [Hit(h.text, h.source, h.page, self.score_map(h.score)) for h in hits]


class Rig:
    def __init__(self, cfg: Config, workdir: Path, embed=bow_embed, model=None):
        self.cfg = cfg
        self.embed_spy = EmbedSpy(embed)
        self.embed = self.embed_spy
        for wrap in cfg.embed_wrappers:
            self.embed = wrap(self.embed)
        self.model_spy = ModelSpy(model or StubModel())
        self.llm = self.model_spy
        for wrap in cfg.llm_wrappers:
            self.llm = wrap(self.llm)

        raw = Store(Path(workdir) / "index")
        self.store = ScoredStore(raw, cfg.score_map)
        self._ingest(raw)
        self.embed_spy.calls.clear()  # only questions should show up in the spy

    def _ingest(self, raw: Store) -> None:
        for name, text in DOCS.items():
            pieces = chunk_text(text, self.cfg.chunk_size, self.cfg.overlap)
            chunks = [Chunk(f"{name}:1:{i}", p, name, 1) for i, p in enumerate(pieces)]
            raw.add(chunks, self.embed([c.text for c in chunks], "document"))
            if self.cfg.ingest_twice:  # what a non-idempotent ingest job leaves behind
                dupes = [Chunk(f"{c.id}:again", c.text, c.source, c.page) for c in chunks]
                raw.add(dupes, self.embed([c.text for c in dupes], "document"))

    def retrieve(self, question: str) -> list[Hit]:
        return self.store.query(self.embed([question], "query")[0], k=self.cfg.k)

    def ask(self, question: str, trace=None) -> Answer:
        return answer(question, self.store, self.llm, self.embed, k=self.cfg.k, min_score=self.cfg.min_score, trace=trace)
