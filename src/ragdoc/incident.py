"""Reproduces one real incident: the embedding model changes but the index isn't rebuilt.

    python -m ragdoc incident
"""
import re
import tempfile
import zlib
from pathlib import Path

import numpy as np
from ragchat.rag import answer
from ragchat.store import EmbeddingMismatch, Store

from .corpus import ANSWERABLE
from .rig import STOPWORDS, Config, Rig, StubModel, bow_embed


def bow_embed_v2(texts: list[str], kind: str = "document") -> np.ndarray:
    """Same size as the first model, different meaning: the words hash to different places."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOPWORDS:
                out[row, zlib.crc32(b"v2:" + word.encode()) % 256] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def small_embed(texts: list[str], kind: str = "document") -> np.ndarray:
    """A model with a different vector size, the loud kind of mismatch."""
    return bow_embed_v2(texts, kind)[:, :128]


def measure(index_dir: Path, embed, min_score: float = 0.25) -> dict:
    """What a user would see: does the right document come first, and does the system answer at all?"""
    store = Store(index_dir)
    top1 = answered = 0
    for q, source, _ in ANSWERABLE:
        hits = store.query(embed([q], "query")[0], k=5)
        top1 += bool(hits) and hits[0].source == source
        answered += answer(q, store, StubModel(), embed, min_score=min_score).grounded
    return {"questions": len(ANSWERABLE), "right_document_first": top1, "answered": answered}


def reproduce(workdir: Path | None = None) -> dict:
    workdir = Path(workdir or tempfile.mkdtemp())
    Rig(Config(), workdir / "before")  # builds the index with the original model
    index = workdir / "before" / "index"

    out = {"healthy": measure(index, bow_embed), "after_model_swap": measure(index, bow_embed_v2)}

    # the loud variant, straight from the vector database with no guard in front of it
    try:
        Store(index).col.query(query_embeddings=[small_embed(["cats"])[0].tolist()], n_results=1)
        out["raw_dimension_error"] = None
    except Exception as e:
        out["raw_dimension_error"] = str(e)[:160]

    # with the guard: the index remembers which model built it and refuses another
    guarded = workdir / "guarded"
    Store(guarded, embed_model="model-a")
    try:
        Store(guarded, embed_model="model-b")
        out["guard"] = None
    except EmbeddingMismatch as e:
        out["guard"] = str(e)
    return out


def main() -> None:
    r = reproduce()
    print("1. healthy:            right document first {right_document_first}/{questions}, answered {answered}/{questions}".format(**r["healthy"]))
    print("2. model swapped, no re-index (same vector size, so nothing errors):")
    print("                       right document first {right_document_first}/{questions}, answered {answered}/{questions}".format(**r["after_model_swap"]))
    print(f"3. swapped to a different vector size, unguarded: {r['raw_dimension_error']}")
    print(f"4. with the guard:     {r['guard']}")


if __name__ == "__main__":
    main()
