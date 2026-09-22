# Incident: answers collapsed after an embedding model change

Reproduced with `uv run python -m ragdoc incident`; the numbers below come from that run.

## Summary

The embedding model was changed but the existing index was not rebuilt. Retrieval quality fell to roughly chance level. Nothing errored, no alert fired, and the service kept returning HTTP 200 the whole time. Users saw "I don't know" to almost every question.

## Impact

| | Before | After the swap |
|---|---|---|
| Right document ranked first | 8 of 8 questions | 1 of 8 |
| Questions answered | 8 of 8 | 1 of 8 |
| Errors in the logs | none | none |

With eight documents, picking at random gets about one right, so 1 of 8 means retrieval had stopped working, not that it had gotten a bit worse.

## Timeline (as reproduced)

1. An index is built with model A. All probes pass.
2. Model B is configured for queries. It produces vectors of the same size (256 dimensions) but in a different space, so the vector database accepts them without complaint.
3. Every query now compares B-vectors against A-vectors. Similarity scores are near zero for everything, so almost every question falls under the similarity cutoff and gets "I don't know".
4. The service reports healthy throughout.

## Root cause

An index is only meaningful for the model that built it, and nothing recorded which model that was. Same-size vectors from different models look identical to the database. The only signal was answer quality, which nothing was measuring.

The loud version of the same mistake, a model with a different vector size, did fail, with an exception from inside the vector database (`Collection expecting embedding with dimension of 256, got 128`). The app did not catch it, so a user would have got a generic server error, with nothing pointing at the model change.

## Why it went unnoticed

- Health checks only test that the process is up, not that retrieval works.
- No retrieval-quality signal in production: no canary questions, no alert on decline rate.
- The failure looks like a legitimate answer ("I don't know") rather than an error.

## What changed

| Action | Where | Status |
|---|---|---|
| The index records the embedding model that built it and refuses to open with a different one, telling you to re-index | `Store` in the RAG chatbot | done, tested |
| Vector size is checked before a query or insert, with a clear message instead of a database error | `Store` in the RAG chatbot | done, tested |
| The web app refuses to start on a mismatched index, and returns 409 with the same message if it happens at runtime | `webapp.py` in the RAG chatbot | done, tested |
| Model outages return a clean 502 instead of an unhandled 500 | `webapp.py` in the RAG chatbot | done, tested |
| Per-stage trace events (`retrieve`, `decline`, `generate`) and a log summary that flags a high decline rate, near-miss scores and an empty index | `rag.py` and `ragdoc.logs` | done, tested |
| A probe suite (`ragdoc doctor`) whose `answers_when_it_should` check uses the same measurement as the table above (1 of 8 here) | `ragdoc.probes` | done, tested |

## Still open

- Run the probe suite on a schedule against production, not only on demand. This belongs with monitoring, with an alert on the decline rate.
- Add a few canary questions with known answers to the deploy checklist, so a model change is followed by a retrieval check.
- Re-indexing has to be cheap for the guard to be pleasant to live with. Embeddings are cached by model and text, so switching model means recomputing everything once (about 46 chunks per second on the laptop this was run on); that's a reason to schedule model changes, not a reason to skip the re-index.

## Limits of this reproduction

The two "models" are toy bag-of-words embedders, not real embedding models: only one real embedding model was available on the machine this was run on. The mechanism does not depend on that (any two models that produce the same vector size behave this way), but the exact 1-of-8 figure does. A real pair of models would land somewhere near chance too, not at exactly the same number.
