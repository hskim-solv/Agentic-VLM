# 0020: Protocol-based pluggability for retrieval-side extensions

- **Status**: proposed
- **Date**: 2026-05-12
- **Deciders**: hskim
- **Related**: [ADR 0001](./0001-preserve-naive-baseline.md) (baseline preserved), [ADR 0010](./0010-hybrid-bm25-dense-retrieval-rrf.md) (RRF fusion, retrieval-side), [ADR 0013](./0013-observability-as-additive-pluggable-surface.md) (pluggable surface — sibling pattern, observability domain), issues [#176](https://github.com/hskim-solv/BidMate-DocAgent/issues/176) / [#232](https://github.com/hskim-solv/BidMate-DocAgent/issues/232) / [#345](https://github.com/hskim-solv/BidMate-DocAgent/issues/345) / [#394](https://github.com/hskim-solv/BidMate-DocAgent/issues/394), PRs [#234](https://github.com/hskim-solv/BidMate-DocAgent/pull/234) / [#288](https://github.com/hskim-solv/BidMate-DocAgent/pull/288) / [#296](https://github.com/hskim-solv/BidMate-DocAgent/pull/296) / [#342](https://github.com/hskim-solv/BidMate-DocAgent/pull/342) / [#358](https://github.com/hskim-solv/BidMate-DocAgent/pull/358)

> **NOTE — skeleton only.** Status is `proposed` because the body below
> stops at *Context*: rationale (Decision, Consequences, Alternatives) is
> reserved for the author per the cognitive-ownership boundary noted in
> [#394](https://github.com/hskim-solv/BidMate-DocAgent/issues/394). Remove this note and
> flip to `accepted` once the rationale sections are filled in.

## Context

Two retrieval-side refactors landed in Phase-3 prep with the same
structural shape but without a formal convention:

1. **`VectorStore` Protocol** ([#176](https://github.com/hskim-solv/BidMate-DocAgent/issues/176) Stage 1).
   Leaf module [`rag_vector_store.py`](../../rag_vector_store.py)
   defines a `@runtime_checkable typing.Protocol` (`dimension`, `__len__`,
   `get`, `query`, `persist`). The default `InMemoryVectorStore` wraps
   the original numpy matrix; `QdrantVectorStore` (#288/#296) is the
   second adapter. Dispatch is env-var (`BIDMATE_INDEX_BACKEND={memory,qdrant}`).
   The retrieval orchestrator (`rag_core.retrieve_candidates` after
   [#342](https://github.com/hskim-solv/BidMate-DocAgent/pull/342)) consumes the Protocol via
   `store.query(...)` without knowing the concrete backend.
2. **`Reranker` Protocol** ([#345](https://github.com/hskim-solv/BidMate-DocAgent/issues/345)).
   Leaf module [`rag_reranker.py`](../../rag_reranker.py) defines a
   `@runtime_checkable Reranker` Protocol (`rerank(query, candidates,
   *, top_n) -> (reordered, meta)`). The default `CrossEncoderReranker`
   wraps the existing `rag_rerank.rerank` (preserving its env-var
   backend dispatch — stub / bge / cohere / bge_ko). The retrieval
   orchestrator (`rag_core.apply_fusion_and_reranking` after [#342](https://github.com/hskim-solv/BidMate-DocAgent/pull/342))
   consumes it via `default_reranker().rerank(...)`.

Both PRs justify the pattern in their own bodies, but no ADR codifies
**the pattern itself**. Phase 3 will recur to it (HyDE-style query
rewriting, multi-query retrieval, possibly an embedding-backend
Protocol per [ADR 0019](./0019-embedding-default-stays-minilm.md) condition
4) — without a written convention, each future PR re-litigates the
shape (ABC vs Protocol, factory vs direct import, env-var vs plan-dict
dispatch).

[ADR 0013](./0013-observability-as-additive-pluggable-surface.md) already
established "pluggable surface" semantics in the observability domain
(tracer backends, fail-closed defaults). This ADR codifies the
retrieval-side counterpart so the two pluggability surfaces share a
documented shape.

The observed convention (informal, four properties — to be made
canonical in the *Decision* section below):

- `@runtime_checkable typing.Protocol` (not `abc.ABC`).
- Leaf module (`rag_*.py` at repo root) — imports nothing from
  `rag_core.py`; `rag_core.py` consumes it.
- Default adapter wraps the existing implementation so behavior is
  preserved without flag flips.
- Single `default_*()` factory or env-var dispatch — the **one** hook
  future implementations swap.

## Decision

<!--
TODO (author): one-sentence decision followed by the specifics needed
to act on it. Name the knob/toggle (e.g. "any new retrieval-side
extension point uses the four-property shape above; the dispatch hook
is named `default_<surface>()` or routed via `BIDMATE_<SURFACE>_BACKEND`").
Specify the boundaries — does this apply ONLY to retrieval-side, or
also to ingestion / eval / answer surfaces? Cite the precedent ADRs
(0013 for observability; 0019 condition 4 for the embedding-backend
case if/when it arrives).
-->

## Consequences

<!--
TODO (author): wins and costs. Suggested prompts —
- Positive: forward-compatible extension; testable via `isinstance`
  checks on the runtime_checkable Protocol; behavior preserved by
  wrapping existing implementations (zero-day ranking drift in eval).
- Negative: ceremony cost per extension (Protocol + default adapter +
  factory = three sites instead of one function); a small Python-only
  decision (Protocol vs ABC is a language-version-sensitive call).
- Reversal cost: low (each Protocol is a leaf module, no cross-cutting
  changes; reverting one Protocol does not touch the others).
- Contracts locked: the four-property shape becomes the review checklist
  for future Phase 3 retrieval PRs.
-->

## Alternatives considered

<!--
TODO (author): one or two bullets each, suggested options to address —
- ABC (`abc.ABC` + `@abstractmethod`) vs `typing.Protocol`. The repo
  uses Protocol; ABC requires explicit subclassing, Protocol allows
  structural typing for third-party adapters (e.g., a future user-
  authored Reranker that doesn't import our base class).
- Factory function (`default_reranker()`) vs direct constructor import.
  Factory centralizes the dispatch hook for future plan-based routing
  (without touching `rag_core.py`); direct import would force a
  retrieval-orchestrator edit each time a new impl is added.
- Env-var dispatch vs plan-dict dispatch. VectorStore uses env-var;
  Reranker uses factory-with-future-plan-dispatch. Note whether this
  ADR locks one of the two, or allows both based on the lifecycle of
  the surface (env-var = build-time; plan = per-query).
- Single shared Protocol module vs per-surface modules. Current
  pattern: per-surface (`rag_vector_store.py`, `rag_reranker.py`).
  Alternative: a `rag_protocols.py` collection — note why per-surface
  was preferred (or revisit if the count grows past 3-4 Protocols).
-->
