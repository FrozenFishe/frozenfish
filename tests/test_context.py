"""Tests for the context layer: CAG / CAR / RAG and the hybrid router."""

from __future__ import annotations

import pytest

from frozenfish.config import Config
from frozenfish.context import (
    CagCache,
    CarIndex,
    CarPartition,
    ContextRouter,
    RagIndex,
    context_from_config,
    cosine,
    embed,
    mmr_select,
    rrf_merge,
    tokenize,
)

DOC_A = "the frozen fish swims in the arctic ocean ice"
DOC_B = "gpu routing sends the model to the least busy accelerator"
DOC_C = "the fish hides beneath the ice in the arctic"


def _car():
    car = CarIndex()
    car.add(DOC_A, tool="filesystem")
    car.add(DOC_B, tool="vector_db")
    return car


class TestEmbedding:
    def test_tokenize_lowercases(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_embed_deterministic(self):
        assert embed("abc") == embed("abc")

    def test_cosine_self_one(self):
        v = embed(DOC_A)
        assert cosine(v, v) == pytest.approx(1.0)

    def test_cosine_related_beats_unrelated(self):
        q = embed("arctic ice fish")
        assert cosine(q, embed(DOC_A)) > cosine(q, embed(DOC_B))


class TestMmr:
    def test_mmr_prefers_relevance_then_diversity(self):
        docs = [
            (0.9, {"vec": embed(DOC_A), "id": "a"}),
            (0.85, {"vec": embed(DOC_C), "id": "c"}),  # near-duplicate of a
            (0.75, {"vec": embed(DOC_B), "id": "b"}),
        ]
        picked = mmr_select(embed(DOC_A), docs, k=2, lam=0.7)
        ids = [d["id"] for d in picked]
        assert ids[0] == "a"
        # b (diverse) beats c (duplicate of a) at the second slot
        assert ids[1] == "b"


class TestCag:
    def test_build_and_get(self):
        cag = CagCache(path="/tmp/ff-test/cag.sqlite3")
        key = cag.build("system preface", fingerprint="fp-1")
        ent = cag.get(key)
        assert ent is not None
        assert ent["token_count"] == len(tokenize("system preface"))

    def test_key_is_stable(self):
        k1 = CagCache.key("preface", "fp")
        k2 = CagCache.key("preface", "fp")
        assert k1 == k2
        assert k1 != CagCache.key("preface", "other")

    def test_oversized_preface_rejected(self):
        cag = CagCache(path="/tmp/ff-test/cag.sqlite3", max_tokens=1024)
        big = "word " * 5000
        with pytest.raises(ValueError, match="exceeds max_tokens"):
            cag.build(big)

    def test_invalidate(self):
        cag = CagCache(path="/tmp/ff-test/cag.sqlite3")
        key = cag.build("p1")
        cag.build("p2")
        assert cag.invalidate() == 2
        assert cag.get(key) is None

    def test_disabled_never_hits(self):
        cag = CagCache(path="/tmp/ff-test/cag.sqlite3", enabled=False)
        key = cag.build("x")
        assert cag.get(key) is None

    def test_stats(self):
        cag = CagCache(path="/tmp/ff-test/cag.sqlite3")
        key = cag.build("x")
        cag.get(key)
        cag.get("missing")
        s = cag.stats()
        assert s["hits"] == 1 and s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(0.5)
        assert s["entries"] == 1


class TestCar:
    def test_partition_isolation(self):
        car = _car()
        assert car.partitions() == ["filesystem", "vector_db"]
        hits = car.search("arctic ice fish", tools=["filesystem"])
        assert len(hits) == 1
        assert hits[0]["partition"] == "filesystem"
        assert hits[0]["score"] > 0.35

    def test_no_tool_scans_everything(self):
        car = _car()
        hits = car.search("gpu routing accelerator")
        assert hits[0]["partition"] == "vector_db"

    def test_threshold_filters(self):
        car = _car()
        hits = car.search("completely unrelated quantum banana")
        assert all(h["score"] < 0.35 for h in hits)

    def test_disabled_returns_empty(self):
        car = CarIndex(enabled=False)
        car.add(DOC_A, tool="t")
        assert car.search(DOC_A) == []

    def test_len(self):
        part = CarPartition("t")
        part.add(DOC_A)
        assert len(part) == 1


class TestRag:
    def test_reindex_and_search(self):
        rag = RagIndex()
        n = rag.reindex([DOC_A, DOC_B, DOC_C])
        assert n == 3
        hits = rag.search("arctic ice fish")
        assert len(hits) == 2  # a and c clear the 0.35 bar
        assert hits[0]["text"] in (DOC_A, DOC_C)
        assert rag.last_confidence > 0.35

    def test_threshold_gates(self):
        rag = RagIndex()
        rag.reindex([DOC_A, DOC_B])
        hits = rag.search("quantum banana string")
        assert hits == []
        assert rag.last_confidence == 0.0

    def test_mmr_off_still_ranks(self):
        rag = RagIndex(mmr=False)
        rag.reindex([DOC_A, DOC_B, DOC_C])
        hits = rag.search("arctic ice fish")
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_disabled(self):
        rag = RagIndex(enabled=False)
        rag.reindex([DOC_A])
        assert rag.search(DOC_A) == []

    def test_rrf_merge(self):
        d1, d2, d3 = {"id": "a"}, {"id": "b"}, {"id": "c"}
        fused = rrf_merge([[d1, d2, d3], [d2, d1, d3]])
        assert [f["id"] for f in fused][:2] in (["a", "b"], ["b", "a"])
        # a and b outrank c in at least one list
        assert fused[0]["score"] > fused[-1]["score"]


class TestContextRouter:
    def _router(self, tmp_path):
        cag = CagCache(path=str(tmp_path / "cag.sqlite3"))
        car = _car()
        rag = RagIndex()
        rag.reindex([DOC_A, DOC_B, DOC_C])
        return ContextRouter(cag, car, rag)

    def test_cag_wins(self, tmp_path):
        r = self._router(tmp_path)
        r.cag.build("my preface")
        out = r.resolve("whatever", preface="my preface")
        assert out["source"] == "cag"
        assert out["confidence"] == 1.0
        assert "cag_key" in out
        assert r.counts["cag"] == 1

    def test_car_when_tools_declared(self, tmp_path):
        r = self._router(tmp_path)
        out = r.resolve("arctic ice fish", tools=["filesystem"])
        assert out["source"] == "car"
        assert out["confidence"] > 0.35
        assert r.counts["car"] == 1

    def test_rag_fallback(self, tmp_path):
        r = self._router(tmp_path)
        out = r.resolve("arctic ice fish")  # no tools, no preface
        assert out["source"] == "rag"
        assert out["confidence"] > 0.35
        assert r.counts["rag"] == 1

    def test_none_when_nothing_matches(self, tmp_path):
        r = self._router(tmp_path)
        out = r.resolve("zzz qqq completely novel term")
        assert out["source"] == "none"
        assert out["confidence"] == 0.0
        assert r.counts["empty"] == 1

    def test_stats(self, tmp_path):
        r = self._router(tmp_path)
        r.resolve("arctic ice fish")
        s = r.stats()
        assert s["counts"]["rag"] == 1
        assert s["rag"]["docs"] == 3

    def test_context_from_config(self):
        cfg = Config.load(None)
        r = context_from_config(cfg)
        assert r.cag.max_tokens == 32768
        assert r.rag.threshold == 0.35
        assert r.car.partition_by == "tool"
        assert r.similarity_threshold == 0.35
