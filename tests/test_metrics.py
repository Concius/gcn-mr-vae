"""Metric correctness: NDCG IDCG fix, recall, gini, tail metrics."""
import numpy as np
import pytest

from src.metrics.ranking import ndcg_at_k, ndcg_single, recall_at_k, precision_at_k
from src.metrics.diversity import gini_from_counts, DiversityAccumulator


def notebook_ndcg(relevance, k):
    """The buggy notebook implementation (Cell 6), for contrast."""
    r = np.asarray(relevance, dtype=float)[:k]
    dcg = np.sum(r / np.log2(np.arange(2, k + 2)))
    n_hits = int(np.sum(r))
    if n_hits == 0:
        return 0.0
    idcg = np.sum(1.0 / np.log2(np.arange(1, n_hits + 1) + 1))
    return dcg / idcg


def test_ndcg_perfect_list_is_one():
    # 3 relevant, all at the top, k=5
    assert ndcg_single([1, 1, 1, 0, 0], k=5, n_rel=3) == pytest.approx(1.0)


def test_ndcg_penalises_missed_relevant_items():
    # user has 5 relevant items, only 2 retrieved (at the top). Notebook gave 1.0.
    rel = [1, 1, 0, 0, 0]
    buggy = notebook_ndcg(rel, 5)
    fixed = ndcg_single(rel, k=5, n_rel=5)
    assert buggy == pytest.approx(1.0)
    disc = 1 / np.log2(np.arange(2, 7))
    assert fixed == pytest.approx((disc[0] + disc[1]) / disc.sum())
    assert fixed < buggy


def test_ndcg_idcg_capped_at_k():
    # 100 relevant items, k=3, all 3 hits at the top -> ideal list has 3 items -> 1.0
    assert ndcg_single([1, 1, 1], k=3, n_rel=100) == pytest.approx(1.0)


def test_ndcg_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    hits = rng.random((50, 20)) < 0.2
    n_rel = rng.integers(1, 40, 50)
    v = ndcg_at_k(hits, n_rel, 10)
    for i in range(50):
        assert v[i] == pytest.approx(ndcg_single(hits[i], 10, int(n_rel[i])))


def test_recall_precision():
    hits = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=bool)
    n_rel = np.array([4, 3])
    assert recall_at_k(hits, n_rel, 4).tolist() == pytest.approx([0.5, 0.0])
    assert precision_at_k(hits, 4).tolist() == pytest.approx([0.5, 0.0])


def test_gini():
    assert gini_from_counts(np.ones(10)) == pytest.approx(0.0)
    one_hot = np.zeros(10); one_hot[0] = 100
    assert gini_from_counts(one_hot) == pytest.approx(0.9)


def test_tail_metrics_two_definitions_differ():
    n_items = 10
    lt = np.zeros(n_items, bool); lt[5:] = True          # items 5..9 are long tail
    acc = DiversityAccumulator(n_items, [2], lt)
    # two users, both recommend [5, 0] -> per-user share 0.5; catalog: only item 5 of 5 tail items
    acc.update(np.array([[5, 0], [5, 0]]))
    out = acc.finalize()
    assert out["tail_share_user@2"] == pytest.approx(0.5)
    assert out["tail_catalog_coverage@2"] == pytest.approx(0.2)
    assert out["catalog_coverage@2"] == pytest.approx(0.2)
