"""analysis.split_check: the dispersion index separates a random per-user
split from an item-biased one, and its simulated reference sits near 1."""
import numpy as np

from analysis.split_check import check


def _users_items(n_users=400, n_items=300, per_user=30, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.zipf(1.6, n_items).astype(float)
    pop /= pop.sum()
    return [rng.choice(n_items, size=per_user, replace=False, p=pop) for _ in range(n_users)]


def _flatten(d):
    u = np.concatenate([[k] * len(v) for k, v in d.items()]).astype(np.int64)
    i = np.concatenate(list(d.values())).astype(np.int64)
    return u, i


def _split(rows, rng, biased_items=None):
    tr, te = {}, {}
    for u, items in enumerate(rows):
        k = len(items) // 5
        if biased_items is None:
            idx = rng.choice(len(items), size=k, replace=False)
        else:  # test prefers a fixed half of the catalogue ("future" items)
            w = np.where(np.isin(items, biased_items), 8.0, 1.0)
            idx = rng.choice(len(items), size=k, replace=False, p=w / w.sum())
        mask = np.zeros(len(items), bool)
        mask[idx] = True
        te[u], tr[u] = items[mask], items[~mask]
    return tr, te


def test_random_split_is_near_one_and_matches_its_simulation():
    rows = _users_items()
    tr, te = _split(rows, np.random.default_rng(1))
    r = check(*_flatten(tr), *_flatten(te), n_sims=2)
    assert 0.8 < r["dispersion"] < 1.25
    assert all(0.8 < s < 1.25 for s in r["dispersion_random_split"])


def test_item_biased_split_is_flagged():
    rows = _users_items()
    tr, te = _split(rows, np.random.default_rng(1), biased_items=np.arange(0, 300, 2))
    r = check(*_flatten(tr), *_flatten(te), n_sims=2)
    assert r["dispersion"] > 1.5 * max(r["dispersion_random_split"])
