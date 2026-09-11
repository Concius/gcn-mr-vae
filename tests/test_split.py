import numpy as np
from src.data.splits import holdout_validation


def test_holdout_is_deterministic_and_disjoint():
    rng = np.random.default_rng(0)
    u = np.repeat(np.arange(50), 10)
    i = np.concatenate([rng.choice(100, 10, replace=False) for _ in range(50)])
    a = holdout_validation(u, i, 0.1, split_seed=7)
    b = holdout_validation(u, i, 0.1, split_seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    tr_u, tr_i, va_u, va_i = a
    assert len(tr_u) + len(va_u) == len(u)
    assert len(va_u) == 50  # 10% of 10 = 1 per user
    tr = set(zip(tr_u.tolist(), tr_i.tolist())); va = set(zip(va_u.tolist(), va_i.tolist()))
    assert not (tr & va)


def test_users_below_min_train_untouched():
    u = np.array([0, 1, 1, 2, 2, 2]); i = np.array([5, 6, 7, 8, 9, 10])
    tr_u, tr_i, va_u, va_i = holdout_validation(u, i, 0.5, split_seed=0, min_train=2)
    assert 0 not in va_u          # user 0 has one item
    assert (tr_u == 0).sum() == 1


def test_val_frac_zero_means_no_split():
    u = np.array([0, 0, 1]); i = np.array([1, 2, 3])
    tr_u, tr_i, va_u, va_i = holdout_validation(u, i, 0.0)
    assert len(va_u) == 0 and len(tr_u) == 3
