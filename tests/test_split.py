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


def test_rng_state_roundtrip_survives_map_location():
    """Regression: torch.load(map_location=<device>) moves the saved RNG
    ByteTensors onto that device, but torch.set_rng_state only accepts a CPU
    ByteTensor. rng_state_load must coerce them back."""
    import tempfile, os, random as _random
    import torch
    from src.utils import rng_state_dict, rng_state_load

    torch.manual_seed(7); _random.seed(7); np.random.seed(7)
    st = rng_state_dict()
    expect = (torch.randint(0, 10_000, (5,)).tolist(), _random.random(), float(np.random.rand()))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save({"rng": st}, p)
        ck = torch.load(p, map_location=torch.device("cpu"), weights_only=False)
        rng_state_load(ck["rng"])

    got = (torch.randint(0, 10_000, (5,)).tolist(), _random.random(), float(np.random.rand()))
    assert got == expect


def test_rng_state_load_accepts_non_cpu_shaped_input():
    """Simulate what map_location=cuda produces: the same states as plain
    tensors that must be coerced without error."""
    import torch
    from src.utils import rng_state_dict, rng_state_load
    st = rng_state_dict()
    st = {"python": list(st["python"]), "numpy": st["numpy"],
          "torch": st["torch"].clone().to(torch.uint8)}
    rng_state_load(st)   # must not raise


def test_summary_groups_by_directory_not_stored_tag(tmp_path):
    """Two runs with identical hyperparameters placed in different directories
    (e.g. fresh vs restored Adam) must stay separate. The stored arm_tag is
    computed from hyperparameters and ignores a paths.run_dir override, so
    grouping on it averaged them together and reported n=6 from 3 seeds."""
    import json
    from analysis.summarize import collect_results, summary_table
    for folder, rec in (("fresh", 0.19), ("restored", 0.18)):
        for s in (2020, 2021, 2022):
            p = tmp_path / "gowalla" / folder / f"seed{s}"; p.mkdir(parents=True)
            json.dump({"dataset": "gowalla", "arm": "emb_mr", "arm_tag": "IDENTICAL",
                       "seed": s, "best_epoch": 1, "selected_on": "", "epochs_budget": 1,
                       "warmup_epochs": 0, "lambda_manifold": 0.0,
                       "test": {"recall@20": rec}, "val": {}, "geometry": {"er_table": 1.0}},
                      open(p / "results.json", "w"))
    t = summary_table(collect_results(tmp_path), metrics=["test_recall@20"])
    assert list(t[("n", "seeds")]) == [3, 3]
    assert sorted(t[("mean", "test_recall@20")].round(2)) == [0.18, 0.19]
