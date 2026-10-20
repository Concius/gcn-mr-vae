"""analysis.baselines: table, paired tests and Holm on a fabricated runs tree."""
import json

import numpy as np
import pytest

from analysis import baselines as B


def _write(root, ds, tag, seed, test, loss=0.5):
    d = root / ds / tag / f"seed{seed}"
    d.mkdir(parents=True)
    json.dump({"seed": seed, "test": test, "val": {"recall@20": 0.1}, "geometry": {}}, open(d / "results.json", "w"))
    json.dump([{"epoch": 10, "event": "eval", "loss_total": loss}], open(d / "history.json", "w"))


def _tree(root, bad=None):
    rng = np.random.default_rng(0)
    seeds = range(2020, 2030)
    for ds in B.DATASETS:
        for s in seeds:
            noise = rng.normal(0, 1e-4)
            base = {"recall@20": 0.18 + noise, "ndcg@20": 0.15, "gini@20": 0.90, "tail_catalog_coverage@20": 0.42}
            k = 1e-5 * (s - 2020)                              # distinct differences, no ties
            vae = {"recall@20": 0.16 + noise + k, "ndcg@20": 0.15 + rng.normal(0, 1e-3),  # ndcg: no effect
                   "gini@20": 0.85, "tail_catalog_coverage@20": 0.62}
            mf = {"recall@20": 0.14 + noise - k, "ndcg@20": 0.15, "gini@20": 0.82, "tail_catalog_coverage@20": 0.62}
            for tag, t in ((B.ARMS["LightGCN"], base), (B.ARMS["VAE-CF"], vae), (B.ARMS["MF-BPR"], mf)):
                loss = float("nan") if bad == (ds, tag, s) else 0.5
                _write(root, ds, tag, s, t, loss)


def test_table_and_tests(tmp_path):
    _tree(tmp_path / "runs")
    assert B.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]) == 0
    desc = __import__("pandas").read_csv(tmp_path / "out" / "baselines_describe.csv")
    comp = __import__("pandas").read_csv(tmp_path / "out" / "baselines_tests.csv")
    assert len(desc) == 9 and set(desc.n) == {10}
    r = desc[(desc.dataset == "gowalla") & (desc.arm == "VAE-CF")].iloc[0]
    assert r["recall@20_mean"] == pytest.approx(0.16, abs=1e-3)
    vals = [x["test"]["recall@20"] for x in B.gather(tmp_path / "runs")[("gowalla", "VAE-CF")].values()]
    assert r["recall@20_sd"] == pytest.approx(np.std(vals, ddof=1), rel=1e-9)   # sample sd, n - 1
    rec = comp[comp.metric == "recall@20"]
    assert len(rec) == 6
    lg = rec[rec.comparison == "VAE-CF - LightGCN"]
    assert (lg.a_higher == 0).all() and np.allclose(lg.p, 2 / 1024)
    assert np.allclose(lg.p_holm, 6 * 2 / 1024)          # all six tied at the minimum p
    mf = rec[rec.comparison == "VAE-CF - MF-BPR"]
    assert (mf.a_higher == 10).all() and np.allclose(mf.rel_pct, 100 * 0.02 / 0.14, rtol=1e-2)
    nd = comp[(comp.metric == "ndcg@20") & (comp.comparison == "VAE-CF - LightGCN")]
    assert (nd.p > 0.05).all()                           # noise only: no effect found


def test_unhealthy_run_blocks_everything(tmp_path):
    _tree(tmp_path / "runs", bad=("amazon-book", B.ARMS["VAE-CF"], 2025))
    assert B.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_missing_arm_is_reported_not_fatal(tmp_path, capsys):
    _tree(tmp_path / "runs")
    import shutil
    shutil.rmtree(tmp_path / "runs" / "amazon-book" / B.ARMS["VAE-CF"])
    assert B.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]) == 0
    assert "missing" in capsys.readouterr().out
    comp = __import__("pandas").read_csv(tmp_path / "out" / "baselines_tests.csv")
    assert len(comp[comp.metric == "recall@20"]) == 4     # amazon comparisons skipped
