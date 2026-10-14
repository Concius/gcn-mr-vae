"""analysis.vae_val_test: its vectorised reads equal a plain per-user loop, and
its trajectory is the one ``python -m src.train`` produces for the same overrides."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import src.train  # noqa: F401  registers the armtag resolver
from analysis.vae_val_test import main as vt_main
from analysis.vae_val_test import matched_subset, read_all
from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.sampling import dense_rows

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("vtdata")
    make_synthetic(root / "synthetic", n_users=300, n_items=200, seed=0)
    return root


def _loop_reads(sc_all, ds, S, users, k=20):
    """Reference: one user at a time, sets and Python lists."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    acc = {key: [] for key in ("val", "test", "test_liang", "test_ndcg", "val_clean", "test_matched")}
    for j, u in enumerate(users):
        u = int(u)
        tr = set(ds.UserItemNet[u].indices.tolist())
        v, t, s = set(ds.valDict[u]), set(ds.testDict[u]), set(S[u].indices.tolist())

        def hits(masked, rel):
            sc = sc_all[j].clone()
            sc[list(masked)] = -float("inf")
            return np.array([i in rel for i in torch.topk(sc, k).indices.tolist()], float)

        h = hits(tr, v); acc["val"].append(h.sum() / len(v))
        h = hits(tr | v, t)
        acc["test"].append(h.sum() / len(t)); acc["test_liang"].append(h.sum() / min(k, len(t)))
        acc["test_ndcg"].append((h * disc).sum() / disc[:min(k, len(t))].sum())
        h = hits(tr | t, v); acc["val_clean"].append(h.sum() / len(v))
        h = hits(tr | v | (t - s), s); acc["test_matched"].append(h.sum() / len(s))
    return {key: float(np.mean(x)) for key, x in acc.items()}


def test_vectorised_reads_match_a_per_user_loop(synth):
    ds = InteractionDataset("synthetic", root=synth, val_frac=0.1, split_seed=2020, verbose=False)
    S = matched_subset(ds, 7)
    users = np.array(sorted(set(ds.valDict) & set(ds.testDict)), dtype=np.int64)
    # Scores that put test items (and some validation items) near the top, so
    # every masking choice changes the top-k: random base + bonuses.
    g = torch.Generator().manual_seed(0)
    base = torch.rand((ds.n_users, ds.n_items), generator=g, dtype=torch.float64)
    T = dense_rows(ds.TestNet, np.arange(ds.n_users), "cpu").double()
    V = dense_rows(ds.ValNet, np.arange(ds.n_users), "cpu").double()
    sc_all = (base + 0.9 * T * torch.rand(T.shape, generator=g, dtype=torch.float64)
              + 0.8 * V * torch.rand(V.shape, generator=g, dtype=torch.float64))[torch.from_numpy(users)]
    pos = {int(u): j for j, u in enumerate(users)}
    score = lambda bu: sc_all[[pos[int(u)] for u in bu]]  # noqa: E731
    sizes_ok = all(len(S[u].indices) == min(len(ds.valDict[u]), len(ds.testDict[u])) for u in users)
    assert sizes_ok and all(set(S[u].indices) <= set(ds.testDict[u]) for u in users)
    got = read_all(score, ds, S, users, batch=37, device=torch.device("cpu"))
    ref = _loop_reads(sc_all, ds, S, users)
    for key in ref:
        assert got[key] == pytest.approx(ref[key], abs=1e-12), key
    # the masking choices matter on these scores (so a wrong mask would be caught)
    assert got["val_clean"] > got["val"] + 0.005
    assert abs(got["test_matched"] - got["test"]) > 0.005
    # k = 2: many users have more than k test items, so min(k, n_rel) differs from n_rel
    got2 = read_all(score, ds, S, users, batch=37, device=torch.device("cpu"), k=2)
    ref2 = _loop_reads(sc_all, ds, S, users, k=2)
    for key in ref2:
        assert got2[key] == pytest.approx(ref2[key], abs=1e-12), key
    assert got2["test_liang"] > got2["test"] + 0.005


def test_trajectory_matches_src_train(synth, tmp_path):
    over = ["dataset=synthetic", "arm=vae_cf", "seed=2021", "device=cpu", f"paths.data={synth}",
            "eval.batch_size=128", "eval.topks=[10,20]", "arm.vae.hidden=16", "arm.vae.latent=4",
            "arm.vae.batch_size=64", "geometry.np_max_samples=50", "geometry.np_ref_k=[5]", "geometry.np_k=[5]"]
    cmd = [sys.executable, "-m", "src.train", *over, "epochs=4", "eval.every=2", f"paths.runs={tmp_path}",
           f"hydra.run.dir={tmp_path}/hydra"]
    out = subprocess.run(cmd, cwd=REPO, env=dict(os.environ), capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    hist = [h for h in json.load(open(tmp_path / "synthetic" / "vae_cf" / "seed2021" / "history.json"))
            if h.get("event") == "eval"]
    vt_main(["--at", "1,2,4", "--out", str(tmp_path / "vt.json"), *over])
    reads = {r["epoch"]: r for r in json.load(open(tmp_path / "vt.json"))["reads"]}
    assert set(reads) == {1, 2, 4}
    assert reads[1]["val"] == pytest.approx(reads[1]["val_src_evaluate"], abs=1e-12)
    assert reads[1]["test"] == pytest.approx(reads[1]["test_src_evaluate"], abs=1e-12)
    for h in hist:   # all users of the synthetic set have val and test items
        assert reads[h["epoch"]]["val"] == pytest.approx(h["val_recall@20"], abs=1e-9)
        assert reads[h["epoch"]]["test"] == pytest.approx(h["test_recall@20"], abs=1e-9)
