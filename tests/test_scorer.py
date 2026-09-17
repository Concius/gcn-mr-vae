"""The scorer seam: evaluate() must honour whatever scoring rule it is given,
and the default must equal the notebook's dot product exactly. This is the
contract Module 3 (VAE) plugs into."""
import numpy as np
import pytest
import torch

from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.metrics.evaluate import evaluate, dot_product_scorer
from src.models.lightgcn import LightGCN
from src.utils import set_seed


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make_synthetic(root / "synthetic", n_users=120, n_items=80, seed=1)
    set_seed(0)
    ds = InteractionDataset("synthetic", root=str(root), val_frac=0.1, verbose=False)
    dev = torch.device("cpu")
    m = LightGCN(ds.n_users, ds.n_items, dim=8, n_layers=2, graph=ds.sparse_graph(dev))
    m.eval()
    au, ai = m.computer()
    return ds, dev, au, ai


def test_dot_product_scorer_matches_inline_expression(setup):
    ds, dev, au, ai = setup
    u = torch.arange(0, 17)
    assert torch.equal(dot_product_scorer(au, ai)(u), au[u] @ ai.t())


def _planted_scorer(ds, val_score=10.0, train_score=20.0):
    """Scores each user's TRAIN positives highest, VAL positives next, rest 0.
    If exclusion works, train items cannot occupy top-k on the val split, so
    val positives must be retrieved; if exclusion is broken, train items win."""
    def score(users):
        u = users.cpu().numpy()
        out = torch.zeros(len(u), ds.n_items)
        tr = ds.UserItemNet[u].tocoo(); va = ds.ValNet[u].tocoo()
        out[torch.from_numpy(tr.row), torch.from_numpy(tr.col)] = train_score
        out[torch.from_numpy(va.row), torch.from_numpy(va.col)] = val_score
        return out
    return score


def test_evaluate_honours_custom_scorer_and_excludes_train(setup):
    """A custom scorer is used, and train positives are masked on the val split.

    Scores train items above val items. With masking, rank 1 is a val item for
    every user, so recall@1 = 1/|val(u)|; without masking a train item takes
    rank 1 and recall@1 collapses to 0. Asserted at k=1 for the same reason as
    the test below: at larger k the val items still fit in the list either way
    and the check would not discriminate.
    """
    ds, dev, au, ai = setup

    def score(users):
        u = users.cpu().numpy()
        out = torch.zeros(len(u), ds.n_items)
        va = ds.ValNet[u].tocoo()
        out[torch.from_numpy(va.row), torch.from_numpy(va.col)] = 0.5
        tr = ds.UserItemNet[u].tocoo()
        out[torch.from_numpy(tr.row), torch.from_numpy(tr.col)] = 1.0
        return out

    res = evaluate(score, ds, "val", [1], batch_size=32, device=dev)
    users = sorted(ds.valDict)
    expected = sum(1.0 / len(ds.valDict[u]) for u in users) / len(users)
    assert res["recall@1"] == pytest.approx(expected, abs=1e-9)

    # and the scorer really was consulted (model-based scoring differs)
    ref = evaluate(dot_product_scorer(au, ai), ds, "val", [1], batch_size=32, device=dev)
    assert ref["recall@1"] != pytest.approx(expected, abs=1e-9)


def test_test_split_excludes_val_positives(setup):
    """Val positives are known at test time and must be masked.

    Scores val items above test items above everything else, and reads
    recall@1. With masking, rank 1 is a test item for every user, so
    recall@1 = 1/|test(u)|. Without masking, rank 1 is a *val* item for every
    user that has one, so their recall@1 is 0. k must be small enough that the
    val items actually displace test items -- at k=10 on this fixture every
    test item still fits in the list either way and the check is vacuous,
    which is why this asserts at k=1.
    """
    ds, dev, au, ai = setup

    def score(users):
        u = users.cpu().numpy()
        out = torch.zeros(len(u), ds.n_items)
        te = ds.TestNet[u].tocoo()
        out[torch.from_numpy(te.row), torch.from_numpy(te.col)] = 0.5
        va = ds.ValNet[u].tocoo()
        out[torch.from_numpy(va.row), torch.from_numpy(va.col)] = 1.0
        return out

    users = sorted(ds.testDict)
    with_val = [u for u in users if ds.valDict.get(u)]
    assert len(with_val) > 0.5 * len(users), "fixture must exercise the masking"

    res = evaluate(score, ds, "test", [1], batch_size=32, device=dev)
    expected = sum(1.0 / len(ds.testDict[u]) for u in users) / len(users)
    assert res["recall@1"] == pytest.approx(expected, abs=1e-9)


def test_scorer_shape_is_validated(setup):
    ds, dev, au, ai = setup
    bad = lambda users: torch.zeros(len(users), ds.n_items - 1)
    with pytest.raises(ValueError, match="scorer returned"):
        evaluate(bad, ds, "val", [5], batch_size=32, device=dev)
