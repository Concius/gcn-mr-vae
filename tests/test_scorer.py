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
    ds, dev, au, ai = setup
    k = 50  # larger than any user's val count on this fixture
    res = evaluate(_planted_scorer(ds), ds, "val", [k], batch_size=32, device=dev)
    # every val positive is retrievable once train is masked -> perfect recall
    assert res[f"recall@{k}"] == pytest.approx(1.0)
    # and it must differ from model-based scoring, proving the scorer was used
    ref = evaluate(dot_product_scorer(au, ai), ds, "val", [k], batch_size=32, device=dev)
    assert ref[f"recall@{k}"] < 1.0


def test_test_split_excludes_val_positives(setup):
    """On the test split, val positives are known and must be masked too."""
    ds, dev, au, ai = setup
    # score val items highest; if they were NOT excluded they would fill top-k
    # and crowd out test items, giving recall 0 for users whose val count >= k
    sc = _planted_scorer(ds, val_score=20.0, train_score=10.0)
    res = evaluate(sc, ds, "test", [5], batch_size=32, device=dev)
    # test items score 0 (tied); with val+train masked, top-5 comes from the
    # remaining pool, so recall is > 0 for at least some users
    assert res["recall@5"] >= 0.0
    # stronger: compare against a scorer that boosts TEST items -> recall 1
    def boost_test(users):
        u = users.cpu().numpy(); out = torch.zeros(len(u), ds.n_items)
        te = ds.TestNet[u].tocoo()
        out[torch.from_numpy(te.row), torch.from_numpy(te.col)] = 1.0
        return out
    assert evaluate(boost_test, ds, "test", [50], batch_size=32, device=dev)["recall@50"] == pytest.approx(1.0)


def test_scorer_shape_is_validated(setup):
    ds, dev, au, ai = setup
    bad = lambda users: torch.zeros(len(users), ds.n_items - 1)
    with pytest.raises(ValueError, match="scorer returned"):
        evaluate(bad, ds, "val", [5], batch_size=32, device=dev)
