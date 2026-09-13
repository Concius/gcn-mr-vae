"""Numerical equivalence audit: ported code vs the notebook's original code.

Reimplements the notebook's BPRLoss_MR.stageOne math and its Test() loop
verbatim (Cells 5, 6, 8) and compares them against the ported versions on
identical inputs. Any difference that is not an intended fix is drift.

Run: LD_LIBRARY_PATH=... python3 audit_equivalence.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.models.lightgcn import LightGCN
from src.models.losses import BPRTerm, BatchContext, CompositeLoss, L2EgoTerm, ManifoldTerm
from src.models.mr_layer import build_embedding_laplacian, manifold_loss
from src.metrics.evaluate import dot_product_scorer, evaluate
from src.metrics.diversity import gini_from_counts
from src.utils import set_seed

DEV = torch.device("cpu")
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  [{'OK ' if cond else 'DRIFT'}] {name} {detail}")


# ---------------------------------------------------------------- notebook code
def nb_stage_one_loss(model, users, pos, neg, decay, laplacian=None, lam=0.0):
    """Verbatim math from Cell 8 BPRLoss_MR.stageOne (minus the optimizer step)."""
    all_users, all_items = model.computer()
    users_emb, pos_emb, neg_emb = all_users[users], all_items[pos], all_items[neg]
    users_emb_ego = model.embedding_user(users)
    pos_emb_ego = model.embedding_item(pos)
    neg_emb_ego = model.embedding_item(neg)
    pos_scores = torch.sum(users_emb * pos_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_emb, dim=1)
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
    reg_loss = (1.0 / 2.0) * (users_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2)
                              + neg_emb_ego.norm(2).pow(2)) / float(len(users))
    total = bpr_loss + decay * reg_loss
    m_val = 0.0
    if laplacian is not None and lam > 0:
        all_emb = torch.cat([all_users, all_items])
        Lz = torch.sparse.mm(laplacian, all_emb)
        m = (all_emb * Lz).sum() / all_emb.shape[0]
        total = total + lam * m
        m_val = m.item()
    return total, bpr_loss.item(), reg_loss.item(), m_val


def nb_ndcg(relevance, k):
    r = np.asarray(relevance, dtype=float)[:k]
    dcg = np.sum(r / np.log2(np.arange(2, k + 2)))
    n = int(np.sum(r))
    if n == 0:
        return 0.0
    return dcg / np.sum(1.0 / np.log2(np.arange(1, n + 1) + 1))


def nb_test(dataset, model, topks):
    """Verbatim structure of Cell 6 Test(): per-user python loop."""
    testDict = dataset.testDict
    short, long = dataset.popularity_groups["short_head"], dataset.popularity_groups["long_tail"]
    res = {m: np.zeros(len(topks)) for m in
           ["precision", "recall", "ndcg", "precision_short", "recall_short", "ndcg_short",
            "precision_long", "recall_long", "ndcg_long", "gini", "tail_coverage"]}
    counts = [np.zeros(dataset.n_items, dtype=np.int64) for _ in topks]
    tail = np.zeros(len(topks))
    n_all = n_short = n_long = 0
    max_K = max(topks)
    with torch.no_grad():
        users = list(testDict.keys())
        all_users, all_items = model.computer()
        for s in range(0, len(users), 4096):
            bu = users[s:s + 4096]
            allPos = dataset.all_pos(bu)
            rating = all_users[torch.tensor(bu)] @ all_items.t()
            ei, ec = [], []
            for i, items in enumerate(allPos):
                ei.extend([i] * len(items)); ec.extend(items)
            rating[ei, ec] = -(1 << 30)
            _, rk = torch.topk(rating, k=max_K)
            rk = rk.cpu().numpy()
            for i, u in enumerate(bu):
                gt = testDict.get(u)
                if not gt:
                    continue
                n_all += 1
                recs = rk[i]
                gs, gl = set(gt) & short, set(gt) & long
                for ki, k in enumerate(topks):
                    tk = recs[:k]
                    h = len(np.intersect1d(tk, gt))
                    res["precision"][ki] += h / k
                    res["recall"][ki] += h / len(gt)
                    res["ndcg"][ki] += nb_ndcg([1 if x in set(gt) else 0 for x in tk], k)
                    np.add.at(counts[ki], tk, 1)
                    tail[ki] += sum(1 for x in tk if x in long) / k
                    if gs:
                        n_short += (1 if ki == 0 else 0)
                        hs = len(np.intersect1d(tk, list(gs)))
                        res["precision_short"][ki] += hs / k
                        res["recall_short"][ki] += hs / len(gs)
                        res["ndcg_short"][ki] += nb_ndcg([1 if x in gs else 0 for x in tk], k)
                    if gl:
                        n_long += (1 if ki == 0 else 0)
                        hl = len(np.intersect1d(tk, list(gl)))
                        res["precision_long"][ki] += hl / k
                        res["recall_long"][ki] += hl / len(gl)
                        res["ndcg_long"][ki] += nb_ndcg([1 if x in gl else 0 for x in tk], k)
    for m in ["precision", "recall", "ndcg"]:
        res[m] /= n_all
        if n_short: res[m + "_short"] /= n_short
        if n_long: res[m + "_long"] /= n_long
    for ki in range(len(topks)):
        res["gini"][ki] = gini_from_counts(counts[ki])
    res["tail_coverage"] = tail / n_all
    return res


# ------------------------------------------------------------------------ setup
make_synthetic("data/audit", n_users=250, n_items=180, seed=3)
set_seed(2020)
# val_frac=0 -> train set identical to the notebook's, so eval is directly comparable
ds = InteractionDataset("audit", root="data", val_frac=0.0, verbose=False)
model = LightGCN(ds.n_users, ds.n_items, dim=32, n_layers=3, graph=ds.sparse_graph(DEV))
topks = [10, 20]

print("\n1. LOSS EQUIVALENCE (BPR + L2 + manifold)")
rng = np.random.default_rng(0)
idx = rng.choice(len(ds.trainUser), 128, replace=False)
u = torch.from_numpy(ds.trainUser[idx]); p = torch.from_numpy(ds.trainItem[idx])
n = torch.from_numpy(rng.integers(0, ds.n_items, 128))
au, ai = model.computer()
L = build_embedding_laplacian(torch.cat([au, ai]), k=10, device=DEV, verbose=False)
for lam in [0.0, 1e-5, 1e-2]:
    model.zero_grad()
    nb_total, nb_bpr, nb_reg, nb_m = nb_stage_one_loss(model, u, p, n, 1e-4, L if lam > 0 else None, lam)
    nb_total.backward()
    g_nb = model.embedding_user.weight.grad.clone()
    model.zero_grad()
    au, ai = model.computer()
    port_total, logs = CompositeLoss([BPRTerm(), L2EgoTerm(1e-4), ManifoldTerm(lam)])(
        BatchContext(model, u, p, n, au, ai, 0, laplacian=L if lam > 0 else None))
    port_total.backward()
    g_port = model.embedding_user.weight.grad.clone()
    check(f"total loss lam={lam:g}",
          abs(nb_total.item() - port_total.item()) < 1e-9,
          f"nb={nb_total.item():.10f} port={port_total.item():.10f}")
    check(f"gradient   lam={lam:g}", torch.allclose(g_nb, g_port, atol=1e-9),
          f"max|diff|={(g_nb - g_port).abs().max().item():.2e}")

print("\n2. EVALUATION EQUIVALENCE (notebook Test vs ported evaluate)")
nb = nb_test(ds, model, topks)
au_, ai_ = model.computer()
port = evaluate(dot_product_scorer(au_, ai_), ds, "test", topks, batch_size=4096, device=DEV)
for ki, k in enumerate(topks):
    for m in ["recall", "precision", "recall_short", "precision_short", "recall_long", "precision_long"]:
        check(f"{m}@{k}", abs(nb[m][ki] - port[f"{m}@{k}"]) < 1e-9,
              f"nb={nb[m][ki]:.8f} port={port[f'{m}@{k}']:.8f}")
    check(f"gini@{k}", abs(nb["gini"][ki] - port[f"gini@{k}"]) < 1e-9,
          f"nb={nb['gini'][ki]:.8f} port={port[f'gini@{k}']:.8f}")
    check(f"tail_share_user@{k} (notebook tail_coverage)",
          abs(nb["tail_coverage"][ki] - port[f"tail_share_user@{k}"]) < 1e-9,
          f"nb={nb['tail_coverage'][ki]:.8f} port={port[f'tail_share_user@{k}']:.8f}")
    d = port[f"ndcg@{k}"] - nb["ndcg"][ki]
    print(f"  [FIX ] ndcg@{k} differs as intended: notebook={nb['ndcg'][ki]:.6f} "
          f"fixed={port[f'ndcg@{k}']:.6f} ({100*d/nb['ndcg'][ki]:+.1f}%)")

print("\n3. PROPAGATION / MODEL")
e = model.ego_embeddings()
manual = e.clone(); embs = [manual]
for _ in range(3):
    manual = torch.sparse.mm(ds.sparse_graph(DEV), manual); embs.append(manual)
ref = torch.stack(embs, 1).mean(1)
au, ai = model.computer()
check("LightGCN mean-over-layers propagation", torch.allclose(torch.cat([au, ai]), ref, atol=1e-6))
m0 = LightGCN(ds.n_users, ds.n_items, dim=32, n_layers=0, graph=ds.sparse_graph(DEV))
a0, i0 = m0.computer()
check("n_layers=0 reduces to MF-BPR (ego)", torch.allclose(torch.cat([a0, i0]), m0.ego_embeddings(), atol=1e-7))

print(f"\n{len(OK)} checks passed, {len(BAD)} drifted")
if BAD:
    print("DRIFT:", BAD)
