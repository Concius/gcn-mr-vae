"""Would a new command reproduce an existing run's configuration?

Composes the config that `python -m src.train <overrides>` WOULD run (nothing
is trained) and compares it, key by key, with the config stored in an existing
run's results.json. Use it before reusing old runs as arms or controls of a
new experiment: anything other than the intended differences means the old
and new runs are not comparable.

    python -m analysis.config_diff runs/gowalla/emb_mr_w100_lam1e-05_k20_r50_K3_d256/seed2020 \\
        dataset=gowalla seed=2020 arm=emb_mr arm.lambda_manifold=1e-6

`paths.*` is ignored (run folders legitimately differ). Keys present on only
one side are reported: a key added to config.yaml after the old run means the
old run used whatever the code did before that key existed -- check that its
default reproduces that behaviour.
"""
from __future__ import annotations

import json
import pathlib
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import src.train  # noqa: F401  registers the armtag resolver

ROOT = pathlib.Path(__file__).resolve().parents[1]
MISSING = "<absent>"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def composed(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def diff(stored: dict, new: dict, ignore=("paths.",)):
    a, b = flatten(stored), flatten(new)
    rows = []
    for k in sorted(set(a) | set(b)):
        if k.startswith(ignore):
            continue
        va, vb = a.get(k, MISSING), b.get(k, MISSING)
        if va != vb:
            rows.append((k, va, vb))
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__); return 2
    run, overrides = pathlib.Path(argv[0]), argv[1:]
    stored = json.load(open(run / "results.json"))["config"]
    rows = diff(stored, composed(overrides))
    print(f"stored: {run}\nnew:    {' '.join(overrides)}")
    if not rows:
        print("identical apart from paths.*")
    else:
        w = max(len(k) for k, _, _ in rows)
        print(f"{len(rows)} difference(s) (paths.* ignored):")
        for k, va, vb in rows:
            print(f"  {k:{w}s}  stored={va!r:<24}  new={vb!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
