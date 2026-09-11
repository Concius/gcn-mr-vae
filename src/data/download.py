"""Download Gowalla / Yelp2018 / Amazon-Book splits from the official
LightGCN-PyTorch repository (ported from notebook Cell 3).

    python -m src.data.download            # all three
    python -m src.data.download gowalla    # one
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://github.com/gusye1234/LightGCN-PyTorch/raw/master/data"
DATASETS = ["gowalla", "yelp2018", "amazon-book"]
FILES = ["train.txt", "test.txt"]


def download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(desc=dest.name, total=total, unit="iB",
                                     unit_scale=True, unit_divisor=1024) as bar:
        for chunk in r.iter_content(chunk_size=1 << 16):
            bar.update(f.write(chunk))


def ensure_dataset(name: str, root: str | Path = "data") -> Path:
    root = Path(root)
    for fn in FILES:
        dest = root / name / fn
        if dest.exists():
            continue
        download_file(f"{BASE_URL}/{name}/{fn}", dest)
    return root / name


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = argv or DATASETS
    for n in names:
        p = ensure_dataset(n)
        print(f"{n}: ready at {p}")


if __name__ == "__main__":
    main()
