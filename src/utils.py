"""Small shared utilities (ported from notebook Cell 2 / Cell 4).

Everything here is side-effect free at import time. No global config object:
the notebook's global ``config`` was a source of silent state leakage
(override/restore patterns, ``config.device`` referenced from inside classes)
and does not exist in this codebase.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    ``deterministic=False`` reproduces the notebook setting
    (cudnn.benchmark=True, cudnn.deterministic=False). Sparse matmul on CUDA
    is non-deterministic regardless, so two runs with the same seed can differ
    at float precision. Paired-by-seed comparisons remain valid; bit-identical
    replay is not promised.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available. "
                           "Use device=cpu (tests) or fix the CUDA install "
                           "(RTX 5060 Ti needs a cu128+ PyTorch wheel).")
    return torch.device(name)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Timer:
    """``with Timer() as t: ...; t.elapsed``"""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_dump(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def json_load(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def rng_state_dict() -> dict:
    """Capture python/numpy/torch RNG state (for exact warm-start resume)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def rng_state_load(state: dict) -> None:
    """Restore RNG state saved by :func:`rng_state_dict`.

    The tensors are forced back to CPU first. ``torch.load(map_location=cuda)``
    moves *every* tensor in a checkpoint to the GPU, including these ByteTensor
    RNG states, and ``torch.set_rng_state`` only accepts a CPU ByteTensor — so
    resuming a warm-start on a GPU would otherwise raise here even though the
    same code path is fine on CPU.
    """
    def _cpu(t):
        return t.cpu() if torch.is_tensor(t) else t

    random.setstate(_tuplify(state["python"]))
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu(state["torch"]).to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu(t).to(torch.uint8) for t in state["cuda"]])


def _tuplify(obj):
    """``random.setstate`` needs tuples; JSON/torch round-trips can give lists."""
    if isinstance(obj, list):
        return tuple(_tuplify(o) for o in obj)
    return obj
