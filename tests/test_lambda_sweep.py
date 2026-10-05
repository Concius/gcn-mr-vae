"""The lambda sweep's pre-registered rules (analysis/lambda_sweep.py).

The rules are pure functions so each can be pinned exactly; the end-to-end
tests write results.json files with a known answer into the real folder
layout and read the script's output."""
import contextlib
import io
import json

import pytest

from analysis import lambda_sweep as L
from src.train import _arm_tag


# ---------------------------------------------------------------- pure rules
@pytest.mark.parametrize("x,expected", [(2.94e-6, 2.9e-6), (2.96e-6, 3.0e-6), (1.234e-5, 1.2e-5),
                                        (0.000123, 0.00012), (0.0, 0.0), (7.0e-6, 7.0e-6)])
def test_round_sig(x, expected):
    assert L.round_sig(x) == pytest.approx(expected, rel=1e-12)


def test_matched_lambda_interpolates_linearly_in_lambda():
    lams = [0.0, 1e-6, 3e-6, 1e-5]
    er = [100.0, 99.9, 99.7, 99.0]                       # decreasing
    lam, seg = L.matched_lambda(lams, er, 99.35)          # halfway between 3e-6 and 1e-5
    assert seg == (3e-6, 1e-5) and lam == pytest.approx(6.5e-6)


def test_matched_lambda_uses_the_zero_segment():
    lam, seg = L.matched_lambda([0.0, 1e-6, 1e-5], [100.0, 99.0, 98.0], 99.5)
    assert seg == (0.0, 1e-6) and lam == pytest.approx(5e-7)


def test_matched_lambda_endpoint_and_flat_segment():
    assert L.matched_lambda([0.0, 1e-6, 3e-6], [100.0, 99.0, 98.0], 99.0)[0] == pytest.approx(1e-6)
    assert L.matched_lambda([0.0, 1e-6, 3e-6], [100.0, 99.0, 99.0], 99.0)[0] == pytest.approx(1e-6)


def test_matched_lambda_takes_the_first_crossing_when_not_monotone():
    lams = [0.0, 1e-6, 3e-6, 1e-5]
    er = [100.0, 99.0, 99.8, 98.0]                        # 99.5 is crossed twice
    lam, seg = L.matched_lambda(lams, er, 99.5)
    assert seg == (0.0, 1e-6) and lam == pytest.approx(5e-7)


def test_matched_lambda_sorts_its_input_and_reports_no_match():
    lam, seg = L.matched_lambda([1e-5, 0.0, 3e-6], [99.0, 100.0, 99.7], 99.35)
    assert seg == (3e-6, 1e-5) and lam == pytest.approx(6.5e-6)
    assert L.matched_lambda([0.0, 1e-5], [100.0, 99.0], 101.0) == (None, None)
    assert L.matched_lambda([0.0, 1e-5], [100.0, 99.0], 98.0) == (None, None)


def test_select_lambda_argmax_and_ties_to_smaller():
    assert L.select_lambda([0.0, 1e-6, 1e-5], [0.10, 0.12, 0.11]) == 1e-6
    assert L.select_lambda([0.0, 1e-6, 1e-5], [0.12, 0.10, 0.12]) == 0.0
    assert L.select_lambda([1e-5, 1e-6, 0.0], [0.12, 0.12, 0.11]) == 1e-6


def test_on_grid():
    g = (1e-6, 3e-6, 1e-5)
    assert L.on_grid(3e-6, g) == 3e-6 and L.on_grid(0.0, g) == 0.0
    assert L.on_grid(3e-6 * (1 + 1e-12), g) == 3e-6
    assert L.on_grid(2.9e-6, g) is None


# ---------------------------------------------------------------- end to end
GRID = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
SEEDS = list(range(2020, 2030))


def _write(root, name, lam, seed, er, np_ref, recall, val, gini=0.5, tail=0.3):
    tag = _arm_tag(name, lam, 20, "none", 100, 50, 3, 256)
    d = root / "gowalla" / tag / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": seed, "val": {"recall@20": val},
               "test": {"recall@20": recall, "gini@20": gini, "tail_catalog_coverage@20": tail},
               "geometry": {"er_table": er, "np_ref@20": np_ref}}, open(d / "results.json", "w"))
    json.dump([{"epoch": e, "event": "eval", "loss_total": 0.5, "loss_bpr": 0.5} for e in (10, 20)],
              open(d / "history.json", "w"))


def _curve(root, val_slope=-1.0, dw_er=99.7, dw_np=0.495):
    """ER = 100 - 1e5*lambda, np_ref = 0.5 - 1e3*lambda, plus a seed offset
    shared by every arm (so paired differences are exact)."""
    for i, s in enumerate(SEEDS):
        off = 0.01 * i
        for lam in (0.0,) + GRID:
            name = "mr_off" if lam == 0 else "emb_mr"
            _write(root, name, lam, s, er=100 - 1e5 * lam + off, np_ref=0.5 - 1e3 * lam + off / 100,
                   recall=0.18 + off / 100, val=0.17 + val_slope * lam + off / 100 + 1e-9 * i * lam)
        _write(root, "emb_mr_dw", 1e-5, s, er=dw_er + off, np_ref=dw_np + off / 100,
               recall=0.18 + off / 100, val=0.17 + off / 100)


def _run(root, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.main(["--runs", str(root), *extra])
    return buf.getvalue()


def test_incomplete_prints_counts_only(tmp_path):
    _curve(tmp_path)
    tag = _arm_tag("emb_mr", 3e-5, 20, "none", 100, 50, 3, 256)
    (tmp_path / "gowalla" / tag / "seed2025" / "results.json").unlink()
    out = _run(tmp_path)
    assert "INCOMPLETE" in out and "STAGE 1" not in out and "lambda*" not in out


def test_matched_lambda_reuses_grid_runs(tmp_path):
    _curve(tmp_path, dw_er=99.7)                        # ER(3e-6) = 99.7 exactly
    out = _run(tmp_path)
    assert "ER-matched lambda'  : 3e-06" in out
    assert "reused from stage 1" in out and "PRIMARY" in out


def test_stage2_command_off_grid_then_test(tmp_path):
    _curve(tmp_path, dw_er=99.65, dw_np=0.4975)          # lambda' = 3.5e-6 on ER
    out = _run(tmp_path)
    assert "ER-matched lambda'  : 3.5e-06" in out
    assert "arm.lambda_manifold=3.5e-06 seed=2020,2021" in out and "PRIMARY" not in out
    # descriptive np_ref match: 0.5 - 1e3*lam = 0.4975 -> 2.5e-6, i.e. NOT the ER-matched lambda
    assert "np_ref-matched (descr.): 2.5e-06" in out
    for i, s in enumerate(SEEDS):                        # stage-2 runs, np_ref above dw by 0.001
        off = 0.01 * i
        _write(tmp_path, "emb_mr", 3.5e-6, s, er=99.65 + off, np_ref=0.4985 + off / 100,
               recall=0.18 + off / 100, val=0.17)
    out = _run(tmp_path)
    prim = out.split("PRIMARY")[1].splitlines()[1]
    assert "np_ref@20" in prim and "-0.00100" in prim and "10/0" in prim and "*" in prim
    check = out.split("MATCHING CHECK")[1].splitlines()[1]
    assert "+0.00000" in check


def test_selection_and_boundary_rule(tmp_path):
    _curve(tmp_path, val_slope=+1.0)                     # validation recall rises with lambda
    out = _run(tmp_path)
    assert "lambda* = 0.0001" in out and "BOUNDARY" in out
    _curve(tmp_path, val_slope=-1.0)                     # falls with lambda: no MR wins
    out = _run(tmp_path)
    assert "lambda* = 0 " in out and "BOUNDARY" not in out


def test_dose_response_holm_within_metric(tmp_path):
    _curve(tmp_path)
    out = _run(tmp_path)
    block = out.split("DOSE-RESPONSE")[1].split("SELECTION")[0]
    er_lines = block.split("er_table")[1].split("np_ref@20")[0].strip().splitlines()
    assert len(er_lines) == len(GRID)
    # every lambda lowers ER in all 10 seeds: p = 2/1024 each, Holm x5 = 0.0098
    assert all("10/0" in l and "Holm=0.0098" in l for l in er_lines)


def _tag_dir(root, name, lam, seed):
    return root / "gowalla" / _arm_tag(name, lam, 20, "none", 100, 50, 3, 256) / f"seed{seed}"


def test_health_gate_stops_on_a_diverged_run(tmp_path):
    """A run whose loss went non-finite late still reports an earlier
    checkpoint; the gate must catch it from history.json."""
    _curve(tmp_path)
    d = _tag_dir(tmp_path, "emb_mr", 1e-4, 2023)
    json.dump([{"epoch": 10, "event": "eval", "loss_total": 0.5},
               {"epoch": 640, "event": "eval", "loss_total": float("nan")}], open(d / "history.json", "w"))
    out = _run(tmp_path)
    assert "UNHEALTHY" in out and "seed2023: loss_total = nan at epoch 640" in out
    assert "STAGE 1" not in out and "lambda*" not in out


def test_health_gate_catches_non_finite_metrics_and_missing_history(tmp_path):
    _curve(tmp_path)
    d = _tag_dir(tmp_path, "emb_mr_dw", 1e-5, 2027)
    r = json.load(open(d / "results.json")); r["geometry"]["er_table"] = float("inf")
    json.dump(r, open(d / "results.json", "w"))
    (_tag_dir(tmp_path, "mr_off", 0.0, 2021) / "history.json").unlink()
    out = _run(tmp_path)
    assert "seed2027: geometry.er_table = inf" in out and "seed2021: history.json missing" in out
    assert "STAGE 1" not in out


def test_healthy_runs_pass_the_gate(tmp_path):
    _curve(tmp_path)
    assert "all losses and metrics finite" in _run(tmp_path)


def test_stage2_runs_must_be_healthy(tmp_path):
    _curve(tmp_path, dw_er=99.65, dw_np=0.4975)          # lambda' = 3.5e-6, off grid
    for i, s in enumerate(SEEDS):
        _write(tmp_path, "emb_mr", 3.5e-6, s, er=99.65, np_ref=0.4985, recall=0.18, val=0.17)
    d = _tag_dir(tmp_path, "emb_mr", 3.5e-6, 2029)
    json.dump([{"epoch": 300, "event": "eval", "loss_total": float("inf")}], open(d / "history.json", "w"))
    out = _run(tmp_path)
    assert "seed2029: loss_total = inf at epoch 300" in out and "PRIMARY" not in out


def test_boundary_prints_the_extension_commands(tmp_path):
    _curve(tmp_path, val_slope=+1.0)
    out = _run(tmp_path)
    assert "arm.lambda_manifold=0.0003,0.001" in out
    assert "--grid 1e-06 3e-06 1e-05 3e-05 0.0001 0.0003 0.001" in out


def test_boundary_after_extension_does_not_extend_again(tmp_path):
    _curve(tmp_path, val_slope=+1.0)
    for i, s in enumerate(SEEDS):                        # the extension runs, still rising
        off = 0.01 * i
        for lam in (3e-4, 1e-3):
            _write(tmp_path, "emb_mr", lam, s, er=100 - 1e5 * lam + off, np_ref=0.5 - 1e3 * lam + off / 100,
                   recall=0.18, val=0.17 + lam + off / 100)
    out = _run(tmp_path, "--grid", "1e-6", "3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3")
    assert "lambda* = 0.001" in out and "boundary selection" in out
    assert "arm.lambda_manifold=\n" not in out and "arm.lambda_manifold= " not in out
    er = out.split("DOSE-RESPONSE")[1].split("np_ref@20")[0]
    assert er.count("lambda=") == 7                      # Holm family is the seven lambdas run


def test_curve_csv(tmp_path):
    _curve(tmp_path)
    _run(tmp_path, "--out", str(tmp_path / "out"))
    rows = (tmp_path / "out" / "gowalla_curve.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + len(SEEDS) * (len(GRID) + 2)
