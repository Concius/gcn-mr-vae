"""analysis/config_diff.py: the pre-flight check for reusing old runs."""
from analysis.config_diff import MISSING, composed, diff, flatten


def test_one_override_gives_exactly_one_difference():
    a = composed(["dataset=gowalla", "seed=2020", "arm=emb_mr"])
    b = composed(["dataset=gowalla", "seed=2020", "arm=emb_mr", "arm.lambda_manifold=1e-6"])
    assert diff(a, b) == [("arm.lambda_manifold", 1e-5, 1e-6)]


def test_paths_are_ignored_and_absent_keys_reported():
    a = {"paths": {"run_dir": "x"}, "optim": {"lr": 1e-3}}
    b = {"paths": {"run_dir": "y"}, "optim": {"lr": 1e-3, "new_knob": True}}
    assert diff(a, b) == [("optim.new_knob", MISSING, True)]
    assert flatten({"a": {"b": {"c": 1}}}) == {"a.b.c": 1}
