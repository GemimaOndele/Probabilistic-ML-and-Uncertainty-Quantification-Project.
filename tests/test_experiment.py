from copy import deepcopy
import json
from pathlib import Path
import subprocess
import pytest
from experiment.cli import main, make_plan, record_manifest
from experiment.config import validate_config


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())


def test_pending_dataset_is_valid_but_cannot_record(config, tmp_path):
    assert make_plan(config)["status"] == "blocked_dataset"
    with pytest.raises(ValueError, match="dataset pending"):
        record_manifest(config, tmp_path / "run.json", tmp_path)
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("field,value", [
    ("dataset", "unknown"), ("split_seed", True), ("mc_passes", 0),
    ("ece_bins", -1), ("training_seeds", [42,42]), ("training_seeds", []),
    ("device", "magic"), ("preserve_official_test", False),
    ("train_pool_fractions", {"train": .9, "validation": .1, "calibration": .1}),
    ("train_pool_fractions", {"train": float("nan"), "validation": .1, "calibration": .1}),
])
def test_invalid_config(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_plan_pairs_methods_without_execution(config):
    plan = make_plan(config)
    assert len(plan["runs"]) == 6
    assert plan["execution_implemented"] is False
    assert plan == make_plan(deepcopy(config))
    for seed in config["training_seeds"]:
        runs = [r for r in plan["runs"] if r["training_seed"] == seed]
        assert {r["method"] for r in runs} == {"deterministic", "mc_dropout"}
        assert {r["split_seed"] for r in runs} == {42}


def test_manifest_provenance_no_overwrite_and_dirty_rejection(config, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init")
    (repo / "README.md").write_text("test fixture")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    config["dataset"] = "Fashion-MNIST"  # Fixture only; not a team decision.
    target = tmp_path / "manifest.json"
    result = record_manifest(config, target, repo)
    assert result["git_commit"] == git("rev-parse", "HEAD")
    assert result["metrics"] is None
    assert result["status"] == "prepared_not_executed"
    assert json.loads(target.read_text()) == result
    with pytest.raises(FileExistsError):
        record_manifest(config, target, repo)
    (repo / "untracked.txt").write_text("dirty")
    with pytest.raises(ValueError, match="untracked"):
        record_manifest(config, tmp_path / "another.json", repo)


def test_cli_config_error_returns_nonzero(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{")
    assert main(["validate", "--config", str(path)]) == 2
    assert "error:" in capsys.readouterr().err
