"""Plan and record experiments without pretending to train a model."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from .config import validate_config


def make_plan(config):
    validate_config(config)
    return {
        "status": "blocked_dataset" if config["dataset"] is None else "planned",
        "execution_implemented": False,
        "dataset": config["dataset"],
        "runs": [{"training_seed": seed, "method": method, "split_seed": config["split_seed"],
                  "mc_passes": config["mc_passes"] if method == "mc_dropout" else 1}
                 for seed in config["training_seeds"]
                 for method in ("deterministic", "mc_dropout")],
        "stages": ["prepare_data", "train", "predict", "calibrate", "evaluate"],
        "pending_issues": [5, 7, 9, 10, 14, 19, 20, 21, 25],
    }


def record_manifest(config, output, repo):
    validate_config(config, require_dataset=True)
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("commit changes or remove untracked files before recording a run")
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    versions = {}
    for name in ("numpy", "torch", "torchvision", "pytest", "probabilistic-ml-uq"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {"schema_version": 1, "status": "prepared_not_executed", "git_commit": commit,
                "config_sha256": hashlib.sha256(payload).hexdigest(), "config": config,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": versions, "plan": make_plan(config), "metrics": None}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "plan", "record"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        if args.command == "validate":
            result = {"valid": True, "dataset_selected": config["dataset"] is not None}
        elif args.command == "plan":
            result = make_plan(config)
        else:
            if args.output is None:
                raise ValueError("record requires --output")
            result = record_manifest(config, args.output, args.repo)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
