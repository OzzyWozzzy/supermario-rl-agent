"""Run a matched PPO and PPO + RND comparison on stage 1-1."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import apply_smoke_overrides, load_config, validate_config

METHODS = {"ppo": False, "ppo_rnd": True}


def comparison_configs(config_path, seed, timesteps, n_envs=4, pilot=False):
    """Build matched settings that differ only in the curiosity flag."""
    configs = {}
    for method, curiosity in METHODS.items():
        cfg = load_config(config_path)
        if cfg.env.env_id != "SuperMarioBros-1-1-v0" or cfg.env.humanlike or cfg.env.coverage:
            raise ValueError("Comparison requires stage 1-1 with base rewards only")
        cfg.run.seed = seed
        cfg.env.n_envs = n_envs
        cfg.ppo.total_timesteps = timesteps
        cfg.intrinsic.use_intrinsic = curiosity
        if pilot:
            apply_smoke_overrides(cfg)
            cfg.ppo.total_timesteps = 2048
            cfg.run.eval_freq = 1024
            cfg.run.checkpoint_freq = 1024
            cfg.run.n_eval_episodes = 3
        rollout = cfg.ppo.n_steps * cfg.env.n_envs
        cfg.ppo.total_timesteps = ((cfg.ppo.total_timesteps + rollout - 1) // rollout) * rollout
        validate_config(cfg)
        configs[method] = cfg
    return configs


def write_manifest(path, manifest):
    """Replace the progress record after each completed step."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def run_comparison(config_path, output, seeds, timesteps, n_envs=4, episodes=10, pilot=False):
    """Train both methods for each seed and evaluate their final checkpoints."""
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide distinct nonnegative seeds")
    if min(timesteps, n_envs, episodes) < 1:
        raise ValueError("timesteps, n_envs and episodes must be positive")
    # Validate all settings before creating files or starting training.
    settings = {
        seed: comparison_configs(config_path, seed, timesteps, n_envs, pilot) for seed in seeds
    }
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    manifest = {
        "kind": "pilot" if pilot else "experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path),
        "requested_timesteps": 2048 if pilot else timesteps,
        "evaluation_episodes": episodes,
        "checkpoint_selection": "final",
        "evaluation_actions": "deterministic",
        "runs": [],
    }
    manifest_path = root / "comparison.json"
    write_manifest(manifest_path, manifest)
    for index, seed in enumerate(seeds):
        # Alternate method order to reduce a fixed ordering effect on timing.
        methods = list(METHODS) if index % 2 == 0 else list(reversed(METHODS))
        for method in methods:
            cfg = settings[seed][method]
            run_dir = root / f"{method}_seed{seed}"
            config = inputs / f"{method}_seed{seed}.yaml"
            config.write_text(yaml.safe_dump(asdict(cfg)))
            record = {
                "method": method,
                "seed": seed,
                "directory": run_dir.name,
                "budget": cfg.ppo.total_timesteps,
                "status": "running",
            }
            manifest["runs"].append(record)
            write_manifest(manifest_path, manifest)
            print(
                f"Training {method}, seed {seed}, {cfg.ppo.total_timesteps:,} transitions",
                flush=True,
            )
            started = time.perf_counter()
            try:
                with (root / f"{run_dir.name}.train.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "src.train",
                            "--config",
                            str(config),
                            "--run-dir",
                            str(run_dir),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                record["training_seconds"] = time.perf_counter() - started
                with (root / f"{run_dir.name}.eval.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "src.evaluate",
                            "--config",
                            str(run_dir / "config.yaml"),
                            "--model",
                            str(run_dir / "final_model.zip"),
                            "--episodes",
                            str(episodes),
                            "--metrics",
                            str(run_dir / "evaluation.json"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                record["status"] = "complete"
            except BaseException:
                record["status"] = "failed"
                write_manifest(manifest_path, manifest)
                raise
            write_manifest(manifest_path, manifest)
            print(f"Completed {run_dir.name} in {record['training_seconds']:.1f}s", flush=True)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare PPO and PPO + RND under matched settings."
    )
    parser.add_argument("--config", default="configs/ppo_world1_1.yaml")
    parser.add_argument("--output", help="New directory for the comparison and its models.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--pilot", action="store_true", help="Use 2,048 transitions and small rollouts."
    )
    parser.add_argument("--report-dir", help="New directory for portable metrics and figures.")
    args = parser.parse_args()
    if args.report_dir and Path(args.report_dir).exists():
        parser.error("--report-dir must be a new directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or f"runs/comparisons/{'pilot' if args.pilot else 'experiment'}_{stamp}"
    manifest = run_comparison(
        args.config, output, args.seeds, args.timesteps, args.n_envs, args.episodes, args.pilot
    )
    from .analyze import build_report

    build_report(manifest, args.report_dir or manifest.parent / "report")
    print(f"Comparison record: {manifest}")


if __name__ == "__main__":
    main()
