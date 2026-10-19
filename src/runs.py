"""Run directories and small, inspectable experiment records."""

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml


def prepare_run(cfg, *, smoke=False, run_dir=None, resume_path=None):
    if run_dir:
        path = Path(run_dir)
    elif resume_path:
        path = Path(resume_path).resolve().parent
        if path.name in {"checkpoints", "best_model"}:
            path = path.parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = (
            Path(cfg.run.log_dir)
            / ("smoke" if smoke else "experiments")
            / f"{stamp}_seed{cfg.run.seed}"
        )
    if path.exists() and any(path.iterdir()) and not resume_path:
        raise FileExistsError(f"Run directory is not empty: {path}. Choose a new --run-dir.")
    path.mkdir(parents=True, exist_ok=True)
    cfg.run.log_dir = str(path)
    return path


def record_run(cfg, model, *, source=None, mode="fresh"):
    """Store requested config and actual PPO settings, including inherited settings."""
    path = Path(cfg.run.log_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config = asdict(cfg)
    from .intrinsic import find_rnd

    rnd = find_rnd(model.get_env())
    if rnd is not None:
        config["intrinsic"].update(
            intrinsic_coef=rnd.intrinsic_coef,
            update_proportion=rnd.update_proportion,
            lr=rnd.optimizer.param_groups[0]["lr"],
        )
    actual = {
        name: getattr(model, name)
        for name in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "seed",
            "num_timesteps",
        )
    }
    actual.update(
        learning_rate_at_start=float(model.lr_schedule(1.0)),
        clip_range_at_start=float(model.clip_range(1.0)),
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    metadata = {
        "created_utc": stamp,
        "mode": mode,
        "source_checkpoint": source,
        "command": sys.argv,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(model.device),
        "actual_ppo": actual,
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "stable-baselines3",
                "gymnasium",
                "nes-py",
                "gym-super-mario-bros",
                "numpy",
            )
        },
    }
    (path / f"config_{stamp}.yaml").write_text(yaml.safe_dump(config))
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    (path / f"metadata_{stamp}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def validate_resume(cfg, checkpoint):
    """Prevent an accidental change of task or curiosity on an existing run."""
    from .config import load_config

    source = Path(checkpoint).resolve().parent
    if source.name in {"checkpoints", "best_model"}:
        source = source.parent
    saved = source / "config.yaml"
    if saved.exists():
        previous = load_config(str(saved))
        left, right = asdict(previous.env), asdict(cfg.env)
        for key in ("n_envs", "use_subproc"):
            left.pop(key)
            right.pop(key)
        if left != right:
            raise ValueError(
                "Resume environment/reward settings differ from the saved config. Use --init-from."
            )
        for key in ("use_intrinsic", "method", "feature_dim"):
            if getattr(previous.intrinsic, key) != getattr(cfg.intrinsic, key):
                raise ValueError(
                    f"Resume intrinsic.{key} differs from the saved config. Use --init-from."
                )
    if Path(checkpoint).with_suffix(".rnd.pt").exists() and not cfg.intrinsic.use_intrinsic:
        raise ValueError(
            "Checkpoint includes RND state but curiosity is disabled. Use --init-from."
        )
