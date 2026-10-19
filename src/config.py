"""Typed YAML configuration, validation, and training schedules."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import yaml


@dataclass
class EnvConfig:
    env_id: str = "SuperMarioBros-1-1-v0"
    action_set: str = "SIMPLE_MOVEMENT"  # RIGHT_ONLY | SIMPLE_MOVEMENT | COMPLEX_MOVEMENT
    n_envs: int = 8
    frame_skip: int = 4
    frame_stack: int = 4
    reward_scale: float = 1.0
    reward_clip: Optional[float] = None
    use_subproc: bool = True
    # Human-like reward shaping: reward score/coins/powerups, not just rushing right.
    humanlike: bool = False
    score_coef: float = 0.01
    coin_coef: float = 2.0
    powerup_coef: float = 10.0
    # Go-Explore-style cell-coverage exploration bonus.
    coverage: bool = False
    coverage_coef: float = 0.5


@dataclass
class PPOConfig:
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.9
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    lr_linear_decay: bool = True
    clip_range: float = 0.2
    clip_range_linear_decay: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 5_000_000


@dataclass
class IntrinsicConfig:
    use_intrinsic: bool = False
    method: str = "rnd"  # only "rnd" supported
    intrinsic_coef: float = 0.5  # weight on the (normalised) intrinsic reward
    lr: float = 1e-4  # predictor learning rate
    feature_dim: int = 512  # embedding size of target/predictor heads
    update_proportion: float = 1.0  # fraction of incoming frames used per predictor step


@dataclass
class RunConfig:
    seed: int = 0
    log_dir: str = "runs/ppo_world1_1"
    checkpoint_freq: int = 100_000  # in total agent transitions across all envs
    eval_freq: int = 100_000
    n_eval_episodes: int = 5
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _filter_known(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Reject unknown keys and non-mapping configuration sections."""
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


def load_config(path: str) -> Config:
    """Load a YAML file into a typed Config, applying dataclass defaults."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
        if raw is None:
            raw = {}

    raw = _filter_known(Config, raw)
    cfg = Config(
        env=EnvConfig(**_filter_known(EnvConfig, raw.get("env", {}))),
        ppo=PPOConfig(**_filter_known(PPOConfig, raw.get("ppo", {}))),
        intrinsic=IntrinsicConfig(**_filter_known(IntrinsicConfig, raw.get("intrinsic", {}))),
        run=RunConfig(**_filter_known(RunConfig, raw.get("run", {}))),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Reject invalid settings before constructing environments or networks."""
    for section in dataclasses.fields(cfg):
        obj = getattr(cfg, section.name)
        for config_field in dataclasses.fields(obj):
            value = getattr(obj, config_field.name)
            annotation = config_field.type
            if annotation == "bool" and type(value) is not bool:
                raise ValueError(f"{section.name}.{config_field.name} must be a boolean")
            if annotation == "int" and type(value) is not int:
                raise ValueError(f"{section.name}.{config_field.name} must be an integer")
            if annotation == "str" and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{section.name}.{config_field.name} must be a nonempty string")
            if annotation == "float" and value is None:
                raise ValueError(f"{section.name}.{config_field.name} must be a finite number")
            if annotation in {"float", "Optional[float]"} and value is not None:
                if type(value) not in {float, int} or not math.isfinite(value):
                    raise ValueError(f"{section.name}.{config_field.name} must be a finite number")

    for label, value in {
        "n_envs": cfg.env.n_envs,
        "frame_skip": cfg.env.frame_skip,
        "frame_stack": cfg.env.frame_stack,
        "n_steps": cfg.ppo.n_steps,
        "batch_size": cfg.ppo.batch_size,
        "n_epochs": cfg.ppo.n_epochs,
        "total_timesteps": cfg.ppo.total_timesteps,
        "checkpoint_freq": cfg.run.checkpoint_freq,
        "eval_freq": cfg.run.eval_freq,
        "n_eval_episodes": cfg.run.n_eval_episodes,
        "feature_dim": cfg.intrinsic.feature_dim,
    }.items():
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if cfg.run.seed < 0:
        raise ValueError("seed must be nonnegative")
    if cfg.env.action_set not in {"RIGHT_ONLY", "SIMPLE_MOVEMENT", "COMPLEX_MOVEMENT"}:
        raise ValueError("Unsupported action_set")
    if cfg.intrinsic.method != "rnd":
        raise ValueError("Only intrinsic.method: rnd is supported")
    if cfg.ppo.batch_size < 2 or cfg.ppo.n_steps * cfg.env.n_envs < cfg.ppo.batch_size:
        raise ValueError("batch_size must be >= 2 and fit inside a rollout")
    for label, value in {
        "gamma": cfg.ppo.gamma,
        "gae_lambda": cfg.ppo.gae_lambda,
        "update_proportion": cfg.intrinsic.update_proportion,
    }.items():
        if not 0 < value <= 1:
            raise ValueError(f"{label} must be in (0, 1]")
    for label, value in {
        "learning_rate": cfg.ppo.learning_rate,
        "clip_range": cfg.ppo.clip_range,
        "max_grad_norm": cfg.ppo.max_grad_norm,
        "intrinsic.lr": cfg.intrinsic.lr,
    }.items():
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    for label, value in {
        "ent_coef": cfg.ppo.ent_coef,
        "vf_coef": cfg.ppo.vf_coef,
        "intrinsic_coef": cfg.intrinsic.intrinsic_coef,
        "score_coef": cfg.env.score_coef,
        "coin_coef": cfg.env.coin_coef,
        "powerup_coef": cfg.env.powerup_coef,
        "coverage_coef": cfg.env.coverage_coef,
    }.items():
        if value < 0:
            raise ValueError(f"{label} must be nonnegative")
    if cfg.env.reward_clip is not None and cfg.env.reward_clip <= 0:
        raise ValueError("reward_clip must be positive or null")


def apply_smoke_overrides(cfg: Config) -> Config:
    """Reduce training size and evaluation frequency for a quick pipeline check."""
    cfg.env.n_envs = min(cfg.env.n_envs, 2)
    cfg.env.use_subproc = False
    cfg.ppo.n_steps = 64
    cfg.ppo.batch_size = 64
    cfg.ppo.n_epochs = 2
    cfg.ppo.total_timesteps = 2_000
    cfg.run.checkpoint_freq = 1_000
    cfg.run.eval_freq = 1_000
    cfg.run.n_eval_episodes = 1
    return cfg


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """Interpolate from the initial value to the final value as training progresses."""

    def _schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return _schedule
