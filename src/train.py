"""PPO training, checkpoint resume, and policy transfer."""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from .callbacks import MarioMetricsCallback, RNDCheckpointCallback, SaveBestRNDCallback
from .config import Config, apply_smoke_overrides, linear_schedule, load_config, validate_config
from .env import build_config_env
from .intrinsic import load_rnd, save_rnd
from .runs import prepare_run, record_run, validate_resume


def build_envs(cfg: Config):
    """Construct the training and (extrinsic-only) evaluation vec envs."""
    train_venv = build_config_env(cfg, training=True)
    try:
        return train_venv, build_config_env(cfg)
    except Exception:
        train_venv.close()
        raise


def build_model(cfg: Config, train_venv) -> PPO:
    lr = (
        linear_schedule(cfg.ppo.learning_rate) if cfg.ppo.lr_linear_decay else cfg.ppo.learning_rate
    )
    clip_range = (
        linear_schedule(cfg.ppo.clip_range)
        if cfg.ppo.clip_range_linear_decay
        else cfg.ppo.clip_range
    )
    return PPO(
        policy="CnnPolicy",
        env=train_venv,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        learning_rate=lr,
        clip_range=clip_range,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        tensorboard_log=cfg.run.log_dir,
        seed=cfg.run.seed,
        device=cfg.run.device,
        verbose=1,
    )


def build_callbacks(cfg: Config, eval_venv):
    """Create periodic checkpoints, evaluation, and Mario metrics callbacks."""
    n_envs = cfg.env.n_envs
    ckpt_dir = os.path.join(cfg.run.log_dir, "checkpoints")
    best_dir = os.path.join(cfg.run.log_dir, "best_model")

    checkpoint_cb = RNDCheckpointCallback(
        save_freq=max(cfg.run.checkpoint_freq // n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_mario",
    )
    # Each invocation keeps its own evaluation history, including resumed runs.
    eval_root = os.path.join(cfg.run.log_dir, "eval")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=best_dir,
        log_path=os.path.join(eval_root, stamp),
        eval_freq=max(cfg.run.eval_freq // n_envs, 1),
        n_eval_episodes=cfg.run.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=SaveBestRNDCallback(best_dir),
    )
    for path in glob.glob(os.path.join(eval_root, "**", "evaluations.npz"), recursive=True):
        with np.load(path) as history:
            if history["results"].size:
                eval_cb.best_mean_reward = max(
                    eval_cb.best_mean_reward, float(history["results"].mean(axis=1).max())
                )
    return [checkpoint_cb, eval_cb, MarioMetricsCallback()]


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Find the highest-step checkpoint in the latest available run."""
    ckpts = glob.glob(os.path.join(log_dir, "checkpoints", "*_steps.zip"))
    if not ckpts:
        # Prefer the most recently modified experiment. smoke outputs are excluded.
        roots = glob.glob(os.path.join(log_dir, "experiments", "*"))
        roots.sort(key=os.path.getmtime, reverse=True)
        for root in roots:
            found = find_latest_checkpoint(root)
            if found:
                return found
    if not ckpts:
        return None

    def _step(path: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros (pixels-only).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to validate the pipeline.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training. '--resume' (or '--resume auto') continues from the "
        "newest checkpoint in the run's log_dir. '--resume <path.zip>' from a specific one.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Warm-start weights from another model .zip (e.g. a cleared level) but "
        "start a FRESH run (step count reset). Use for transfer learning to a new level.",
    )
    parser.add_argument(
        "--transfer-from",
        default=None,
        help="Partial weight transfer from a model with a DIFFERENT action space: copies "
        "every shape-matching tensor (the CNN vision, value head) and re-initializes the "
        "rest (the action head). Use when switching action_set (e.g. SIMPLE->COMPLEX).",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Override the level, e.g. '1-2'. Sets env_id=SuperMarioBros-<level>-v0 and "
        "log_dir=runs/ppo_world<W>_<S> so each level is its own run.",
    )
    parser.add_argument(
        "--run-dir", help="Explicit output directory. must be empty for a fresh run."
    )
    args = parser.parse_args()

    if sum(x is not None for x in (args.resume, args.init_from, args.transfer_from)) > 1:
        parser.error("--resume, --init-from, --transfer-from are mutually exclusive.")

    cfg = load_config(args.config)
    if args.level:
        w, s = args.level.split("-")
        cfg.env.env_id = f"SuperMarioBros-{w}-{s}-v0"
        cfg.run.log_dir = f"runs/ppo_world{w}_{s}"
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)

    set_random_seed(cfg.run.seed)

    # Resolve a resume checkpoint (if requested) before building the model.
    resume_path = None
    if args.resume is not None:
        resume_path = (
            find_latest_checkpoint(args.run_dir or cfg.run.log_dir)
            if args.resume == "auto"
            else args.resume
        )
        if resume_path is None or not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"--resume requested but no checkpoint found (looked for '{resume_path}')."
            )

    if args.init_from is not None and not os.path.exists(args.init_from):
        raise FileNotFoundError(f"--init-from model not found: {args.init_from}")

    if args.transfer_from is not None and not os.path.exists(args.transfer_from):
        raise FileNotFoundError(f"--transfer-from model not found: {args.transfer_from}")
    validate_config(cfg)
    if resume_path:
        validate_resume(cfg, resume_path)
    prepare_run(cfg, smoke=args.smoke, run_dir=args.run_dir, resume_path=resume_path)
    print(f"Run directory: {cfg.run.log_dir}")
    train_venv, eval_venv = build_envs(cfg)
    model = None
    try:
        model = initialize_model(cfg, train_venv, resume_path, args.init_from, args.transfer_from)
        record_run(
            cfg,
            model,
            source=resume_path or args.init_from or args.transfer_from,
            mode="resume"
            if resume_path
            else "warm_start"
            if args.init_from
            else "transfer"
            if args.transfer_from
            else "fresh",
        )
        model.learn(
            total_timesteps=cfg.ppo.total_timesteps,
            callback=build_callbacks(cfg, eval_venv),
            progress_bar=True,
            reset_num_timesteps=(resume_path is None),
        )
    finally:
        try:
            if model is not None:
                final_path = os.path.join(cfg.run.log_dir, "final_model.zip")
                model.save(final_path)
                save_rnd(train_venv, final_path)
                print(f"Saved final model to {final_path}")
        finally:
            train_venv.close()
            eval_venv.close()


def initialize_model(cfg, train_venv, resume_path=None, init_from=None, transfer_from=None):
    """Restore a saved run or copy policy weights into a fresh model."""
    if resume_path is not None:
        # Continue an existing run: load weights/optimizer and attach the (possibly
        # re-parallelized) env. reset_num_timesteps=False keeps the global step count,
        # so checkpoints/TensorBoard continue from where we left off.
        print(f"Resuming from {resume_path} (n_envs={cfg.env.n_envs}).")
        model = PPO.load(
            resume_path, env=train_venv, device=cfg.run.device, tensorboard_log=cfg.run.log_dir
        )
        load_rnd(train_venv, resume_path)
        print("Resume uses saved PPO settings. Metadata records actual settings.")
    elif init_from is not None:
        # Fresh optimizer and YAML settings. reuse only the learned policy weights.
        print(f"Warm-starting {cfg.env.env_id} from {init_from} (fresh step count).")
        source = PPO.load(init_from, device=cfg.run.device)
        if (
            source.observation_space != train_venv.observation_space
            or source.action_space != train_venv.action_space
        ):
            raise ValueError(
                "Warm-start spaces must match. Use --transfer-from for a different action space."
            )
        model = build_model(cfg, train_venv)
        model.policy.load_state_dict(source.policy.state_dict())
    elif transfer_from is not None:
        # Partial transfer across a DIFFERENT action space: build a fresh policy for the
        # new action_set, then copy every shape-matching tensor (CNN vision + value head)
        # from the source model. the action head (size differs) stays freshly initialized.
        if not os.path.exists(transfer_from):
            raise FileNotFoundError(f"--transfer-from model not found: {transfer_from}")
        print(
            f"Partial transfer from {transfer_from} into a fresh "
            f"{cfg.env.action_set} ({train_venv.action_space.n}-action) policy."
        )
        model = build_model(cfg, train_venv)
        old = PPO.load(transfer_from, device=cfg.run.device)
        new_sd, old_sd = model.policy.state_dict(), old.policy.state_dict()
        merged, copied, reinit = {}, 0, 0
        for k, v in new_sd.items():
            if k in old_sd and old_sd[k].shape == v.shape:
                merged[k] = old_sd[k]
                copied += 1
            else:
                merged[k] = v
                reinit += 1
        model.policy.load_state_dict(merged)
        print(f"  transferred {copied} tensors, re-initialized {reinit} (action head etc.)")
        del old
    else:
        model = build_model(cfg, train_venv)
    return model


if __name__ == "__main__":
    main()
