"""Watch a policy play and optionally follow new checkpoints."""

from __future__ import annotations

import argparse
import glob
import os
import time

from stable_baselines3 import PPO

from .config import load_config
from .env import build_config_env


def _candidate_model_paths(log_dir: str):
    """All plausible saved-model paths, newest first."""
    patterns = [
        os.path.join(log_dir, "best_model", "best_model.zip"),
        os.path.join(log_dir, "checkpoints", "*.zip"),
        os.path.join(log_dir, "final_model.zip"),
        os.path.join(log_dir, "experiments", "*", "checkpoints", "*.zip"),
        os.path.join(log_dir, "experiments", "*", "best_model", "best_model.zip"),
        os.path.join(log_dir, "experiments", "*", "final_model.zip"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    # De-dup while keeping order, then sort by mtime (newest first).
    paths = list(dict.fromkeys(paths))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def _newest_model(log_dir: str):
    paths = _candidate_model_paths(log_dir)
    return paths[0] if paths else None


def _wait_for_model(log_dir: str, timeout: float = 600.0) -> str:
    """Block until a model file exists (the first checkpoint at ~100k steps)."""
    start = time.time()
    waited = False
    while True:
        path = _newest_model(log_dir)
        if path is not None:
            if waited:
                print(f"Found model: {path}")
            return path
        if time.time() - start > timeout:
            raise TimeoutError(
                f"No model appeared under {log_dir} within {timeout:.0f}s. "
                "Has training saved a checkpoint yet (first one is at 100k steps)?"
            )
        if not waited:
            print(f"No checkpoint yet under {log_dir}. waiting for the first one...")
            waited = True
        time.sleep(3.0)


def watch(
    config_path: str,
    model_path: str | None,
    episodes: int,
    follow: bool,
    deterministic: bool,
    speed: float = 1.0,
) -> None:
    cfg = load_config(config_path)

    # Real-time pacing. One agent decision covers `frame_skip` emulator frames. the
    # NES runs ~60 fps, so pace each decision to frame_skip/60 s (divided by `speed`)
    # — otherwise render_mode="human" runs unthrottled and the window is a blur.
    step_dt = (cfg.env.frame_skip / 60.0) / max(speed, 1e-6)

    if model_path is None:
        model_path = _wait_for_model(cfg.run.log_dir)

    # Single env, on-screen rendering. Same preprocessing + transpose as training
    # so the loaded policy's observation space matches.
    venv = build_config_env(cfg, render_mode="human", seed_offset=123)

    model = PPO.load(model_path, device=cfg.run.device)
    print(f"Watching: {model_path}  (action_set={cfg.env.action_set})")
    print("Close the game window or press Ctrl-C to stop.")

    try:
        ep = 0
        while episodes <= 0 or ep < episodes:
            # In --follow mode, hot-swap to the newest checkpoint between episodes
            # so the window tracks training progress as it improves.
            if follow:
                newest = _newest_model(cfg.run.log_dir)
                if newest and newest != model_path:
                    model_path = newest
                    model = PPO.load(model_path, device=cfg.run.device)
                    print(f"[follow] switched to newer model: {os.path.basename(model_path)}")

            obs = venv.reset()
            done = False
            total_reward = 0.0
            max_x = 0
            flag = False
            while not done:
                t0 = time.time()
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = venv.step(action)
                total_reward += float(rewards[0])
                done = bool(dones[0])
                info = infos[0]
                if "x_pos" in info:
                    max_x = max(max_x, int(info["x_pos"]))
                flag = flag or bool(info.get("flag_get", False))
                # Throttle to real-time so the window is watchable.
                sleep_for = step_dt - (time.time() - t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            ep += 1
            print(
                f"Episode {ep}: reward={total_reward:.0f} max_x_pos={max_x} "
                f"flag_get={flag}  [{os.path.basename(model_path)}]"
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        venv.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Watch a Mario PPO agent play in a live window.")
    p.add_argument("--config", required=True, help="YAML config (must match training env).")
    p.add_argument(
        "--model",
        default=None,
        help="Model .zip to load. Default: newest under log_dir (waits if none yet).",
    )
    p.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 = loop forever).")
    p.add_argument(
        "--follow",
        action="store_true",
        help="Reload the newest checkpoint between episodes (watch it improve).",
    )
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax.")
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time NES speed).",
    )
    args = p.parse_args()

    watch(
        config_path=args.config,
        model_path=args.model,
        episodes=args.episodes,
        follow=args.follow,
        deterministic=not args.stochastic,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
