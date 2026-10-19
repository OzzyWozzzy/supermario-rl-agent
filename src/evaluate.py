"""Checkpoint evaluation with episode metrics and RGB video."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from .config import load_config
from .env import build_config_env


def _get_rgb_frame(venv) -> np.ndarray:
    """Grab the current full-resolution RGB frame from the (single) underlying env."""
    # VecEnvWrappers forward get_images() down to the base env's rgb_array render.
    # .copy() snapshots the reused emulator screen buffer (else frames alias to one).
    images = venv.get_images()
    return np.asarray(images[0]).copy()


def evaluate(
    config_path: str,
    model_path: str,
    episodes: int = 3,
    video_path: str | None = None,
    deterministic: bool = True,
    fps: float | None = None,
    metrics_path: str | None = None,
) -> dict:
    if episodes < 1 or (fps is not None and fps <= 0):
        raise ValueError("episodes and fps must be positive")
    cfg = load_config(config_path)
    set_random_seed(cfg.run.seed)
    fps = fps or 60 / cfg.env.frame_skip

    # Single-env evaluation stack, mirroring training preprocessing + transpose so
    # the model's observation space matches. No intrinsic wrapper at eval time.
    venv = build_config_env(cfg, render_mode="rgb_array", seed_offset=10000)

    writer = None
    records = []
    try:
        model = PPO.load(model_path, env=venv, device=cfg.run.device)
        if video_path is not None:
            import imageio.v2 as imageio

            Path(video_path).resolve().parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=fps)
        for ep in range(episodes):
            venv.seed(cfg.run.seed + 10_000 + ep)
            obs = venv.reset()
            done = False
            total_reward = 0.0
            max_x = 0
            flag = False
            steps = score = coins_collected = max_coverage = 0
            world = stage = 1
            cleared = set()
            previous_coins = int(venv.reset_infos[0].get("coins", 0))
            while not done:
                if writer is not None:
                    writer.append_data(_get_rgb_frame(venv))
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = venv.step(action)
                total_reward += float(rewards[0])
                steps += 1
                done = bool(dones[0])
                info = infos[0]
                if "x_pos" in info:
                    max_x = max(max_x, int(info["x_pos"]))
                stage_clears = info.get("stage_clears", [])
                flag = flag or bool(stage_clears) or bool(info.get("flag_get", False))
                cleared.update(tuple(stage) for stage in stage_clears)
                world, stage = max(
                    (world, stage), (int(info.get("world", 1)), int(info.get("stage", 1)))
                )
                score = max(score, int(info.get("score", 0)))
                coins = int(info.get("coins", previous_coins))
                delta = coins - previous_coins
                coins_collected += delta if delta >= 0 else delta + 100
                previous_coins = coins
                max_coverage = max(max_coverage, int(info.get("coverage_cells", 0)))
                if info.get("flag_get", False):
                    cleared.add((int(info.get("world", 1)), int(info.get("stage", 1))))
            records.append(
                {
                    "episode": ep + 1,
                    "seed": cfg.run.seed + 10_000 + ep,
                    "extrinsic_return": total_reward,
                    "agent_steps": steps,
                    "max_x_pos": max_x,
                    "any_stage_cleared": flag,
                    "stages_cleared": len(cleared),
                    "furthest_world": world,
                    "furthest_stage": stage,
                    "max_game_score": score,
                    "coins_collected": coins_collected,
                    "coverage_cells": max_coverage,
                }
            )
            print(
                f"Episode {ep + 1}/{episodes}: reward={total_reward:.1f} "
                f"max_x_pos={max_x} flag_get={flag}"
            )
    finally:
        if writer is not None:
            writer.close()
            print(f"Saved video to {video_path}")
        venv.close()

    returns = [r["extrinsic_return"] for r in records]
    result = {
        "model": str(Path(model_path).resolve()),
        "config": asdict(cfg),
        "model_timesteps": model.num_timesteps,
        "deterministic": deterministic,
        "episodes": records,
        "summary": {
            "mean_extrinsic_return": float(np.mean(returns)),
            "std_extrinsic_return": float(np.std(returns)),
            "any_stage_clear_rate": float(np.mean([r["any_stage_cleared"] for r in records])),
        },
    }
    output = (
        Path(metrics_path) if metrics_path else Path(model_path).with_suffix(".evaluation.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Mean extrinsic return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"Saved evaluation metrics to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO Mario checkpoint + record video.")
    parser.add_argument("--config", required=True, help="YAML config (must match training env).")
    parser.add_argument("--model", required=True, help="Path to a saved model .zip.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--video", default=None, help="Output video path (.mp4 or .gif). Omit to skip."
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Sample actions instead of argmax."
    )
    parser.add_argument("--metrics", help="Output JSON. defaults beside the model.")
    parser.add_argument(
        "--fps", type=float, default=None, help="Playback fps. defaults to 60/frame_skip."
    )
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        model_path=args.model,
        episodes=args.episodes,
        video_path=args.video,
        deterministic=not args.stochastic,
        fps=args.fps,
        metrics_path=args.metrics,
    )


if __name__ == "__main__":
    main()
