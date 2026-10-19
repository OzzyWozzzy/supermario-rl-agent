"""Record checkpoints and keep the highest scoring gameplay videos."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Optional

import imageio
import numpy as np
from stable_baselines3 import PPO

from .config import load_config
from .env import build_config_env

SCORE_DIGITS = 9  # zero-pad width so filenames sort lexically by score


def _checkpoint_step(path: str) -> int:
    m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _score_key(game_score: int) -> int:
    """Rank gameplay recordings by their maximum game score."""
    return int(game_score)


def _build_eval_venv(cfg):
    return build_config_env(cfg, render_mode="rgb_array", seed_offset=777)


def _play_episode(venv, model, deterministic: bool):
    """Play one episode. Returns (frames, reward, max_x, flag, max_score, world, stage)."""
    frames = []
    obs = venv.reset()
    done = False
    total_reward = 0.0
    max_x = 0
    max_score = 0
    flag = False
    world = stage = 1
    while not done:
        # .copy() is REQUIRED: the emulator returns one reused screen buffer, so
        # without a snapshot every appended frame aliases to the final image.
        frames.append(np.asarray(venv.get_images()[0]).copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        total_reward += float(rewards[0])
        done = bool(dones[0])
        info = infos[0]
        if "x_pos" in info:
            max_x = max(max_x, int(info["x_pos"]))
        if "score" in info:
            max_score = max(max_score, int(info["score"]))
        flag = flag or bool(info.get("flag_get", False))
        world = int(info.get("world", world))
        stage = int(info.get("stage", stage))
    return frames, total_reward, max_x, flag, max_score, world, stage


def _prune_to_top_k(rec_dir: str, k: int) -> None:
    """Keep only the K highest-scoring mp4s (filenames begin with zero-padded score)."""
    vids = glob.glob(os.path.join(rec_dir, "score*.mp4"))
    # Lexical sort works because score is fixed-width zero-padded at the front.
    vids.sort(reverse=True)
    for stale in vids[k:]:
        try:
            os.remove(stale)
        except OSError:
            pass


def autorecord(
    config_path: str,
    top_k: int,
    episodes: int,
    poll_seconds: float,
    deterministic: bool,
    fps: int,
    level: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> None:
    if top_k < 1 or episodes < 1 or poll_seconds <= 0 or fps <= 0:
        raise ValueError("top_k, episodes, poll_seconds and fps must be positive")
    cfg = load_config(config_path)
    if level:
        w, s = level.split("-")
        cfg.env.env_id = f"SuperMarioBros-{w}-{s}-v0"
        cfg.run.log_dir = f"runs/ppo_world{w}_{s}"
    log_dir = run_dir or cfg.run.log_dir
    rec_dir = os.path.join(log_dir, "recordings")
    os.makedirs(rec_dir, exist_ok=True)

    state_path = os.path.join(rec_dir, "_processed.json")
    processed = set()
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                processed = set(json.load(f))
        except Exception:
            processed = set()

    print(
        f"[autorecord] watching {log_dir}/checkpoints | top_k={top_k} "
        f"episodes/ckpt={episodes} mode={'deterministic' if deterministic else 'stochastic'}"
    )

    while True:
        ckpts = sorted(
            glob.glob(os.path.join(log_dir, "checkpoints", "*_steps.zip")),
            key=_checkpoint_step,
        )
        new = [c for c in ckpts if _checkpoint_step(c) not in processed]

        for ckpt in new:
            step = _checkpoint_step(ckpt)
            try:
                model = PPO.load(ckpt, device=cfg.run.device)
            except Exception as e:  # checkpoint may be mid-write. retry next poll
                print(f"[autorecord] skip {os.path.basename(ckpt)} (load failed: {e})")
                continue

            venv = _build_eval_venv(cfg)
            best_score = -1
            best_ws = "1-1"
            try:
                for ep in range(episodes):
                    frames, reward, max_x, flag, max_score, world, stage = _play_episode(
                        venv, model, deterministic
                    )
                    key = _score_key(max_score)
                    fname = (
                        f"score{key:0{SCORE_DIGITS}d}_step{step:08d}_ep{ep}"
                        f"_w{world}-{stage}_pts{max_score}_x{max_x}{'_FLAG' if flag else ''}.mp4"
                    )
                    out = os.path.join(rec_dir, fname)
                    imageio.mimwrite(out, frames, fps=fps, macro_block_size=None)
                    if max_score > best_score:
                        best_score, best_ws = max_score, f"{world}-{stage}"
                    _prune_to_top_k(rec_dir, top_k)
            finally:
                venv.close()

            processed.add(step)
            with open(state_path, "w") as f:
                json.dump(sorted(processed), f)
            kept = len(glob.glob(os.path.join(rec_dir, "score*.mp4")))
            print(
                f"[autorecord] step {step}: recorded {episodes} ep(s), best score={best_score} "
                f"(reached {best_ws}) | leaderboard holds {kept}/{top_k}"
            )

        time.sleep(poll_seconds)


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-record a top-K leaderboard of best runs.")
    p.add_argument("--config", required=True)
    p.add_argument("--top-k", type=int, default=50, help="Max recordings to keep (default 50).")
    p.add_argument("--episodes", type=int, default=3, help="Episodes recorded per new checkpoint.")
    p.add_argument("--poll", type=float, default=20.0, help="Seconds between checkpoint scans.")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Greedy policy (default: stochastic, varied runs).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Playback fps. 15 = 60/frame_skip = real-time NES speed.",
    )
    p.add_argument(
        "--level", default=None, help="Level override, e.g. '1-2' (matches train.py --level)."
    )
    p.add_argument("--run-dir", help="Run directory containing checkpoints/.")
    args = p.parse_args()

    autorecord(
        config_path=args.config,
        top_k=args.top_k,
        episodes=args.episodes,
        poll_seconds=args.poll,
        deterministic=args.deterministic,
        fps=args.fps,
        level=args.level,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
