"""Mario environments, pixel preprocessing, and reward wrappers."""

from __future__ import annotations

from typing import Callable, List, Optional

import gym_super_mario_bros
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)
from nes_py.wrappers import JoypadSpace

from .perception import FRAME_HEIGHT, FRAME_WIDTH, preprocess_frame

# Map config strings -> nes-py action lists.
ACTION_SETS = {
    "RIGHT_ONLY": RIGHT_ONLY,
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT,
}


class SkipFrame(gym.Wrapper):
    """Repeat the chosen action `skip` times and max-pool the last two frames."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip < 1:
            raise ValueError("skip must be positive")
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        frames = []
        stage_clears = set()
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # The emulator reuses its screen buffer. snapshot the last two frames.
            frames.append(obs.copy())
            frames = frames[-2:]
            total_reward += reward
            if info.get("flag_get", False):
                stage_clears.add((int(info.get("world", 1)), int(info.get("stage", 1))))
            if terminated or truncated:
                break
        info["stage_clears"] = sorted(stage_clears)
        max_frame = np.maximum.reduce(frames)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GrayScaleResize(gym.ObservationWrapper):
    """Expose each preprocessed frame as a channels-last uint8 image."""

    def __init__(self, env: gym.Env, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        frame = preprocess_frame(obs, width=self._width, height=self._height)
        return frame[:, :, None]  # (H, W) -> (H, W, 1)


class RewardScale(gym.RewardWrapper):
    """Scale and optionally clip the base game reward."""

    def __init__(self, env: gym.Env, scale: float = 1.0, clip: float | None = None):
        super().__init__(env)
        self._scale = scale
        self._clip = clip

    def reward(self, reward):
        r = reward * self._scale
        if self._clip is not None:
            r = float(np.clip(r, -self._clip, self._clip))
        return r


class CollectionReward(gym.Wrapper):
    """Add bonuses for score, coins, and power-ups using game info."""

    _STATUS = {"small": 0, "tall": 1, "fireball": 2}

    def __init__(
        self,
        env: gym.Env,
        score_coef: float = 0.01,
        coin_coef: float = 2.0,
        powerup_coef: float = 10.0,
    ):
        super().__init__(env)
        self._score_coef = score_coef
        self._coin_coef = coin_coef
        self._powerup_coef = powerup_coef
        self._score = 0
        self._coins = 0
        self._status = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._score = int(info.get("score", 0))
        self._coins = int(info.get("coins", 0))
        self._status = self._STATUS.get(info.get("status", "small"), 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        score = int(info.get("score", 0))
        coins = int(info.get("coins", 0))
        status = self._STATUS.get(info.get("status", "small"), 0)

        d_score = max(0, score - self._score)
        d_coins = coins - self._coins
        if d_coins < 0:  # coin counter wraps 99 -> 0 on the 1-up
            d_coins += 100
        d_status = max(0, status - self._status)

        bonus = (
            self._score_coef * d_score
            + self._coin_coef * max(0, d_coins)
            + self._powerup_coef * d_status
        )
        # Surface the shaping for logging/metrics, never into the observation.
        info["humanlike_bonus"] = float(bonus)

        self._score, self._coins, self._status = score, coins, status
        return obs, reward + bonus, terminated, truncated, info


class CoverageReward(gym.Wrapper):
    """Reward new spatial cells within each episode."""

    CELL = 16  # pixel bin size for the coverage grid

    def __init__(self, env: gym.Env, coverage_coef: float = 0.5):
        super().__init__(env)
        self._coef = coverage_coef
        self._visited = set()

    def _cell(self, info):
        return (
            int(info.get("world", 1)),
            int(info.get("stage", 1)),
            int(info.get("x_pos", 0)) // self.CELL,
            int(info.get("y_pos", 0)) // self.CELL,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited = {self._cell(info)}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cell = self._cell(info)
        if cell not in self._visited:
            self._visited.add(cell)
            reward += self._coef
            info["coverage_new_cell"] = 1
        info["coverage_cells"] = len(self._visited)
        return obs, reward, terminated, truncated, info


def make_single_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_set: str = "SIMPLE_MOVEMENT",
    skip: int = 4,
    reward_scale: float = 1.0,
    reward_clip: Optional[float] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    humanlike: bool = False,
    score_coef: float = 0.01,
    coin_coef: float = 2.0,
    powerup_coef: float = 10.0,
    coverage: bool = False,
    coverage_coef: float = 0.5,
) -> gym.Env:
    """Build one Mario environment with configured actions and reward wrappers."""
    if action_set not in ACTION_SETS:
        raise ValueError(f"Unknown action_set '{action_set}'. Choose from {list(ACTION_SETS)}.")

    env = gym_super_mario_bros.make(env_id, render_mode=render_mode)
    env = JoypadSpace(env, ACTION_SETS[action_set])
    env = SkipFrame(env, skip=skip)
    if reward_scale != 1.0 or reward_clip is not None:
        env = RewardScale(env, scale=reward_scale, clip=reward_clip)
    if humanlike:
        env = CollectionReward(
            env, score_coef=score_coef, coin_coef=coin_coef, powerup_coef=powerup_coef
        )
    if coverage:
        env = CoverageReward(env, coverage_coef=coverage_coef)
    env = GrayScaleResize(env)

    if seed is not None:
        # Gymnasium removed Env.seed(). RNG is seeded via reset(seed=...) (SB3 does
        # this) and action sampling via action_space.seed() for reproducible rollouts.
        env.action_space.seed(seed)
    return env


def make_env_thunk(
    rank: int,
    env_id: str,
    action_set: str,
    skip: int,
    reward_scale: float,
    reward_clip: Optional[float],
    seed: int,
    render_mode: Optional[str] = None,
    humanlike: bool = False,
    score_coef: float = 0.01,
    coin_coef: float = 2.0,
    powerup_coef: float = 10.0,
    coverage: bool = False,
    coverage_coef: float = 0.5,
) -> Callable[[], gym.Env]:
    """Return an environment factory with a distinct worker seed."""

    def _thunk() -> gym.Env:
        return make_single_env(
            env_id=env_id,
            action_set=action_set,
            skip=skip,
            reward_scale=reward_scale,
            reward_clip=reward_clip,
            seed=seed + rank,
            render_mode=render_mode,
            humanlike=humanlike,
            score_coef=score_coef,
            coin_coef=coin_coef,
            powerup_coef=powerup_coef,
            coverage=coverage,
            coverage_coef=coverage_coef,
        )

    return _thunk


def make_vec_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_set: str = "SIMPLE_MOVEMENT",
    n_envs: int = 8,
    skip: int = 4,
    frame_stack: int = 4,
    reward_scale: float = 1.0,
    reward_clip: Optional[float] = None,
    seed: int = 0,
    use_subproc: bool = True,
    render_mode: Optional[str] = None,
    humanlike: bool = False,
    score_coef: float = 0.01,
    coin_coef: float = 2.0,
    powerup_coef: float = 10.0,
    coverage: bool = False,
    coverage_coef: float = 0.5,
):
    """Build seeded workers with episode monitoring and channels-last frame stacks."""
    # Imported here (not at module top) so that `import env` works for syntax/
    # tooling even in an environment where SB3 is not installed.
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
        VecMonitor,
    )

    thunks: List[Callable[[], gym.Env]] = [
        make_env_thunk(
            rank=i,
            env_id=env_id,
            action_set=action_set,
            skip=skip,
            reward_scale=reward_scale,
            reward_clip=reward_clip,
            seed=seed,
            render_mode=render_mode,
            humanlike=humanlike,
            score_coef=score_coef,
            coin_coef=coin_coef,
            powerup_coef=powerup_coef,
            coverage=coverage,
            coverage_coef=coverage_coef,
        )
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        venv = SubprocVecEnv(thunks, start_method="spawn")
    else:
        venv = DummyVecEnv(thunks)

    # VecMonitor sits *below* frame-stacking and any intrinsic wrapper so that the
    # logged "rollout/ep_rew_mean" is the shaped extrinsic reward, excluding RND.
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=frame_stack, channels_order="last")
    venv.seed(seed)
    return venv


def build_config_env(cfg, *, training=False, render_mode=None, seed_offset=10_000):
    """Shared config-to-environment mapping. evaluation excludes curiosity only."""
    from stable_baselines3.common.vec_env import VecTransposeImage

    from .intrinsic import maybe_wrap_intrinsic

    venv = make_vec_env(
        env_id=cfg.env.env_id,
        action_set=cfg.env.action_set,
        n_envs=cfg.env.n_envs if training else 1,
        skip=cfg.env.frame_skip,
        frame_stack=cfg.env.frame_stack,
        reward_scale=cfg.env.reward_scale,
        reward_clip=cfg.env.reward_clip,
        seed=cfg.run.seed + (0 if training else seed_offset),
        use_subproc=cfg.env.use_subproc if training else False,
        render_mode=render_mode,
        humanlike=cfg.env.humanlike,
        score_coef=cfg.env.score_coef,
        coin_coef=cfg.env.coin_coef,
        powerup_coef=cfg.env.powerup_coef,
        coverage=cfg.env.coverage,
        coverage_coef=cfg.env.coverage_coef,
    )
    if training:
        venv = maybe_wrap_intrinsic(venv, cfg.intrinsic, device=cfg.run.device)
    return VecTransposeImage(venv)
