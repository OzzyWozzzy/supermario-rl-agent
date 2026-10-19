"""RND curiosity rewards and checkpoint state."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper


class _RunningMeanStd:
    """Welford-style running mean/variance (per the RND paper's normalisation)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


class _RNDNetwork(nn.Module):
    """Encode image stacks into features for the RND target or predictor."""

    def __init__(self, in_channels: int, feature_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 84x84 input through the conv stack -> 7x7x64 = 3136 features.
        self.head = nn.Linear(3136, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(x))


class RNDRewardWrapper(VecEnvWrapper):
    """Augment extrinsic reward with RND novelty. Wrap on top of VecFrameStack."""

    def __init__(
        self,
        venv,
        feature_dim: int = 512,
        intrinsic_coef: float = 0.5,
        lr: float = 1e-4,
        update_proportion: float = 1.0,
        device: str = "auto",
    ):
        super().__init__(venv)
        self.intrinsic_coef = intrinsic_coef
        self.update_proportion = update_proportion

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # frame_stack is the channel count (obs is channels-last from VecFrameStack).
        in_channels = self.observation_space.shape[-1]
        self.target = _RNDNetwork(in_channels, feature_dim).to(self.device)
        self.predictor = _RNDNetwork(in_channels, feature_dim).to(self.device)
        for p in self.target.parameters():  # target is frozen
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        self.obs_rms = _RunningMeanStd(shape=(in_channels, 84, 84))
        self.reward_rms = _RunningMeanStd(shape=())

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        reward_obs = obs.copy()
        for i, done in enumerate(dones):
            if done:
                reward_obs[i] = infos[i]["terminal_observation"]
        intrinsic = self._compute_and_train(reward_obs)
        # Expose intrinsic reward for logging without polluting the observation.
        for i, info in enumerate(infos):
            info["intrinsic_reward"] = float(intrinsic[i])
        rewards = rewards + self.intrinsic_coef * intrinsic
        return obs, rewards, dones, infos

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        """(N, H, W, C) uint8 -> (N, C, H, W) float, observation-normalised & clipped."""
        x = obs.astype(np.float32).transpose(0, 3, 1, 2)
        self.obs_rms.update(x)
        x = (x - self.obs_rms.mean) / (self.obs_rms.std + 1e-8)
        return np.clip(x, -5.0, 5.0).astype(np.float32)

    def _compute_and_train(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(self._normalise_obs(obs)).to(self.device)

        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)

        # Per-sample novelty = mean squared predictor error over feature dims.
        per_sample_err = (pred_feat - target_feat).pow(2).mean(dim=1)

        # Train predictor on a (sub)sample of the batch.
        loss = per_sample_err
        if self.update_proportion < 1.0:
            mask = (torch.rand(len(loss), device=self.device) < self.update_proportion).float()
            denom = torch.clamp(mask.sum(), min=1.0)
            loss = (loss * mask).sum() / denom
        else:
            loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalise the intrinsic reward by its running std (RND paper).
        intrinsic = per_sample_err.detach().cpu().numpy().astype(np.float64)
        self.reward_rms.update(intrinsic)
        intrinsic = intrinsic / (self.reward_rms.std + 1e-8)
        return intrinsic.astype(np.float32)


def maybe_wrap_intrinsic(venv, intrinsic_cfg, device: str = "auto"):
    """Apply configured RND curiosity during training."""
    if not intrinsic_cfg.use_intrinsic:
        return venv
    method = intrinsic_cfg.method.lower()
    if method == "rnd":
        return RNDRewardWrapper(
            venv,
            feature_dim=intrinsic_cfg.feature_dim,
            intrinsic_coef=intrinsic_cfg.intrinsic_coef,
            lr=intrinsic_cfg.lr,
            update_proportion=intrinsic_cfg.update_proportion,
            device=device,
        )
    raise NotImplementedError(
        f"intrinsic.method='{method}' is not implemented. Only 'rnd' is available. "
        "Choose method: rnd."
    )


def find_rnd(venv):
    """Locate curiosity inside the transpose wrapper without coupling PPO to it."""
    while venv is not None:
        if isinstance(venv, RNDRewardWrapper):
            return venv
        venv = getattr(venv, "venv", None)
    return None


def save_rnd(venv, model_path):
    """Save RND beside its corresponding PPO ZIP (trusted local checkpoints)."""
    from pathlib import Path

    rnd = find_rnd(venv)
    if rnd is None:
        return
    state = {
        "target": rnd.target.state_dict(),
        "predictor": rnd.predictor.state_dict(),
        "optimizer": rnd.optimizer.state_dict(),
        "intrinsic_coef": rnd.intrinsic_coef,
        "update_proportion": rnd.update_proportion,
        "obs_rms": vars(rnd.obs_rms),
        "reward_rms": vars(rnd.reward_rms),
    }
    path = Path(model_path).with_suffix(".rnd.pt")
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_rnd(venv, model_path):
    """Restore RND state from the checkpoint sidecar."""
    from pathlib import Path

    rnd = find_rnd(venv)
    if rnd is None:
        return
    path = Path(model_path).with_suffix(".rnd.pt")
    if not path.exists():
        raise FileNotFoundError(f"Missing RND state: {path}. Use --init-from for a fresh run.")
    state = torch.load(path, map_location=rnd.device, weights_only=False)
    rnd.target.load_state_dict(state["target"])
    rnd.predictor.load_state_dict(state["predictor"])
    rnd.optimizer.load_state_dict(state["optimizer"])
    rnd.intrinsic_coef = state["intrinsic_coef"]
    rnd.update_proportion = state["update_proportion"]
    for name in ("obs_rms", "reward_rms"):
        for key, value in state[name].items():
            setattr(getattr(rnd, name), key, value)
