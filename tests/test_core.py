"""Fast tests for custom logic, using deterministic environments rather than ROMs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.config import Config, load_config, validate_config
from src.env import CollectionReward, CoverageReward, SkipFrame
from src.intrinsic import RNDRewardWrapper, load_rnd, save_rnd
from src.runs import prepare_run


class PixelEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, end=3, infos=None):
        self.end = end
        self.infos = infos or [{"score": 0, "coins": 0, "status": "small", "x_pos": 0}]
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.full((84, 84, 1), 1, dtype=np.uint8), self.infos[0].copy()

    def step(self, action):
        self.t += 1
        info = self.infos[min(self.t, len(self.infos) - 1)].copy()
        return (
            np.full((84, 84, 1), 10 + self.t, dtype=np.uint8),
            1.0,
            self.t >= self.end,
            False,
            info,
        )


def rnd_env():
    env = VecFrameStack(DummyVecEnv([lambda: PixelEnv(end=1)]), n_stack=4, channels_order="last")
    return RNDRewardWrapper(env, feature_dim=8, device="cpu")


class CoreTests(unittest.TestCase):
    def test_skip_early_end_has_no_stale_frames(self):
        base = PixelEnv(end=4)
        env = SkipFrame(base, skip=4)
        env.reset()
        env.step(0)
        env.reset()
        base.end = 1
        obs, reward, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(reward, 1)
        self.assertTrue(np.all(obs == 11))

    def test_skip_keeps_intermediate_stage_clear(self):
        env = SkipFrame(
            PixelEnv(
                infos=[
                    {"flag_get": False},
                    {"flag_get": True, "world": 1, "stage": 1},
                    {"flag_get": False, "world": 1, "stage": 2},
                ]
            ),
            skip=3,
        )
        env.reset()
        info = env.step(0)[4]
        self.assertFalse(info["flag_get"])
        self.assertEqual(info["stage_clears"], [(1, 1)])

    def test_callback_retains_stage_clear_until_rollout_logging(self):
        from types import SimpleNamespace

        from src.callbacks import MarioMetricsCallback

        logged = {}
        callback = MarioMetricsCallback()
        callback.init_callback(
            SimpleNamespace(
                get_env=lambda: SimpleNamespace(num_envs=1),
                logger=SimpleNamespace(record=lambda key, value: logged.__setitem__(key, value)),
            )
        )
        for info, done in [({"stage_clears": [(1, 1)]}, False), ({}, True), ({}, False)]:
            callback.locals = {"infos": [info], "dones": [done]}
            callback._on_step()
        callback._on_rollout_end()
        self.assertEqual(logged["rollout/max_stages_cleared"], 1)
        self.assertEqual(logged["rollout/flag_get_rate"], 1)

    def test_coin_wrap_and_powerup_bonus(self):
        base = PixelEnv(
            infos=[
                {"score": 100, "coins": 99, "status": "small"},
                {"score": 300, "coins": 0, "status": "tall"},
            ]
        )
        env = CollectionReward(base, score_coef=0.01, coin_coef=2, powerup_coef=10)
        env.reset()
        self.assertEqual(env.step(0)[1], 15)  # base 1 + score 2 + coin 2 + powerup 10

    def test_coverage_counts_cells_once_and_resets(self):
        env = CoverageReward(
            PixelEnv(infos=[{"x_pos": 0}, {"x_pos": 16}, {"x_pos": 17}]), coverage_coef=2
        )
        env.reset()
        self.assertEqual(env.step(0)[1], 3)
        self.assertEqual(env.step(0)[1], 1)
        env.reset()
        self.assertEqual(env.step(0)[1], 3)

    def test_config_rejects_typos_and_invalid_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.yaml"
            for content in (
                "ppoo: {}",
                "[]",
                "ppo: {learning_rate: null}",
                "env: {frame_skip: 0}",
                "env: {use_subproc: 'yes'}",
                "ppo: {gamma: 1.1}",
                "intrinsic: {method: icm}",
                "ppo: {learning_rate: .nan}",
            ):
                p.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_config(str(p))
        for p in Path("configs").glob("*.yaml"):
            validate_config(load_config(str(p)))

    def test_run_isolation_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.run.log_dir = d
            path = prepare_run(cfg, smoke=True)
            self.assertEqual(path.parent.name, "smoke")
            (path / "existing").write_text("data")
            with self.assertRaises(FileExistsError):
                prepare_run(cfg, run_dir=str(path))

    def test_resume_rejects_reward_changes(self):
        from dataclasses import asdict

        import yaml

        from src.runs import validate_resume

        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            path = Path(d) / "final_model.zip"
            (Path(d) / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
            validate_resume(cfg, path)
            cfg.env.coin_coef = 999
            with self.assertRaises(ValueError):
                validate_resume(cfg, path)

    def test_rnd_uses_terminal_observation(self):
        env = rnd_env()
        try:
            env.reset()
            with patch.object(
                env, "_compute_and_train", return_value=np.array([0.0], dtype=np.float32)
            ) as fn:
                obs, _, dones, infos = env.step([0])
            self.assertTrue(dones[0])
            np.testing.assert_array_equal(fn.call_args.args[0][0], infos[0]["terminal_observation"])
            self.assertFalse(np.array_equal(obs[0], fn.call_args.args[0][0]))
        finally:
            env.close()

    def test_rnd_checkpoint_restores_prediction_and_optimizer(self):
        a, b = rnd_env(), rnd_env()
        try:
            a.reset()
            a.step([0])
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "model.zip"
                save_rnd(a, path)
                load_rnd(b, path)
            for name in ("target", "predictor"):
                for left, right in zip(
                    getattr(a, name).parameters(), getattr(b, name).parameters()
                ):
                    torch.testing.assert_close(left, right)
            np.testing.assert_array_equal(a.obs_rms.mean, b.obs_rms.mean)
            self.assertEqual(a.reward_rms.count, b.reward_rms.count)
            self.assertEqual(
                a.optimizer.state_dict()["param_groups"], b.optimizer.state_dict()["param_groups"]
            )
            for key, state in a.optimizer.state_dict()["state"].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, b.optimizer.state_dict()["state"][key][field])
        finally:
            a.close()
            b.close()


class ModelLifecycleTests(unittest.TestCase):
    def test_rnd_does_not_change_initial_policy_weights(self):
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import VecTransposeImage

        from src.train import build_model

        cfg = Config()
        cfg.run.device = "cpu"
        models, envs = [], []
        with tempfile.TemporaryDirectory() as d:
            cfg.run.log_dir = d
            try:
                for curiosity in (False, True):
                    set_random_seed(cfg.run.seed)
                    env = VecFrameStack(DummyVecEnv([PixelEnv]), 4, channels_order="last")
                    if curiosity:
                        env = RNDRewardWrapper(env, feature_dim=8, device="cpu")
                    env = VecTransposeImage(env)
                    envs.append(env)
                    models.append(build_model(cfg, env))
                for a, b in zip(models[0].policy.parameters(), models[1].policy.parameters()):
                    torch.testing.assert_close(a, b)
            finally:
                for env in envs:
                    env.close()

    def test_warm_start_is_fresh_and_resume_keeps_optimizer(self):
        from stable_baselines3.common.vec_env import VecTransposeImage

        from src.train import build_model, initialize_model

        cfg = Config()
        cfg.env.n_envs = 1
        cfg.ppo.n_steps = cfg.ppo.batch_size = 4
        cfg.ppo.n_epochs = 1
        cfg.run.device = "cpu"
        with tempfile.TemporaryDirectory() as d:
            cfg.run.log_dir = d
            env = VecTransposeImage(
                VecFrameStack(DummyVecEnv([PixelEnv]), 4, channels_order="last")
            )
            try:
                source = build_model(cfg, env)
                source.learn(4)
                path = str(Path(d) / "source.zip")
                source.save(path)
                cfg.ppo.ent_coef = 0.07
                warm = initialize_model(cfg, env, init_from=path)
                self.assertEqual(warm.num_timesteps, 0)
                self.assertEqual(warm.ent_coef, 0.07)
                self.assertFalse(warm.policy.optimizer.state)
                for a, b in zip(source.policy.parameters(), warm.policy.parameters()):
                    torch.testing.assert_close(a, b)
                resumed = initialize_model(cfg, env, resume_path=path)
                self.assertEqual(resumed.num_timesteps, 4)
                self.assertEqual(resumed.ent_coef, source.ent_coef)
                self.assertTrue(resumed.policy.optimizer.state)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
