"""Training metrics and paired PPO and RND checkpoints."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from .intrinsic import save_rnd


class MarioMetricsCallback(BaseCallback):
    """Log progress, stage clears, coverage, and curiosity each rollout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._max_x_pos = 0
        self._episodes = 0
        self._flags = 0
        self._intrinsic_sum = 0.0
        self._intrinsic_n = 0
        self._max_score = 0
        self._max_coverage = 0
        self._max_stages_cleared = 0

    def _on_step(self) -> bool:
        if not hasattr(self, "_episode_flags"):
            self._episode_flags = [False] * self.training_env.num_envs
            self._cleared_stages = [set() for _ in self._episode_flags]
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            # x_pos: horizontal position. used for an exploration metric only.
            if "x_pos" in info:
                self._max_x_pos = max(self._max_x_pos, int(info["x_pos"]))
            if "score" in info:
                self._max_score = max(self._max_score, int(info["score"]))
            if "coverage_cells" in info:
                self._max_coverage = max(self._max_coverage, int(info["coverage_cells"]))
            if "intrinsic_reward" in info:
                self._intrinsic_sum += float(info["intrinsic_reward"])
                self._intrinsic_n += 1
            clears = {tuple(stage) for stage in info.get("stage_clears", [])}
            if info.get("flag_get", False):
                clears.add((info.get("world"), info.get("stage")))
            if clears:
                self._episode_flags[i] = True
                self._cleared_stages[i].update(clears)
            self._max_stages_cleared = max(self._max_stages_cleared, len(self._cleared_stages[i]))
            # Rate means episodes with at least one clear, including intermediate stages.
            if i < len(dones) and dones[i]:
                self._episodes += 1
                if self._episode_flags[i]:
                    self._flags += 1
                self._episode_flags[i] = False
                self._cleared_stages[i].clear()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/max_x_pos", self._max_x_pos)
        self.logger.record("rollout/max_score", self._max_score)
        self.logger.record("rollout/max_coverage_cells", self._max_coverage)
        self.logger.record("rollout/max_stages_cleared", self._max_stages_cleared)
        if self._episodes > 0:
            self.logger.record("rollout/flag_get_rate", self._flags / self._episodes)
        if self._intrinsic_n > 0:
            self.logger.record("rollout/intrinsic_reward", self._intrinsic_sum / self._intrinsic_n)
        # Reset window accumulators each rollout for a per-window view.
        self._max_x_pos = 0
        self._episodes = 0
        self._flags = 0
        self._intrinsic_sum = 0.0
        self._intrinsic_n = 0
        self._max_score = 0
        self._max_coverage = 0
        self._max_stages_cleared = 0


class RNDCheckpointCallback(CheckpointCallback):
    """Pair each periodic PPO checkpoint with curiosity state."""

    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            save_rnd(self.training_env, self._checkpoint_path(extension="zip"))
        return result


class SaveBestRNDCallback(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def _on_step(self):
        from pathlib import Path

        save_rnd(self.training_env, Path(self.directory) / "best_model.zip")
        return True
