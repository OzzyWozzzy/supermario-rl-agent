"""Checks for experimental controls and report aggregation."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from src.analyze import aggregate_runs, load_runs
from src.compare import comparison_configs


class ComparisonTests(unittest.TestCase):
    def test_configs_change_only_curiosity(self):
        configs = comparison_configs(
            "configs/ppo_world1_1.yaml", seed=7, timesteps=100_000, pilot=True
        )
        baseline, rnd = (asdict(configs[name]) for name in ("ppo", "ppo_rnd"))
        self.assertFalse(baseline["intrinsic"]["use_intrinsic"])
        self.assertTrue(rnd["intrinsic"]["use_intrinsic"])
        rnd["intrinsic"]["use_intrinsic"] = False
        self.assertEqual(baseline, rnd)
        self.assertEqual(baseline["ppo"]["total_timesteps"], 2048)
        self.assertEqual(baseline["run"]["seed"], 7)

    def test_report_weights_training_seeds_equally(self):
        runs = [
            {
                "summary": {
                    "method": method,
                    "extrinsic_return": value,
                    "any_stage_cleared": 0.5,
                    "max_game_score": 100,
                    "max_x_pos": 200,
                    "agent_steps": 30,
                    "training_seconds": 10,
                }
            }
            for method in ("ppo", "ppo_rnd")
            for value in (10, 30)
        ]
        for aggregate in aggregate_runs(runs):
            self.assertEqual(aggregate["training_seeds"], 2)
            self.assertEqual(aggregate["mean_extrinsic_return"], 20)
            self.assertAlmostEqual(aggregate["std_extrinsic_return"], 2**0.5 * 10)

    def test_report_rejects_mismatched_final_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "ppo_seed0"
            run.mkdir()
            (run / "evaluation.json").write_text(
                json.dumps(
                    {
                        "config": asdict(
                            comparison_configs("configs/ppo_world1_1.yaml", 0, 2048, pilot=True)[
                                "ppo"
                            ]
                        ),
                        "deterministic": True,
                        "model_timesteps": 1024,
                        "episodes": [],
                    }
                )
            )
            manifest = root / "comparison.json"
            manifest.write_text(
                json.dumps(
                    {
                        "evaluation_episodes": 1,
                        "runs": [
                            {
                                "status": "complete",
                                "directory": run.name,
                                "budget": 2048,
                                "method": "ppo",
                                "seed": 0,
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "budget"):
                load_runs(manifest)


if __name__ == "__main__":
    unittest.main()
