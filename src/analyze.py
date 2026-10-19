"""Export comparison metrics and learning curves without model artifacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("extrinsic_return", "any_stage_cleared", "max_game_score", "max_x_pos", "agent_steps")


def load_runs(manifest_path):
    """Load completed runs and reject mixed budgets or evaluation protocols."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    runs = []
    reference = None
    for record in manifest["runs"]:
        if record["status"] != "complete":
            raise ValueError("All runs must be complete before building a report")
        directory = path.parent / record["directory"]
        evaluation = json.loads((directory / "evaluation.json").read_text())
        config = evaluation["config"]
        intrinsic = config["intrinsic"].copy()
        enabled = intrinsic.pop("use_intrinsic")
        if enabled != (record["method"] == "ppo_rnd"):
            raise ValueError("Curiosity flag differs from the method label")
        if config["run"]["seed"] != record["seed"]:
            raise ValueError("Training seed differs from the manifest")
        protocol = (config["env"], config["ppo"], intrinsic, evaluation["deterministic"])
        if reference is not None and protocol != reference:
            raise ValueError("Runs have different environment, PPO or evaluation settings")
        reference = protocol
        if evaluation["model_timesteps"] != record["budget"]:
            raise ValueError("Final checkpoint budget differs from the planned budget")
        if len(evaluation["episodes"]) != manifest["evaluation_episodes"]:
            raise ValueError("Evaluation episode count differs from the protocol")
        points = {}
        for history in sorted(directory.glob("eval/**/evaluations.npz")):
            with np.load(history) as data:
                for steps, returns in zip(data["timesteps"], data["results"]):
                    points[int(steps)] = float(np.mean(returns))
        row = {
            "method": record["method"],
            "seed": record["seed"],
            "transitions": evaluation["model_timesteps"],
            "training_seconds": record["training_seconds"],
        }
        row.update(
            {
                metric: float(np.mean([ep[metric] for ep in evaluation["episodes"]]))
                for metric in METRICS
            }
        )
        metadata_path = sorted(directory.glob("metadata_*.json"))[-1]
        metadata = json.loads(metadata_path.read_text())
        provenance = {
            key: metadata[key]
            for key in ("python", "platform", "device", "versions", "git_commit", "git_dirty")
        }
        runs.append(
            {
                "summary": row,
                "episodes": evaluation["episodes"],
                "curve": [
                    {"transitions": x, "extrinsic_return": points[x]} for x in sorted(points)
                ],
                "config": config,
                "provenance": provenance,
            }
        )
    if not runs:
        raise ValueError("Comparison contains no runs")
    methods = {run["summary"]["method"] for run in runs}
    if methods != {"ppo", "ppo_rnd"}:
        raise ValueError("Report requires both PPO and PPO + RND")
    seeds = [
        {run["summary"]["seed"] for run in runs if run["summary"]["method"] == method}
        for method in sorted(methods)
    ]
    if seeds[0] != seeds[1] or len(runs) != 2 * len(seeds[0]):
        raise ValueError("Each seed must have exactly one run for each method")
    return manifest, runs


def aggregate_runs(runs):
    """Average per-seed metrics with equal weight for each training run."""
    result = []
    for method in ("ppo", "ppo_rnd"):
        rows = [run["summary"] for run in runs if run["summary"]["method"] == method]
        row = {"method": method, "training_seeds": len(rows)}
        for metric in (*METRICS, "training_seconds"):
            values = [entry[metric] for entry in rows]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def plot_runs(runs, output, pilot=False):
    """Show individual seed curves and final completion rates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = {"ppo": "#2878B5", "ppo_rnd": "#D66A27"}
    labels = {"ppo": "PPO", "ppo_rnd": "PPO + RND"}
    for method in colors:
        selected = [run for run in runs if run["summary"]["method"] == method]
        for run in selected:
            curve = run["curve"]
            axes[0].plot(
                [point["transitions"] for point in curve],
                [point["extrinsic_return"] for point in curve],
                color=colors[method],
                alpha=0.5,
                linewidth=1.2,
            )
        grids = [[point["transitions"] for point in run["curve"]] for run in selected]
        if grids[0] and all(grid == grids[0] for grid in grids):
            mean = np.mean(
                [[point["extrinsic_return"] for point in run["curve"]] for run in selected], axis=0
            )
            axes[0].plot(grids[0], mean, color=colors[method], linewidth=2.5, label=labels[method])
        else:
            axes[0].plot([], [], color=colors[method], label=labels[method])
        index = list(colors).index(method)
        values = [run["summary"]["any_stage_cleared"] for run in selected]
        axes[1].bar(index, np.mean(values), color=colors[method], alpha=0.7, width=0.55)
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
        axes[1].scatter(index + np.asarray(offsets), values, color="black", s=24, zorder=3)
    axes[0].set(
        xlabel="Agent transitions", ylabel="Evaluation extrinsic return", title="Learning curves"
    )
    axes[0].legend(frameon=False)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=list(labels.values()),
        ylim=(-0.03, 1.05),
        ylabel="Completion rate",
        title="Final policy on stage 1-1",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Pipeline pilot, insufficient training for conclusions"
        if pilot
        else "PPO and RND comparison"
    )
    fig.savefig(Path(output) / "comparison.png", dpi=180)
    fig.savefig(Path(output) / "comparison.svg")
    svg = Path(output) / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def build_report(manifest_path, output):
    """Write portable per-seed data, aggregate metrics, and a figure."""
    manifest, runs = load_runs(manifest_path)
    aggregates = aggregate_runs(runs)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = [run["summary"] for run in runs]
    with (root / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for run in runs:
        run["config"]["run"].pop("log_dir")
    report = {
        "kind": manifest["kind"],
        "created_utc": manifest["created_utc"],
        "evaluation_episodes": manifest["evaluation_episodes"],
        "checkpoint_selection": "final",
        "aggregates": aggregates,
        "runs": runs,
    }
    (root / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_runs(runs, root, pilot=manifest["kind"] == "pilot")
    limitation = (
        "This pilot checks the workflow. Its training budget is too small to compare learning quality."
        if manifest["kind"] == "pilot"
        else "These results describe this task and budget. The seed sample limits broader conclusions."
    )
    table = "| Method | Seeds | Completion | Mean return | Training time |\n"
    table += "|---|---:|---:|---:|---:|\n"
    for row in aggregates:
        label = "PPO" if row["method"] == "ppo" else "PPO + RND"
        table += (
            f"| {label} | {row['training_seeds']} | {row['mean_any_stage_cleared']:.0%} | "
            f"{row['mean_extrinsic_return']:.1f} | {row['mean_training_seconds']:.1f}s |\n"
        )
    (root / "README.md").write_text(
        f"# PPO and RND {manifest['kind']}\n\n{limitation}\n\n"
        + table
        + "\n![Comparison](comparison.png)\n\n"
        + "Thin curves show individual training seeds. Thick curves show their mean.\n"
        "Dots show completion rates for each final policy. Standard deviations in\n"
        "metrics.json describe variation across training seeds, not confidence intervals.\n\n"
        "See per_seed.csv for final metrics and training wall-clock time. Timing includes\n"
        "training-time evaluation and checkpoint saving, but excludes final evaluation.\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Export a completed PPO and RND comparison.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New directory for report artifacts.")
    args = parser.parse_args()
    build_report(args.manifest, args.output)


if __name__ == "__main__":
    main()
