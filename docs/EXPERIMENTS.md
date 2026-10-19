# PPO and RND comparison

Question: does RND improve stage 1-1 learning under the same base reward and budget?

Both methods use seven actions, identical preprocessing and PPO settings, and fresh
models. RND is the only change within a training seed. Collection and coverage
rewards are disabled. Evaluation uses final policies rather than selecting a best
checkpoint after seeing the results.

## Pilot

Install the plotting dependency and run a short check:

```bash
uv pip install -e '.[analysis]'
python -m src.compare --pilot --seeds 0 1 --episodes 3 \
  --report-dir results/my_pilot
```

The pilot uses 2,048 transitions per run, two workers, and small PPO rollouts. It
checks training, checkpointing, evaluation, aggregation, and plotting. Its budget
is too small to support a conclusion about which method learns better.

## Main experiment

```bash
python -m src.compare --seeds 0 1 2 --timesteps 500000 --n-envs 4 \
  --episodes 10 --output runs/comparisons/stage1_1 \
  --report-dir results/stage1_1
```

Directories must be new. The runner rounds the requested budget to a whole rollout
and records the actual budget. Each run has a config, model, logs, and evaluation
JSON. A manifest records progress and wall-clock training time. Runs execute
sequentially to avoid competing for hardware.

Use the pilot timing to estimate compute cost. A 500,000-transition experiment is
an initial learning comparison. Longer budgets may be needed to observe reliable
completion. Choose the main protocol before inspecting its results.

## Reports

The report contains per-seed CSV, portable metrics JSON, and PNG and SVG figures.
Learning curves come from training-time evaluation. Final metrics use a separate
set of episode evaluations. Each training seed has equal weight in the aggregate.

Report completion rate, base reward, game score, horizontal progress, and compute
cost. Thin lines show individual seeds and thick lines show means. Final completion
dots represent independent trained policies. Reported standard deviations describe
variation across training seeds and are not confidence intervals.

To rebuild a report:

```bash
python -m src.analyze --manifest runs/comparisons/stage1_1/comparison.json \
  --output results/stage1_1_rebuilt
```

Three training seeds provide limited evidence. Evaluation seeds also do not imply
that a deterministic emulator generates diverse level layouts. Claims should stay
specific to this level, budget, and evaluation protocol.
