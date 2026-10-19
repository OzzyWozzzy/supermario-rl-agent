# PPO and RND pilot

This pilot checks the workflow. Its training budget is too small to compare learning quality.

| Method | Seeds | Completion | Mean return | Training time |
|---|---:|---:|---:|---:|
| PPO | 2 | 0% | 240.5 | 9.8s |
| PPO + RND | 2 | 0% | 231.0 | 12.6s |

![Comparison](comparison.png)

Thin curves show individual training seeds. Thick curves show their mean.
Dots show completion rates for each final policy. Standard deviations in
metrics.json describe variation across training seeds, not confidence intervals.

See per_seed.csv for final metrics and training wall-clock time. Timing includes
training-time evaluation and checkpoint saving, but excludes final evaluation.
