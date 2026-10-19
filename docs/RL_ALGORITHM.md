# RL implementation

The agent uses Stable-Baselines3 PPO with a CNN policy. Optional rewards encourage
collection, spatial coverage, and visual curiosity.

## Observations and actions

`build_config_env` shares the environment setup across training and evaluation:

1. `JoypadSpace` maps discrete actions to NES buttons.
2. `SkipFrame` repeats actions, sums rewards, and pools the last two available frames.
3. Optional wrappers scale the base reward and add collection or coverage bonuses.
4. `GrayScaleResize` produces an 84×84 grayscale uint8 image.
5. Vector workers collect episodes and stack four frames for temporal context.
6. Training optionally adds RND before transposing observations to 4×84×84.

The policy sees images only. Collection and coverage rewards use game-state fields.
`SIMPLE_MOVEMENT` has seven actions. `COMPLEX_MOVEMENT` has twelve.

## PPO

NatureCNN uses three convolutional layers and a 512-unit feature layer. The heads
predict a categorical action distribution and a scalar value. SB3 scales image
values to [0, 1].

The full-game configuration collects 512 transitions from each of twelve workers
per rollout. It uses GAE with gamma 0.9 and lambda 0.95, then trains minibatches of
256 for four epochs.

For probability ratio `q = pi_new / pi_old` and advantage `A`, the policy objective
is `min(q*A, clip(q, 0.8, 1.2)*A)`. Value error has weight 0.5 and entropy has weight
0.03 in the full-game configuration. Gradient norm is capped at 0.5. The learning
rate starts at 0.00025 and decays linearly.

Budgets count agent transitions across workers. Frame skip four means each decision
covers up to four emulator frames. PPO completes whole rollouts, so ordinary
training may exceed its requested budget. The comparison runner rounds both methods
to the same whole-rollout budget before training.

## Rewards

The baseline uses the emulator's native shaped reward, including progress, time,
score, coins, power-ups, stage completion, and death. The full-game
configuration uses:

```text
0.4 * summed base reward
+ 0.01 * positive score gain
+ 75.0 * coins gained
+ 20.0 * positive power-up status gain
+ 1.0 * new coverage cell
+ 0.3 * normalized RND error
```

Coin differences account for the 99-to-0 counter wrap. Power-up levels are small,
tall, and fireball. Coverage cells are `(world, stage, x//16, y//16)` and reset each
episode. This is an episodic coverage bonus, not the full Go-Explore algorithm.
Base reward scaling happens before the bonuses. Score and coin rewards overlap.
The legacy `humanlike` flag enables `CollectionReward`.

## RND

A frozen random target and a trainable predictor encode stacked images into 512
features. Each vector step normalizes observations using running statistics,
clips them to [-5, 5], and computes mean squared feature error.

Adam updates the predictor online. The pre-update error is divided by its running
standard deviation and added to the PPO reward. PPO uses one value head for the
combined reward. This differs from the original RND implementation's rollout-based
predictor updates and separate intrinsic value head.

Episode-end curiosity uses terminal observations. Episode return logging and
evaluation exclude curiosity but retain configured collection and coverage bonuses.

## Runs and checkpoints

Fresh and smoke runs use separate directories. Each invocation records config,
seed, versions, source checkpoint, and actual PPO settings. Resume retains saved PPO
settings and optimizer state. Warm starts use policy weights with a fresh optimizer.
Partial transfer copies matching tensors, which does not validate action meanings.

RND checkpoints need both the PPO ZIP and its `.rnd.pt` state file. Resume restores
networks, optimizer, and running statistics. It does not restore emulator or global
RNG state for an identical continuation. Old models can warm-start a new run.

The training callback selects the best model by mean extrinsic return. Evaluation
histories are kept across sessions. Standalone evaluation seeds each episode and
saves JSON. Stage-clear flags are collected across repeated frames. Full-game metrics count
observed clears, not whole-game success.
Horizontal position is stage-local.

## Comparison protocol

`src.compare` builds PPO and PPO + RND settings from the same baseline config.
Only the curiosity flag differs within a seed. Runs start fresh, use matched
budgets, and alternate execution order. Final policies are evaluated with the same
episode seeds and deterministic actions.

`src.analyze` exports per-seed metrics and curves. Aggregates weight training seeds
equally. The figure shows individual runs and their mean without treating episode
variation as uncertainty across training runs.

## Historical checkpoint

The inspected full-game checkpoint had 2,899,884 transitions and entropy coefficient
0.01, while the current YAML specifies 0.03. Historical artifacts lack complete
config provenance and are excluded from the controlled comparison.
