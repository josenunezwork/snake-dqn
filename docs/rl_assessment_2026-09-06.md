# RL assessment — 2026-09-06

## Verdict

Keep Apex as the incumbent. Raster/PQN remains plausible: a cell-exact
vectorized simulator, dual-scale spatial observations, masked discrete actions,
and lightweight self-play fit this game. The completed Mac screen does **not**
yet provide strong evidence that the current PQN trainer learns a competitive
policy, and it promotes no model. The next work should make PQN a faithful,
measurable experiment before replacing it with PPO, Rainbow, or search.

This report separates three questions that should not be conflated:

1. The reward ablation asks whether a small sustained-living-mass term changes
   a short, paired training screen.
2. Existing raster and live cross-stack runs check the evaluator and reveal a
   large current gap to scripted/Apex references.
3. The promotion gate is the only authority for replacing the incumbent. These
   short diagnostic screens do not supply the full planned promotion evidence.

The intended reader is choosing the next learning experiment on this Mac and
the conditions for a later GPU run. It assumes familiarity with discounted
returns and paired evaluation. Every reported result comes from completed local
artifacts under [`runs/rl_assessment_20260906`](../runs/rl_assessment_20260906/);
they are review evidence, not published model artifacts.

## Current objective and the first correctness issue

The repository retains a 61-feature feed-forward Apex DQN champion while the
candidate path uses a raster dueling network and synchronous Q(lambda), without
replay, a target network, or PER. The split is documented in
[the README](../README.md#redesign-in-progress), and the candidate configuration
and update path are in [`PQNConfig`](../src/training/pqn_trainer.py#L88) and
[`PQNTrainer`](../src/training/pqn_trainer.py#L350). This is a useful design
seam: a fast exact simulator makes fresh-rollout methods attractive, and the
six masked actions keep Q-learning and PPO manageable.

Reward v2 is distinct from the promotion metric. It computes:

```
r_t = gamma * Phi(s_(t+1)) - Phi(s_t) - 3 * death_t
      + 0.3 * sum(victim_lengths_t),  Phi(length) = length / 10.
```

[`compute_reward_v2`](../src/core/reward_events.py#L76) and
[`BatchSim._compute_rewards`](../src/simd_env/batch_sim.py#L1024) set the terminal
potential to zero. With the learner's discount, shaping telescopes:

```
sum_t gamma^t [gamma Phi(s_(t+1)) - Phi(s_t)]
  = -Phi(s_0) + gamma^T Phi(s_T).
```

On death, `Phi(s_T)=0`; shaping contributes the fixed start-state constant
`-Phi(s_0)`. Terminal-zero potential is therefore strict episodic PBRS. It
changes credit assignment and does **not** independently reward sustained mass
or impose a mass-at-death penalty. This is the policy-invariance result of
[Ng, Harada, and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
and the finite-episode treatment in [Grześ (2017)](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf).
The corrected code comment and redesign claim in this worktree supersede the
prior contrary documentation. Production reward arithmetic is unchanged.

Promotion instead uses the undiscounted mass integral—mean hero mass across the
whole horizon, with zero after death—implemented by
[`mass_integral`](../src/scripts/eval_stats.py#L107) and
[`tournament_eval.rollout`](../src/scripts/tournament_eval.py#L283). If sustained
mass is the product objective, the sparse base return does not directly encode
it. That motivates the ablation below; it does not establish that mass should
become the production reward.

## What was run and how it is reproducible

The coordinator, evaluation script, analysis script, raw result JSON, and plot
are retained locally as
[`run_training.py`](../runs/rl_assessment_20260906/run_training.py),
[`evaluate_reward_pairs.py`](../runs/rl_assessment_20260906/evaluate_reward_pairs.py),
[`analyze_results.py`](../runs/rl_assessment_20260906/analyze_results.py),
[`reward_pair_eval.json`](../runs/rl_assessment_20260906/reward_pair_eval.json),
[`analysis.json`](../runs/rl_assessment_20260906/analysis.json), and
[`results.png`](../runs/rl_assessment_20260906/results.png). Use
`/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python` when inspecting or
recreating work; the assessment worktree's temporary `venv` symlink has been
removed. Do not re-run the coordinator into these nonempty
directories: the diagnostic refuses to overwrite them.

There were six completed 100,000-target-step arms: control and mass treatment
for training seeds 101, 202, and 303. Each used CPU, two Torch threads, one
training process at a time, `E=16`, `S=6`, `T=16`, `max_frames=5000`, all hero
slots, and no frozen pool. Control used current reward v2. Treatment added
`0.003 * poststep_alive_logical_length`; it is zero on a death frame. Both arms
of each training seed had identical initial tensor hashes, same simulator seed,
optimizer, augmentation, and epsilon schedule; epsilon reached 0.02 at 60% of
the 100k target. The runner records this contract, source hashes, checkpoints,
and telemetry in each arm's `provenance.json`, `summary.json`, and
`history.jsonl`.

The six arms completed with 603,341 actual hero agent-steps in 744 seconds of
training wall time. A seventh run replayed `control_s101`: including it, the campaign used
703,971 agent-steps, 589,824 recorded SGD draws, 888.535 seconds of process
wall time, and a maximum observed RSS of 891 MiB. CPU was limited to two
numerical threads per arm; the coordinator ran one arm at a time, with a
combined four-thread numerical budget when evaluation ran alongside it. These are
resource facts, not a throughput benchmark for a future GPU configuration.

The reproduction is strong for this small screen: initial and final checkpoint
bytes/parameter hashes matched exactly, and all 85 scientific telemetry records
matched after excluding wall-time and profiling fields. It is an exact replay
of one seed, **not** a fourth independent training replicate.

One provenance repair matters. The first `control_s101` manifest had a path bug
and omitted imported modules. The supplemental
[`source_snapshot.zip`](../runs/rl_assessment_20260906/source_snapshot.zip) and
[`source_manifest.json`](../runs/rl_assessment_20260906/source_manifest.json)
provide the source-size/hash audit; later arm manifests are complete. The
analysis records this limitation rather than silently treating the first per-arm
manifest as complete.

## Completed reward screen

Each learned checkpoint and matching initial checkpoint ran on the SIMD engine
for 400 frames over eight paired world seeds (2000–2007), with five
`scripted:greedy_food` opponents and `configs/mechanics_v2.yaml`. The table
contains the mean mass integral averaged across the three **training-seed
means**. Its intervals use those three learned-policy units. The eight world
seeds are paired measurements within each learned policy and are not counted as
24 independent training runs.

| Checkpoint stage | Mean mass integral | Comparison (95% CI) |
| --- | ---: | --- |
| Initial | 2.176 | mass − initial = 0.521, [−1.398, 2.439] |
| Control | 0.878 | control − initial = −1.298, [−3.354, 0.759] |
| `+0.003` living mass | 2.697 | mass − control = 1.819, [−0.429, 4.066] |
| `scripted:random_safe` reference | 3.672 | separately evaluated over the eight world seeds |
| `scripted:greedy_food` reference | 20.632 | separately evaluated over the eight world seeds |

At `gamma=0.997`, the added coefficient gives one persistent unit of living
mass an approximate discounted scale of `0.003 / (1 - 0.997) = 1`. That makes
the treatment interpretable relative to the sparse `-3` death event, but does
not calibrate it against a product requirement. A stronger coefficient, a
different survival term, or a changed evaluation metric would be a different
experiment and needs its own paired control.

The treatment has a positive point estimate over control, but its three-seed
interval crosses zero. It also does not establish an improvement over the
initial networks. The appropriate conclusion is that the coefficient is a
candidate for a better-powered, held-out experiment—not that it wins, and not
that reward v2 is disproven. Pooling the 24 within-seed world outcomes as 24
independently trained policies would be pseudoreplication.

All six arms completed without a tripwire, non-finite scientific telemetry, or
a Q alarm; their final epsilon was 0.02. This is numerical/pipeline evidence
only. It cannot establish performance because the trained policies remain far
below the greedy-food reference in this short screen.

## Existing models and a cross-stack live check

The existing-raster SIMD comparison used 400 frames, eight world seeds
(1000–1007), and scripted plus mixed opponent mixes. Against a
`scripted:greedy_food` baseline of 33.50 on the scripted mix, the four available
PQN checkpoints scored 3.70 (`pqn_local/latest_pqn`), 4.11 (`champ_s0`), 3.66
(`champ_s11`), and 2.87 (`champ_s22`). Their paired intervals were entirely
negative in that screen. These checkpoints have incomplete recipe/provenance
records, so this shows the available artifacts are not competitive there; it
does not identify why they failed or rank PQN variants causally.

The live cross-stack screen exercised the new evaluator adapter for 200 frames,
four world seeds (3000–3003), and two separately reported mixes. The adapter
permits a raster checkpoint only as hero slot 0 and rejects raster opponents,
because roster-order dispatch would otherwise return the wrong Q row. It is a
necessary correctness boundary, not a full mixed-roster solution.

| Candidate | Scripted mix mean | Mixed mix mean |
| --- | ---: | ---: |
| Apex A5 baseline | 13.859 | 17.186 |
| `scripted:greedy_food` | 17.521 | 19.189 |
| `champ_s0` | 3.429 | 4.265 |
| Seed-101 control | 1.248 | 1.248 |
| Seed-101 mass treatment | 1.523 | 2.280 |

The measured PQN rows are lower than A5 in both mixes, supporting retention of
the incumbent under these short conditions. Pod 0's paired interval versus A5
includes zero in the scripted mix and is negative in the mixed mix. Greedy-food's
apparent advantage over A5 is inconclusive in both mixes. Four short worlds are
insufficient for a broad algorithm ranking or promotion; the dependent mixes
must not be pooled as independent evidence. Both stacks used the same mechanics-v2
arena, although the Apex incumbent was trained under older mechanics. No model
was promoted.

## Why a method switch is premature

The original large PQN-style layout illustrates the current sampler issue. At
`E=128`, `S=8`, `T=32`, slot 0 forced hero, and the other seven slots hero with
probability 0.8, the expected pre-death hero rollout has
`128 * 32 * (1 + 7 * 0.8) = 27,034` rows. The current four independent
256-example minibatches draw 1,024 rows: 3.8% of that nominal set before
deaths. Draws can overlap because [`_sgd`](../src/training/pqn_trainer.py#L596)
creates a fresh permutation for each minibatch. This is neither four full
epochs nor a faithful implementation of the full-rollout shuffled epochs in the
[PQN paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c23f3852601f6dd7f0b39223d031806f-Paper-Conference.pdf)
and its [author implementation](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py#L309-L373).
The ablation is still fair because its sampler is shared across arms; it is not
strong evidence against PQN itself.

The default 50M-step epsilon decay is unsuitable for short screens unless
rescaled; otherwise the agent remains nearly random. This experiment did
rescale it, so its evidence applies only to the stated 100k recipe. Batch-wide
reset remains another utilization risk: environments that hit a population
floor are excluded until every environment does because reset is batch-wide
([`_episode_over`](../src/training/pqn_trainer.py#L325)). `kills_per_ep` remains
a raw all-slot/all-rollout total, not hero kills per completed episode
([`_rollout`](../src/training/pqn_trainer.py#L411)). Neither per-environment
reset nor corrected utilization/kill telemetry is claimed as implemented here.

The rollout-cap fix is implemented: when the cap does not divide the rollout,
the trainer collects only the remaining frames. The regression test covers
`max_frames=3` with `rollout_len=4`, preventing an episode-progress value above
one. It does not resolve the semantic time-horizon question. The target still
bootstraps at `max_frames` and population-floor cutoffs. If those are true task
terminals rather than artificial sampling cutoffs, that treatment is not exact
alignment with the fixed-horizon evaluation objective; the project must clarify
which meaning the cutoffs have before making an objective-level claim.

The current raster network also differs from the published PQN recipe: it
normalizes selected dense representations but not every convolutional/hidden
stage, and its dueling branches add unnormalized hidden layers before output.
The paper's LayerNorm stability argument therefore does not automatically cover
this architecture. That is a testable implementation difference, not proof it
is unstable. Keep dueling for the Q-value UI if product-useful, but add a direct
normalized six-output control to the research plan.

| Candidate direction | What primary evidence supports | Decision now |
| --- | --- | --- |
| Corrected PQN | [PQN](https://proceedings.iclr.cc/paper_files/paper/2025/file/c23f3852601f6dd7f0b39223d031806f-Paper-Conference.pdf) supports fresh-rollout Q(lambda) with full-buffer reuse; this project supplies a fast simulator and Q-value UI. | First choice after sampling, utilization, and normalization controls. |
| Masked PPO | [PPO](https://arxiv.org/abs/1707.06347) supports clipped multi-epoch optimization on fresh rollouts. | Pre-specified fallback if faithful PQN stays unstable or fails a healthy-data milestone; not implemented here. |
| Rainbow/QR-DQN/IQN | [Rainbow](https://arxiv.org/abs/1710.02298), [QR-DQN](https://arxiv.org/abs/1710.10044), and [IQN](https://arxiv.org/abs/1806.06923) support distributional/replay-based controls. | Later isolated control; restore replay/target complexity only after a fair fresh-rollout test. |
| Shallow exact-sim search | The deterministic simulator can reject immediate fatal choices without learning a second world model. | Serving adjunct only after a learned policy passes a clean gate; evaluate policy plus search together. |

## Next decision gate

Before a longer Mac or GPU run, instrument eligible hero rows, unique sampled
indices, total SGD draws, valid-after-floor fraction, reset skew, and hero-only
kill rate. Compare the current sampler with one complete shuffled rollout epoch,
including its final partial minibatch; if it helps, compare one versus two
epochs. Hold reward, seed set, network, epsilon schedule, and evaluator fixed
for that sampler experiment. Then isolate normalization/head design and only
then vary epsilon duration or introduce a pool. These are recommendations; they
were not implemented in this assessment.

If corrected PQN has healthy data use but is unstable or cannot clear a
pre-specified scripted-anchor milestone, build masked PPO as the fallback. Do
not add PPO, distributional replay, reward tuning, and self-play changes to one
run: that would erase the causal interpretation the six-arm design preserves.

For promotion, choose one final contract, use fresh held-out training and world
seeds, run the full 3,000-or-longer frame gate with frozen, scripted, and mixed
opponent mixes, and use
[`promotion_decision`](../src/scripts/tournament_eval.py#L429). It requires a
positive paired 95% mass-integral interval on two distinct mixes and no scripted
anchor regression. That is the only evidence sufficient to replace Apex.

## Verification and source boundary

The final non-slow suite completed in 56.37 seconds:

```
1885 passed, 6 skipped, 3 deselected, 1 warning
```

Three skips require an optional champion checkpoint absent from this worktree;
three reward-mismatch cases are inapplicable because the active config's reward
contract already matches the checkpoint. Three tests marked slow were deselected.
The captured output is
[`pytest_non_slow_final.log`](../runs/rl_assessment_20260906/pytest_non_slow_final.log).
All changed Python files passed Black at the repository's 100-character limit,
isort with the Black profile, flake8, and `git diff --check`; the original main
checkout remains clean at `950d583`. Static checks are distinct from the
training results above. Changes are isolated on `codex/rl-assessment-20260906`
in the adjacent `snake-dqn-rl-assessment` worktree. The implementation includes
the rollout-cap fix, raster-hero live evaluator, experimental runner and tests,
shaping documentation correction, and a hermetic checkpoint-loader test. Run
artifacts remain local and ignored by Git; no training process remains active.

Method and PBRS claims rely only on linked primary papers and current source.
Their benchmark results are not projected onto Snake. The completed local screen
supports reproducibility and the stated narrow observations; it does not support
a method winner or a production promotion.
