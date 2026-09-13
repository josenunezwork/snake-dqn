# PQN decision-phase contract — 2026-09-12

## Outcome

The raster PQN path now has an **opt-in** Watch-aligned decision phase:
`watch_pre_move_v1`. It is available only with `corrected-v3` and
`raster31v3`. The established default remains `pre_transition_v1`; existing
legacy behavior and old checkpoint semantics are preserved rather than
silently reinterpreted.

The opt-in path addresses a confirmed trainer/evaluation/serving phase
discrepancy. It is an implementation and contract change, not a learning
result, an algorithm comparison, a promotion candidate, or a claim that every
old decision was wrong. Focused contract tests and the CPU non-slow suite have
passed. The bounded CPU and MPS five-update executions completed, and the
independent receipt audit passed for engineering qualification.

## Confirmed witness and its limit

The completed fixed-seed phase witness compared the trainer's action-observation
boundary with the prepared policy boundary. It observed a change in food count
from 1 to 2, frame from 0 to 1, and the normalized episode-progress scalar
from 0 to 0.0002 before the prepared policy observation, while the original
trainer decision observation remained
pre-transition. The retained v4 evidence is
[the witness JSON](</Users/josenunez/Projects/ml/snake-dqn-artifacts/research-portfolio-20260912/phase-witness/runs/phase-witness-v4.json>)
and [its supervision receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/research-portfolio-20260912/supervisor-runs/phase-witness-v4/receipt.json>).

That proves a phase-contract difference. It does not prove a different chosen
action, an incorrect optimal action, a lower return, a bad learned policy, or
a tournament ranking change. Those are distinct empirical questions. The
witness did not train a network or select a checkpoint.

The portfolio ledger retains the failed witness attempts and the accepted v4
outcome, along with the deferred learning probe:
[experiment ledger](</Users/josenunez/Projects/ml/snake-dqn-research-portfolio/docs/experiments/rl_research_portfolio_2026-09-12/EXPERIMENT_LEDGER.md>).
The planned two-seed LP1 learning probe consumed **zero** MPS arms and zero
evaluation calls; its former manifest is a deferral record, not a negative
PQN result.

## Modes

| Mode | Selection boundary | Availability and compatibility | Checkpoint meaning |
| --- | --- | --- | --- |
| `pre_transition_v1` | The trainer builds the current observation and selects an action before that `BatchSim.step()` performs the next frame's food-maintenance transition. | Default. This preserves the established behavior. | Old checkpoints have no phase field and remain qualified only for this legacy interpretation. |
| `watch_pre_move_v1` | `BatchSim.step_with_policy()` advances the active world through the Watch pre-move preparation, including frame/food maintenance and applicable respawn work, then invokes the selector on that prepared world. | Opt-in; `PQNConfig` rejects it unless `recipe="corrected-v3"` and `obs_spec="raster31v3"`. | Checkpoints explicitly declare the decision phase and the changed target contract. |

The watch path does not precompute an action from a stale observation and then
apply it to the prepared world. The selector receives the prepared state, its
resolved action mask, and chooses the action that is executed in that same
transition. The implementation is in
[the trainer](../src/training/pqn_trainer.py) and uses the simulator's existing
`step_with_policy` surface.

## Successor and bootstrap semantics

In Watch mode, stored rollout observations are prepared decision states. For a
continuing environment, the successor for the next target is the next prepared
Watch decision state, including its resolved mask. A terminal transition keeps
its true post-transition terminal result; the code does not invent a prepared
successor for a dead or completed environment.

At the final rollout edge, a continuing successor is captured from a
`deepcopy` of the simulator. The clone enters its preparation phase, a callback
captures the observation and mask, and a private sentinel stops the clone
before any decision is persisted. The clone is discarded. This preview is
therefore an observation/mask capture for target bootstrapping, not a second
environment transition, a duplicate reward, an episode-state mutation, or a
new persisted simulator state.

The target descriptor records this distinction as:

```text
decision_phase: watch_pre_move_after_frame_food_maintenance_v1
successor_phase: next_watch_pre_move_or_terminal_post_transition_v1
rollout_edge_bootstrap: deepcopy_selector_capture_discards_prepared_clone_v1
```

These fields make the intended semantics inspectable in the checkpoint rather
than relying on a CLI flag remembered outside the artifact.

## Configuration

`train_pqn.py` accepts the mode through the normal corrected-PQN configuration
merge. In YAML, set it beneath the PQN block:

```yaml
pqn:
  decision_phase_mode: watch_pre_move_v1
```

The equivalent CLI override is:

```bash
--decision-phase-mode watch_pre_move_v1
```

For an experiment, freeze the source revision, complete configuration, seeds,
and output directory before starting it. Keep the terminal receipt and raw
telemetry with the output so a later comparison can distinguish a completed
run from a stopped or capped one.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  src/scripts/train_pqn.py \
  --recipe corrected-v3 \
  --device cpu \
  --envs 16 \
  --snakes 6 \
  --rollout-len 16 \
  --decision-phase-mode watch_pre_move_v1 \
  --total-steps 1280 \
  --seed 20260912 \
  --out-dir runs/pqn_watch_example \
  --ckpt-every 0
```

This example demonstrates configuration only; it is not a policy-quality or
promotion run. The old LP1 runner is deferred and must not be reused under the
new mode. Any later 50,000-step study needs a new source/configuration closure
and seed ledger so its result cannot be mistaken for the legacy plan.

## Provenance and continuation boundary

The Watch checkpoint writes `decision_phase_mode: watch_pre_move_v1` and uses
the target-contract version
`pqn-qlambda-corrected-v3-lifecycle-decision-v1`. Its target descriptor includes
the three phase values above. `RunProvenance` binds those target semantics
through the target descriptor and digest; the policy-source descriptor and
sampler contract are unchanged. The metadata validator rejects a Watch target that lacks
the matching top-level phase field, and rejects a legacy lifecycle target that
claims to carry one.

Continuation compares the semantic contracts before installing state. A
pre-transition checkpoint cannot continue as Watch mode, and a Watch checkpoint
cannot continue as pre-transition mode. Weights-only loading remains a separate
fresh-state operation and must still satisfy its own checkpoint validation;
it is not a way to call a cross-phase continuation valid.

This boundary protects the fact that the targets optimized before and after the
change name different observation/successor phases. It also keeps old
checkpoints usable under their original semantics without claiming they were
trained on Watch-prepared states.

## Qualification status

The source implementation is committed in a dedicated worktree (`ec39ece`,
with test-fixture correction `b760b40`). This documentation does not convert a
source edit or a test log into a green policy qualification. The durable
execution root for this repair is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/decision-phase-20260912`.

| Check | Required evidence | Status in this document |
| --- | --- | --- |
| Fixed phase witness | Frozen inputs, raw pre/post observations and masks, supervised terminal receipt | Complete; establishes phase difference only |
| Legacy default compatibility | Explicit legacy regression and old-contract checkpoint checks | Focused regression and lifecycle-contract tests pass; receipt review remains part of the device qualifier |
| Watch selection/transition parity | Prepared observation and selected action must be the executed Watch transition | Focused decision-phase tests pass |
| Target boundaries | Continuing prepared successor, terminal post-transition handling, and discarded preview clone | Focused decision-phase tests pass |
| CLI/YAML/config validation | Accepted opt-in, rejected unsupported mode/recipe combinations | Focused contract tests pass |
| Checkpoint/provenance | Exact phase fields and target version; tampering rejected | Focused contract and continuation tests pass |
| Cross-phase continuation | Both directions reject before state installation | Focused continuation tests pass |
| CPU non-slow suite | Broad regression check excluding marked slow tests | 2,684 passed; 6 skipped; 3 deselected; 1 warning in 95.36 s (96.43 s supervised wall time), with no source drift |
| CPU and MPS qualifier | One bounded five-update run per device with supervision/resource receipts | Both completed naturally without source/driver drift: 1,280 hero transitions and five updates per device; coverage was eligible/drawn/unique = 1,280/1,280/1,280, weights changed, and serialized strict-profile checkpoint validation passed. CPU: 6.61 s worker / 6.96 s supervised wall, peak RSS 737 MB. MPS: 5.26 s worker / 5.71 s supervised wall, peak RSS 713 MB and peak MPS driver memory 1.16 GB. The different seeds mean these times are not a speed comparison. The [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/decision-phase-20260912/independent-audit/phase-qualification-v1.md>) passed engineering qualification. |

Neither brief smoke completed an episode or performed evaluation steps. The
autoreset boundary is covered by the focused contract tests, not demonstrated
by these five updates. The device runs establish bounded training/checkpoint
health only; they do not establish policy quality.

The evidence above does not establish a policy-quality improvement or a
promotion. A later learnability study should use a successor LP1 protocol; it
must not resume or relabel the deferred pre-transition plan.
