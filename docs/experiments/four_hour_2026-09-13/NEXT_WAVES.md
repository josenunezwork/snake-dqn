# Next bounded waves — 2026-09-13

## Status and decision

**PLANNED / UNRUN.** This is an implementation and experiment-design packet. It does
not establish an implementation or experimental result. Future execution follows the
existing user scope, dependency and verification gates, and a declared resource window.
Apex `vector61` remains incumbent, and the repaired paired evaluator remains the only
promotion authority.

The intended audience is the next implementation, verification, and experiment owners.
They need working knowledge of raster/PQN, its reward contract, checkpoint provenance,
and the campaign's serialized resource policy. Start from the current
[handoff](HANDOFF.md), [ledger](EXPERIMENT_LEDGER.md), and artifact
[objective diagnosis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/objective-diagnosis.md>).

The confirmed finding is an alignment gap: the intended discounted training return is
non-equivalent to the 5,000-frame undiscounted mass-integral promotion metric. Whether
that gap caused the observed trajectories remains unproven. Under true death, or an
exact transformed bootstrap for a truncation, length PBRS differs from the sparse base
return by a start-state constant; finite approximate Q(lambda) boundaries can retain
residual effects.

## First control: a PQN-only opt-in base objective term

The proposed control adds `living_mass_reward_coefficient: float = 0.0` to PQN only.
With coefficient `beta`, the training reward is:

```text
r_train = r_v2 + beta * valid * post_alive * post_logical_length
```

The proposed first treatment freezes `beta = 0.003` for `gamma = 0.997`. Call it a
**base objective term**, never mass shaping: it intentionally changes policy preferences
and cannot telescope. Keep `compute_reward_v2` and `BatchSim._compute_rewards`
unchanged. Zero mode must return the current reward unchanged and preserve the current
`pqn_reward_contract` descriptor and digest exactly. Nonzero mode needs a distinct
experiment contract/version recording beta, gamma, maximum frames, post-step convention,
zero-on-death rule, floor/cap inclusion, and no clipping. Existing target/run provenance
digests and resume validation must reject cross-objective continuation.

The treatment is mass-aligned, not product-exact: sparse events, `gamma=0.997`, and
frame-cap bootstrap remain. Exact product alignment is a separate future migration:
`gamma=1`, base `post_alive * mass / 5000`, removal of sparse kill/death base terms,
and a cap terminal objective with zero bootstrap. Do not include that mode in this first
control.

### Boundary contract

| Transition boundary | Overlay | Existing target behavior | Required interpretation |
| --- | --- | --- | --- |
| Ordinary valid step | Add `beta * post_mass` | Existing Q(lambda) path | Treatment term applies. |
| Collision death | Zero | Existing reward-only terminal | Do not pay living mass after death. |
| Population-floor survivor | Include | Existing masked bootstrap | Training-only cutoff remains; record it as a boundary. |
| 5,000-frame cap survivor | Include | Existing masked bootstrap | Isolates the reward term while retaining a known alignment mismatch. |
| Rollout edge | Include | Existing bootstrap | Preserve current return behavior. |
| Invalid or inactive row | Zero | Excluded / break carry | Never create a target contribution. |

### Implementation packets

1. **Trainer and contract owner:** own `src/training/pqn_trainer.py`. Validate the
   configuration, add one helper at the two current reward-copy seams (source
   `44acc33`, around lines 1555 and 1664), create the nonzero dynamic
   `pqn_reward_contract` near line 351, add split base/overlay/total telemetry, and
   preserve checkpoint defaults.
2. **CLI owner:** own `src/scripts/train_pqn.py`. Add
   `--living-mass-reward-coefficient`, its CLI mapping, and field-source reporting.
   Do not overload `reward_version=2`; add YAML schema only if an experiment needs it.
3. **Test owner:** after the interfaces settle, own `tests/test_pqn_trainer.py`,
   `tests/test_pqn_episode_contract.py`, and `tests/test_train_pqn.py`. Cover zero
   byte/numeric preservation; ordinary, growth, boost, death, same-frame kill/death,
   floor, cap, and invalid rows; both rollout paths; telemetry identity; digest/resume
   rejection; weights-only loading; and invalid coefficients.
4. **Verifier:** demonstrate deterministic zero-mode action, RNG, target, and parameter
   parity; run focused tests, full non-slow verification, and a short two-mode CPU smoke.
   These checks are implementation evidence, not a policy result.
5. **Artifact protocol owner:** freeze source, configuration, seeds, initial-weight hash,
   resource limits, and evaluation closure before any numerical arm starts.

Relevant code evidence: `src/core/reward_events.py:8-21,77-123`,
`src/simd_env/batch_sim.py:548-579,1238-1271`,
`src/training/pqn_trainer.py:134-332,351-408,1495-1727,1808-1888,2118-2225,2280-2452`,
`src/scripts/train_pqn.py:277-437,665-765`, `src/simd_env/featurizer.py:782-792`,
and `src/evaluation/metrics.py:106-161`. The frozen training source is
`/Users/josenunez/Projects/ml/snake-dqn-four-hour-source` at `44acc33`.

## Proposed first study

This study is also **PLANNED / UNRUN**. Run exactly two serial CPU arms with two numerical
threads: one control (`beta=0`) and one treatment (`beta=.003`). They share one
matched training seed, byte-identical initial weights and RNG, 50,000 useful hero
transitions, `E16/S6/T16`, corrected-v3/raster31v3 Watch phase, per-environment
derived autoreset, exact one epoch with padded batches, Adam/Huber, epsilon `1 -> .02`
over 30,000 steps, and one hero plus five fixed `random_safe` opponents. Preserve the
same guards and resource limits for both arms.

Evaluate the shared initial checkpoint and both finals on frozen 5,000-frame worlds in
separate `random_safe` and `greedy_food` mixes. One learned pair is descriptive only:
it supplies no confidence interval, promotion, or algorithm claim. Advance to a
separately frozen multi-seed study only if both mix deltas are positive and all gates are
clean. Do not tune beta from this outcome.

## Controls that must remain separate

| Change | Why it is a separate experiment |
| --- | --- |
| PBRS coefficient | Changes credit/Q scale but does not add a base mass goal under valid invariance conditions. |
| Loss or optimizer | Changes optimization, not the return. |
| Replay | Changes distribution, target construction, and memory behavior. |
| Network or observation | Changes function class or available information. |
| Starvation rule | Changes game physics. |
| Exact product-alignment objective | Is a broader objective/runtime migration. |

The qualified circular-spawn fix covers the `GameState` snapped-spawn path. Direct
`GameLogic` helper callers, including `FoodManager` and game-logic spawn paths, can still floor a
position outside the circular safety boundary. They were outside the bounded fix and need a
separate caller inventory, contract decision, and focused regression packet. Do not claim that all
circular spawn paths are repaired.

The original `objective-trace/v1` packet remains frozen and unrun. Its
`objective-trace-execution/v2` successor completed its separately bound four-call diagnostic
parity check; consult the current [handoff](HANDOFF.md) and ledger for its retained terminal.
The training-side TD trace remains unrun. A source audit found no action-index, mask, loss, or
Q(lambda) direction defect, but identified fixed east-facing reset/respawn as a plausible
reflection-symmetry-breaking exposure. Its retained telemetry also shows right was the least
selected relative direction in both operational arms. The training-side trace must therefore
capture per eligible row: world/episode/frame, heading and head position, six mask bits,
exploration flag, selected action, six pre-update Q values, bootstrap argmax, Q(lambda) target,
taken Q, TD residual, and unreduced Huber loss. Aggregate by relative direction, boost, heading,
position/wall quadrant, and world. A later matched intervention can randomize four start headings
or compare physical east/west mirrors; it must preserve initial weights and world content. Read
the receipt-backed [direction-bias audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/direction-bias-audit.md>)
before selecting that intervention.

Each later trace must retain its own manifest, source closure, bound checkpoint hashes, declared
call limit, and receipt; it must not be folded into the living-mass control.

## References and boundaries

The design is grounded in the campaign's
[objective diagnosis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/objective-diagnosis.md>)
and earlier [reward-objective research](</Users/josenunez/Projects/ml/snake-dqn-artifacts/research-portfolio-20260912/research/reward-objective.md>).
For the formal distinctions, consult [Ng, Harada, and Russell on policy-invariant reward
transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[Pardo et al. on time limits](https://proceedings.mlr.press/v80/pardo18a.html), and
[Forbes et al.](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p589.pdf). These references
inform the design; completed verification and declared resource boundaries determine execution
status and the meaning of any result.
