# RETURN boost-drift bug check — 2026-10-16

**No genuine collector/trainer implementation bug was found in the five checks.**
The decisive bank strongly prefers boost: its source-B8 target mean is **−4.72**
for boost versus **−35.80** for non-boost. The corresponding D300 survival rates
are **87.06% versus 50.07%**. Matched outside states do not show that systematic
preference. This is evidence for a selection/objective explanation of the drift,
not proof of its complete causal mechanism. No reward, loss, training recipe,
bank, checkpoint, launch binding, or preregistration was changed. No training
was relaunched and no gameplay screen or promotion evaluation was run.

This implements [the bug-check spec](specs/t4e_bugcheck_spec_2026-10-16.md), using
Deviation 1 of [amendment 1](specs/t4e_prereg_amendment1_2026-10-16.md) and the
[return tooling contract](t4_return_regression_tooling_2026-10-16.md). Initial
checkout: clean `t4d-20261016`, `be09d62061952095d620e50d9b6b1b9bfbcd3816`.

## Evidence scope and missing artifacts

The audit covers **all 97,846 training branches / 18,924 states / 330 shards**
and **5,418 held-out decisive branches / 1,052 states / 30 shards**. Every
published shard hash and exact shard set was checked against its manifest.
Numerical results are descriptive, not a new held-out selection rule.

Both `HALTED_boost_RETURN_s10` and `HALTED_boost_RETURN_s11` contain only
`log.jsonl` and `telemetry.jsonl`; **neither saved a checkpoint**. They stopped
before the first scheduled 2M checkpoint. Their last log rows are 800,000 and
800,128 transitions. Thus targets bootstrapped by the halted online/target
networks, and their Q values on the bank, **cannot be recovered**. Calibration
checkpoints were not substituted for these missing weights.

For the available target calculation, each row uses the source final `B8_sN`
that generated its continuation, as both online selector and target evaluator.
The CTRL comparison uses `CTRL_s(10+N)` on exactly the same initial states.
CTRL 2M is the earliest saved checkpoint and CTRL final is 8M; neither is a
time-matched 0.8M comparison. All checkpoint paths and SHA256 identities are
retained in the numerical evidence.

## 1. Action identity — PASS, with historical trace limitation

The action contract is relative: **0/1/2 = left/straight/right**, and **3/4/5 =
the same turns with boost**. `batch_sim.py` decodes direction using `action % 3`
and boost using `action >= 3`. The bank records that integer unchanged:
`collector._snapshot` obtains advisory-legal indices, `rollout_branch` passes
the index to `bank_engine.step_prepared`, and `publisher._branch_rows` stores
the branch dictionary key. The trainer writes its selected Q-column integer
directly into the simulator action array (`train_m3c.py:2538`). There is no
absolute/relative conversion between these paths.

Snapshots with no advisory action are rejected. On admitted snapshots, the
advisory and resolved masks agree. Survivor terminal inputs/masks are captured
at the actual next prepared selector by `collector._terminal_observation`;
the resolved terminal mask matches the trainer's bootstrap convention.

Replay evidence is recorded in the companion result receipt. The decisive
source is B8_s0, world seed **2151847346**, decision **3960**, initial length
**1318**, all actions **0–5**. Its initial packed raster and scalars are checked
against the saved bank before forcing a branch.

The initial head is `(125,10)`, heading down (2). Reconstructed first frames
match the independent public dispatch/step/observe path, including body/grid,
food, and RNG state hashes. All six complete continuations match saved T,
death, L0/L_T and S (absolute tolerance `1e-9`), with byte-identical terminal
packed observations, scalars, and masks:

| Action | First head / heading | T | Died | Terminal length |
| --- | --- | ---: | --- | ---: |
| 0 | (126,10) / right | 297 | Yes | 0 |
| 1 | (125,11) / down | 300 | No | 1432 |
| 2 | (124,10) / left | 300 | No | 1432 |
| 3 | (127,10) / right | 300 | No | 1325 |
| 4 | (125,12) / down | 300 | No | 1428 |
| 5 | (123,10) / left | 300 | No | 1424 |

**The historical bank has no saved first-frame trace.** Its schema stores s0,
action, T, S, lengths, death status, and survivor terminal inputs. Accordingly,
the replay compares a reconstructed collector first frame with direct action
application and binds the full continuation to the saved row. It cannot claim
comparison against an independently archived historical first frame.

## 2. Bank target statistics — boost preference localized to decisive states

The target is `y = G + gamma^T * Q_B8(sT, masked_argmax Q_B8(sT))` for a
survivor and `y = G` for a death, with gamma 0.998 and kappa 1. Values below
describe every branch, without sampling. Survival means alive after D300;
deaths exactly at transition 300 count as deaths. Huber is applied to these
targets later; they are not clipped to delta 100.

| Split / family | Action class | Branches | Survival | Mean y | p05 | Median | p95 | Min / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train decisive | Non-boost | 25,619 | 50.07% | −35.80 | −105.33 | −45.98 | 13.34 | −147.57 / 19.18 |
| Train decisive | Boost | 24,034 | 87.06% | −4.72 | −85.16 | 8.98 | 14.37 | −149.35 / 21.96 |
| Train outside | Non-boost | 24,958 | 97.52% | 4.56 | −30.69 | 9.29 | 14.25 | −151.57 / 19.84 |
| Train outside | Boost | 23,235 | 97.11% | 4.64 | −31.66 | 9.29 | 14.29 | −151.57 / 20.17 |
| Held-out decisive | Non-boost | 2,796 | 47.89% | −38.85 | −107.29 | −51.57 | 13.37 | −130.69 / 17.03 |
| Held-out decisive | Boost | 2,622 | 83.07% | −9.38 | −89.56 | 7.37 | 14.38 | −130.95 / 16.57 |

Pooling branches gives boost-minus-nonboost mean gaps **+31.08** (training
decisive), **+0.085** (outside), and **+29.47** (held-out decisive). Because
legal-action counts vary, also compare the two action-class means *within each
state*, then average states equally:

| Family | States with both classes | Mean paired gap | Median paired gap | Fraction positive |
| --- | ---: | ---: | ---: | ---: |
| Train decisive | 9,457 / 9,462 | +31.99 | +29.52 | 88.04% |
| Train outside | 9,438 / 9,462 | −0.159 | −0.013 | 49.33% |
| Held-out decisive | 1,051 / 1,052 | +30.66 | +29.01 | 85.44% |

The decisive effect holds for each source B8 final: training paired means range
from +27.39 to +36.60. The published outside sample is source-imbalanced
(8,571 B8_s0 states, 518 B8_s1, 373 B8_s2, no B8_s3/s4); its combined result
does not establish an independently balanced result across all five finals.

The advantage mostly exists **before bootstrap**: decisive mean G is −36.35
non-boost versus −5.93 boost, while mean bootstrap contributions are +0.55
and +1.22. Outside mean G is +2.70 versus +2.56. Decisive selection explicitly
requires the B8 greedy action to die and some alternative to survive; these
outcomes are therefore conditional on selection, not general boost safety.

At the source B8 weights, the mean scalar weighted Huber derivative with
respect to the selected Q is +0.06539 for decisive non-boost versus +0.01581
for decisive boost. Gradient descent therefore has a stronger direct tendency
to lower non-boost predictions in this sample. Outside values are +0.00149
and +0.00169. These are `W * row_weight * clip(Q-y, -100, 100)` class means,
not shared-parameter gradient norms or causal attribution of policy drift.

**Verdict:** systematic preference exists in decisive states, not across the
outside control states. Halted-checkpoint bootstrap statistics are unavailable.

## 3. Target versus trainer reward — PASS

Both paths use gamma **0.998**, kappa **1**, `Phi(L)=L/10`, `Phi(dead)=0`,
death value **−3**, and no kill reward or additional overlays. The collector
retains raw `S = sum(gamma^t * Phi(L_(t+1)) * alive_(t+1))`. Production
`bank.realized_returns` computes:

```text
G = kappa*(1-gamma)*S - Phi(L0)
    + gamma^T*Phi(L_T)                 if alive at T
    - 3*gamma^(T-1)                    if dead at T
```

The trainer adds `mass_term` to the simulator's PBRS/death reward using the
post-step length and alive status (`train_m3c.py:2558`). Boost burn reduces
length; trail pellets enter the same simulator food path and any later growth
enters S. There is no omitted independent trail reward or positive boost bonus.
At kappa 1, a live step simplifies to `Phi(next)-Phi(previous)`: an otherwise
pure one-unit burn costs **0.1**.

The collector's evaluator simulator has gamma .99 / kill-scale .3 defaults,
but its numerical `get_reward()` is unused in constructing S or G. That is
not a target mismatch. The reward replay changes only those reward fields to
the trainer settings, verifies unchanged transitions, and sums the actual
simulator reward plus the trainer's mass helper on a boost branch.
For action 3, the discounted trainer reward is **0.5495780763242173**, versus
closed-form G **0.5495780763234279**, absolute error **7.89e-13**; all 300
transition-state hashes agree after changing only the reward settings.
The 12 formula cases also pass for boost, death, growth, and survival at three
gamma/kappa combinations. The boost fixture now explicitly asserts lengths
`[6,6,5]` across three boosted frames and a tail pellet at `(13,10)` in both
the corpse and food sets; this guards burn/trail accounting.

## 4. Trainer loss and logging — PASS

`learner_m3c.return_regression_loss:169–183` evaluates online Q on s0, gathers
the bank action, evaluates both bootstrap networks on sT, selects with the sT
mask, and detaches the entire target. Dead branches receive no bootstrap.
An empty terminal mask falls back to actions 0–2 exactly as ordinary TD does.
Publication repeats the same initial observation for each action-keyed branch;
`ReturnBank.sample` uses one shared row-index vector for all fields.

Both reference and fast paths use **TD delta 1**, **return delta 100**, and
add **W=0.0032 exactly once**. Effective row weights are decisive 1.0 within
100 decisions of death, decisive 0.3 at 100–299, and outside 0.5. Loss reduction
is `mean(row_weight * Huber(Q(s0,a), y))`, not normalization by sum of weights.
At batch scale 2, `return_frac=.5` adds **128** return rows alongside **256**
full on-policy anchor rows and **768** agent TD rows, as every halted log
reports. Return rows do not consume the anchor allocation.

`return_loss` logs the row-weighted loss **before W**. At the last s10/s11
rows, 200.9203 / 196.0053 corresponds to objective contributions
**0.642945 / 0.627217**. Gradient telemetry includes W and measures the same
update before clipping; the final return/TD norm ratios are **2.7509 / 3.6401**.
The rising ratio is real in the logs, not a missing-weight logging conversion.

The added `test_return_loss_keeps_initial_state_action_and_terminal_state_aligned`
uses distinct initial/terminal state-dependent Q values, actions 3/5/1, masks,
death suppression, and exact gradients. It closes a constant-Q test blind spot;
it did not expose a product bug. Existing reference/fast and separate-delta
tests also pass.

## 5. Q scale — no clamp; same-state evidence qualifies the drift explanation

The halted final logged max|Q| values are **28.36 / 23.47**, with greedy boost
**35.19% / 36.12%**. At approximately 0.8M, the actual CTRL_s10/s11 logs have
max|Q| **106.23 / 105.89** and greedy boost **0.0819% / 0.0989%**. These logs
measure all six Q outputs on sampled agent-current states; they do not compare
identical states. The amendment's 110–140 description should not be treated as
an exact value for these particular two log windows.

On identical saved bank states, source-matched Q values are:

| Train family | Model | All-six median Q | Signed min / max Q | max abs Q |
| --- | --- | ---: | ---: | ---: |
| Decisive | B8 | 7.60 | −125.53 / 14.14 | 125.53 |
| Decisive | CTRL 2M | 8.47 | −121.52 / 14.83 | 121.52 |
| Decisive | CTRL final | 9.23 | −127.90 / 15.82 | 127.90 |
| Outside | B8 | 7.58 | −120.34 / 14.59 | 120.34 |
| Outside | CTRL 2M | 8.44 | −115.63 / 14.88 | 115.63 |
| Outside | CTRL final | 9.19 | −122.97 / 16.44 | 122.97 |

Each row pools six outputs per unique state: **56,772 values for each family**.
Among bank-recorded legal actions alone, CTRL 2M medians are 8.64 / 8.67 and
final medians 9.42 / 9.44. Thus a max absolute magnitude near 120 does not mean
typical positive Q values near 120. The return targets span roughly **−152 to
+22**, with the large negative tail generated by fatal branches; they are not
all confined near 25 either.

There is no output clamp: the network has a linear dueling head, and Huber
delta limits derivatives rather than Q/target values. With constant living
length and kappa 1, G cancels to zero and a D300 survivor has
`y = 0.998^300 * Qtarget(sT)`, approximately **0.54848 times** the endpoint Q.
Death instead retains the substantial initial-potential subtraction. This
objective supplies downward pressure, especially on decisive non-boost
actions. Its increasing gradient contribution and the simultaneous change in
boost/death behavior are consistent with interference in shared network
parameters and a changed sampled-state distribution.

**Verdict:** no Q-scaling implementation error or 25-Q cap was found. The
evidence supports the objective/selection explanation, but cannot establish
why the halted network's sampled maximum settled at that specific magnitude
without its missing weights/target network and replay states. It would be
incorrect to label that mechanism fully proven by these logs.

## Reproduction, changes, and verification

New tools: `research/t4e_return_20261016/bugcheck.py` (streamed numerical audit)
and `bugcheck_replay.py` (saved-state replay). New tests cover action/state
alignment, branch-versus-state gap denominators, and invalid diagnostic inputs.
No changes were made to the production collector, trainer, simulator, or bank.

All work used CPU, one numerical thread per process, with at most four
task-owned OS processes; no subprocess pool, training actor, tmux command,
process discovery, or signal to any existing process was used. Only
task-owned tool sessions were polled. Numerical v2 runtime was **64.36 s**.

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-bugcheck-20261016/`.
The companion `t4e_bugcheck_result_2026-10-16.json` binds the final replay,
statistics, source files, inputs, and verification outputs. Raw per-branch
values are in `statistics-v2/{train,holdout}-rows.npz`; unique-state Q arrays
are in `statistics-v2/{train,holdout}-states.npz`. Full statistics include
quartiles, standard deviations, G/bootstrap decomposition, and source strata.
The final replay is `replay-v3.json`, rerun on the final formatted source.
`replay-v1.json` is a preserved, superseded preliminary check that did not
compare every endpoint field despite its overly broad description; do not
use it as full replay proof. `replay-v2.json` contains the full checks and
was independently reviewed; v3 repeats them after import/string formatting.

```bash
./venv/bin/python -m research.t4e_return_20261016.bugcheck \
  --artifacts /Users/josenunez/Projects/ml/snake-dqn-artifacts \
  --out /path/to/new/create-only-output

./venv/bin/python -m research.t4e_return_20261016.bugcheck_replay \
  --out /path/to/new/create-only-replay.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
./venv/bin/python -m pytest -q tests/test_t4e_bugcheck.py \
  tests/test_t4d_return_bank.py tests/test_t4d_formula.py \
  tests/test_t4e_calibration_telemetry.py tests/test_perf_learner.py \
  tests/test_t4d_collection.py tests/test_t4e_collection.py \
  -m 'not slow and not requires_gpu' -k 'not mps and not trainer_runs'
```

**60 tests passed in 9.91 s**, three deliberately deselected (the two tests
with MPS in their names and the trainer CLI).
Scoped Black (one worker), isort (Black profile, line length 100), flake8, and
`git diff --check` passed. The read-only independent loss/diagnostic review
found no actionable correctness issue. An independent read-only review of
the full replay and report also found no material correctness issue.

Implementation/initial verification commit: `4e80836`. Each commit is followed
by `git fetch origin && git merge origin/main`, then `git push origin HEAD:main`
and `git push origin t4d-20261016`, as specified. The final report commit SHA
is reported in the task's final response to avoid a self-referential hash.
