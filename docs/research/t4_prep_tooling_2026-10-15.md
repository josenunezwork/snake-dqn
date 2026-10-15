# T4 preparation tooling — 2026-10-15

Development tools and measurements only. These results are not strict-gate,
training, serving, or promotion evidence. The T3 bank is an immutable input.
No training or long collection was launched for this task.

## Scope and resource discipline

The implementation follows `specs/t4_prep_spec_2026-10-15.md` and advisor
items 1a, 2a, 2b-data and 2c. Measurements run sequentially, with at most two
single-threaded workers and MPS measurements bounded below five minutes.
Existing tmux sessions and processes are outside this task's ownership.
The existing trainer, learner and T2/T3 probe defaults remain unchanged.

## A. Batched screen probe

`research.t4_screen_probe_20261015.screen_probe` reuses the T2 seed allocator,
roster materializer, evaluation profile, advisory masks and opponent dispatch.
Each checkpoint/mix runs as one GridBatchSim with all requested worlds.
`--path batched` is the screen path; `--path compare` measures it against the
existing CPU rowwise path. `--device auto` selects MPS when available, otherwise
CPU. Both paths use a two-world/eight-frame warmup outside their timed run.
The MPS deadline includes warmup and defaults to 295 seconds.

The comparison reports matching decisions while both heroes are alive, a union
denominator that counts missing post-death decisions as disagreements, per-world
mass and survival differences, exact outcome agreement, and alive/dead outcome
agreement separately. Action agreement after a trajectory diverges is not a
comparison on identical states. It cannot by itself attribute a discrepancy to
a float tie.

A screen must use the same forward path for every arm and control. Batched
floating-point inference may change close action rankings; an observed speed-up
does not establish numerical identity or permit use for strict gates.

Measured on the busy Mac with the fixed `LABELS_s10/final.pth` checkpoint,
32 scripted round-60 worlds at H5000, one process and one CPU thread:

| Measurement | Result |
|---|---:|
| CPU rowwise wall time | 75.304615 s |
| MPS batched wall time | 64.241594 s |
| Rowwise / batched speed-up | **1.172210×** |
| Identical hero decisions (common-time and union) | **123,375 / 123,375 (100%)** |
| Identical mass/survival/death outcomes | **32 / 32** |
| Mean and maximum absolute mass difference | 0 |
| Mean and maximum absolute alive-frame survival difference | 0 |
| Episode-survival difference | 0 |

No float-tie divergence was observed in this sample. The modest measured gain
does not support the advisor's speculative 5–10× estimate on this workload.
This is one paired measurement under concurrent Mac load, not a general speed
guarantee. The MPS timed segment was about 64 seconds, below the five-minute cap.

## B. Bank v2 labels and branch outcomes

The existing exported bank stores a six-action `surviving` mask. That is enough
to derive the smallest sufficient set without rerunning the simulator: retain
surviving non-boost actions 0–2 whenever that subset is nonempty; otherwise retain
surviving boost actions 3–5. This is a label preference, not a new claim that the
removed boost actions die. Preserve the original outcome mask as provenance.

The original provenance JSONL identifies source world, decision and frame; it
does not contain complete per-action death frames and final lengths. Those
cannot be recovered from a survival bit. New collection can opt into complete
branch endpoint records with `--record-outcomes`; ordinary collection keeps
its original defaults. `--label-policy surviving-set` is the default and
`--label-policy prefer-nonboost` selects the new preference.

The offline relabel changed **14,011 of 16,327 rows (85.815%)**. Boost labels
fell from **55.8620% to 8.5810%** of all selected action labels: the derived bank
has 2,474 boost labels among 28,831 total labels. Boost-only escape sets remain
intact. The derived shard stores `surviving_original` and every other source
array is byte-equivalent. All 16,327 rows load through the existing `LabelBank`.
The derived manifest binds the entire source tree and its own shard hashes.

With `--record-outcomes`, both world shards and merged banks carry these arrays:

| Array (shape N×6) | Meaning |
|---|---|
| `outcome_recorded` (bool) | Action was rolled out or proved by the actual fatal trajectory; false means unknown |
| `outcome_survived` (bool) | Raw survival result, independent of label preference |
| `outcome_death_frame` (int64) | Zero-based episode **decision** of death; -1 encodes null |
| `outcome_final_length` (int64) | Logical live mass at horizon; zero for dead branches |

Use `outcome_recorded` before interpreting any endpoint. Advisory-illegal,
unrolled actions are not labelled as observed deaths. The historical
`death_frame` array name uses the explicit decision convention above, rather
than the simulator's one-based frame convention. The default-off rollout API
and ordinary bank schema remain unchanged when outcomes are not requested.

### Future return-regression target (design only)

The T3 lane recipe uses `--reward mass --mass-kappa 1.0 --kill-scale 0` and
fixed gamma 0.998. In `train_m3c.mass_term`, the mass reward is
`kappa * (1 - gamma) * Phi(L_next)` on a surviving transition and zero at
death, where `Phi(L) = L / 10`. `GridBatchSim._compute_rewards` supplies
`gamma * Phi_next - Phi_current` plus death reward -3, with terminal potential
zero. For a branch of T transitions (ending at death or the D300 horizon),
the realised truncated return is therefore

```text
gamma = 0.998; kappa = 1; Phi_terminal = 0
r_t = kappa * (1 - gamma) * Phi(L_(t+1)) * alive_(t+1)
      + gamma * Phi_(t+1) - Phi_t - 3 * died_(t+1)
G_T = sum(t=0..T-1, gamma^t * r_t)
    = kappa * (1 - gamma)
      * sum(t=0..T-1, gamma^t * Phi(L_(t+1)) * alive_(t+1))
      - Phi(L_0) + gamma^T * Phi_T
      - 3 * gamma^(T-1) * died_by_T
```

The initial potential comes from the stored observation's length. Survived,
death frame and final length specify the stopping time and endpoint terms,
but **do not determine the discounted mass sum**. Exact return regression
needs that sum (or the actual reward sequence) recorded during a future rollout;
substituting final length throughout the branch is only an approximation.
A survivor stopped at D300 is a truncation, not a terminal state. For a full
Q target, a separately specified frozen continuation value would add
`gamma^T * V(s_T)` on survivors only; fitting `G_T` alone learns a truncated
return. No return-regression trainer or bootstrap choice is implemented here.
Any future recipe with kill reward or M6 reward overlays must include those
terms as well rather than reuse this formula silently.

## C. Expert iteration

The student checkpoint must drive both the played trajectory and every rollout
continuation. Each collection round needs a fresh audited namespace, new output
directory and hash-bound checkpoint identity. The validation below uses only
two worlds; the two-hour command is documentation for a future authorized run.

The new entry point is
`research.t4_expert_iteration_20261015.collect_for_student`. It uses one serial
CPU worker; no learner is started. The collection time budget is checked between
worlds, so a world already in progress can exceed the nominal limit. Use a fresh
round name and output directory each time. A two-world validation may yield no
qualifying rows; that does not authorize a larger collection to fill a bank.

Validation completed exactly **two** fresh scripted H5000 worlds in
**14.926029 seconds**: 10,000 play decisions, 1,081 dense candidates, and zero
Q/action mismatches. Both heroes survived, so there were **zero death-window
rollout batches and zero bank rows**. Namespace audit found no concrete hits
or overlap with generated NI or dev rounds 52, 60, and 61. The manifest and two
world receipts were written; an empty training shard was not manufactured.

A separate **1.612692-second** continuation smoke reused the same two audited
worlds, captured their prepared first decision, and forced each of their three
legal actions through D300. All six branches used the very same loaded student
network as play, executed 300 transitions, and stored valid survived/null-death/
final-length outcomes. No extra worlds or bank rows were collected. This checks
the real continuation path; fatal endpoint conventions and outcome persistence
also have focused tests. It does not turn these survivor worlds into a test of
late death-window selection.

Commands from the worktree root (the long command below was **not run**):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
T4_ARTIFACTS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4-prep-20261015
T4_STUDENT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t3-mac-20261015/LABELS_s10/final.pth

./venv/bin/python -m research.t4_screen_probe_20261015.screen_probe \
  --student "$T4_STUDENT" --mix scripted --worlds 32 --round-index 60 \
  --path compare --device auto --max-wall-seconds 295 \
  --out "$T4_ARTIFACTS/screen-32.json"

./venv/bin/python -m research.t3_rollout_labels_20261015.relabel_v2 \
  --source /Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/bank \
  --out "$T4_ARTIFACTS/bank-v2" --label-policy prefer-nonboost

./venv/bin/python -m research.t4_expert_iteration_20261015.collect_for_student \
  --student "$T4_STUDENT" --out "$T4_ARTIFACTS/student-validation" \
  --round prep-validation-s10 --worlds 2 --mixes scripted --workers 1 \
  --record-outcomes --label-policy prefer-nonboost

# Future two-hour collection template only; NOT EXECUTED.
# Replace round/output with fresh names for each student/iteration.
./venv/bin/python -m research.t4_expert_iteration_20261015.collect_for_student \
  --student "$T4_STUDENT" --out "$T4_ARTIFACTS/student-round-2" \
  --round student-s10-round-2 --worlds 1400 --max-seconds 7200 \
  --mixes scripted,scripted,scripted,mixed --workers 1 \
  --record-outcomes --label-policy prefer-nonboost
```

## Verification and reproducibility

The following focused suite passed **59 tests**, with **8 existing opt-in
evaluator parity skips**, in 10.27 seconds. The skips require the other dense
workers to have exited; this task did not stop or alter those workers.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
./venv/bin/python -m pytest -q \
  tests/test_t4_bank_tools.py tests/test_t4_screen_probe.py \
  tests/test_t3_collection.py tests/test_t3_death_window.py \
  tests/test_t3_speedup.py tests/test_t3_bank_audit.py \
  tests/test_t3_screen_reroll.py -m 'not slow'

./venv/bin/python -m flake8 \
  research/t4_screen_probe_20261015 research/t4_expert_iteration_20261015 \
  research/t3_rollout_labels_20261015/{bank_engine,collect,death_window,engine,relabel_v2}.py \
  tests/test_t4_bank_tools.py tests/test_t4_screen_probe.py
git diff --check
```

Scoped flake8, Black/isort formatting and whitespace checks passed. Independent
read-only review found no remaining blocking issues after fixing persistence
of `outcome_recorded`, consistent zero mass for dead branches, and generated
namespace exclusions. Additional artifact verification loaded the derived bank
and compared all unchanged arrays to the source.

The compact committed evidence is
[`t4_prep_results_2026-10-15.json`](t4_prep_results_2026-10-15.json).
Full artifacts are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4-prep-20261015/`:
`screen-32.json`, `bank-v2/manifest.json`, `student-validation/manifest.json`,
and `student-continuation-smoke.json`. The evidence JSON records their hashes.

Key identities:

- Student SHA-256: `b6369e9282a171468d39180db79910054309e68957526e1a737467130cc4abd3`.
- Unchanged source manifest SHA-256: `0b216da882b634e79ac640a9970a58157828da96e93ec4dbe68e6924c8141391`.
- Unchanged source shard SHA-256: `96da3b448fc7ebd708304d4fc50d661b5771df23a2cd6a70051a1c5ae32aea67`.
- Derived shard SHA-256: `759b94e6e8e5f336abca542dceed42009c3df88de97c935707e0950d742172b2`.

Implementation commits: `98c7422` (A) and `56fad0e` (B/C), integrated with
remote main in `0a4f2d8`. Each implementation commit was followed by fetch,
merge of `origin/main`, and pushes to both `main` and `t4prep-20261015`.
All task-owned measurement processes exited. No tmux sessions, existing bank
files, trainer/learner defaults, or T2/T3 probes were changed.
