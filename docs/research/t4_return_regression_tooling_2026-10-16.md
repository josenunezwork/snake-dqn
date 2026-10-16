# T4d return-regression tooling — 2026-10-16

Build and bounded CPU verification of
[`specs/t4d_return_regression_spec_2026-10-16.md`](specs/t4d_return_regression_spec_2026-10-16.md).
This is tooling evidence, not an experiment result, gate, or promotion. The only
learning performed for verification is the explicitly requested tiny CPU smoke.
The future MPS commands below are prepared commands and were not executed.

## Target and reward contract

Each recorded branch takes one forced advisory-legal action, then follows the
supplied frozen B8/student greedily. This is an **expert-rollout target**: it is
not an on-policy return under the learner's changing policy. Every recorded
action supplies a magnitude target, including all legal actions at matched
outside-bank states.

For transitions `t = 0, ..., T-1`, `gamma = 0.998`, `Phi(L) = L/10`, and
`Phi(dead) = 0`, the collector records the raw discounted mass sum:

```text
S = sum_t gamma^t Phi(L_(t+1)) alive_(t+1)
G = kappa (1-gamma) S - Phi(L0)
    + gamma^T Phi(L_T) [survivor]
    - 3 gamma^(T-1) [died]
y = G + gamma^T Q_target(s_T, argmax_mask Q_online(s_T, a)) [survivor]
```

The online argmax and target value bootstrap are detached from autograd. The
extra loss is the weighted mean Huber error for recorded actions, using exactly
the trainer's `--td-huber-delta`, multiplied by `--return-weight`. Ordinary TD
and anchor losses remain present. A dead branch never bootstraps. A survivor
stopped at the branch horizon is a truncation and does bootstrap.

The supported reward assembly is the trainer's mass term plus the simulator's
`gamma*Phi(next)-Phi(previous)` shaping and death value `-3`, with kill scale
zero and no additional reward overlays. The bank binds those semantics and its
collection gamma. **Raw S still depends on gamma**: a different discount cannot
be reconstructed from S alone. Reusing the data with a different gamma therefore
fails validation. Retaining raw S makes the mass contribution auditable.

`tests/test_t4d_formula.py` rolls short branches through `GridBatchSim` and sums
`sim.get_reward() + train_m3c.mass_term(...)` with the discount. It compares that
sum both to the written formula and the production bank target helper, covering
survival, growth, boost cost, death, and three gamma/kappa combinations.

## Stored data and sampling

The return bank is separate from the immutable T3/T4a/T4c label banks. It uses one
row per actually rolled-out action; there are no inferred fatal rows and no
unknown actions encoded as observed deaths. The packed observations use the
existing lossless bv8 pack layout.

| Field | Meaning |
|---|---|
| `state_packed`, `state_scalars` | Initial packed raster planes and 60 bv8 scalars |
| `action` | The forced first action represented by this row |
| `T` | Actual number of transitions, stopping at death or the requested horizon |
| `died` | Whether the branch ended in death |
| `S` | Raw float64 discounted sum of next-state living potential |
| `L0`, `L_T` | Initial length and terminal live length; dead terminal length is zero |
| `terminal_packed`, `terminal_scalars`, `terminal_mask` | Survivor's next prepared decision inputs and mask; dead placeholders are zero |
| `row_weight`, `family` | Effective example weight and selection provenance |
| `manifest_json` | Collection discount, reward semantics, pack layout, and provenance |

The directory manifest binds the shard hashes. Collection uses create-only
outputs and records the checkpoint identity. Outside selection must have length
at least 700, at least 300 observed continuation decisions, and at least 300
decisions to an observed death. Last-300 censored states in surviving worlds are
excluded. Matching uses the T4c decisive-bank length buckets and raw `-max Q`
deciles, with bank-overlap exclusion and no cross-stratum substitution.

Matching quotas are global across worlds. The collector retains at most 61
prepared snapshots at the every-five-decision cadence. After 300 observed
decisions, an eligible unique state can be rolled immediately without retaining
its world clone for the rest of collection. If the hero dies later, that death
is necessarily at least 300 decisions away. The full reference quotas produce
a 1:1 outside/reference-decisive state ratio. Incomplete full matching writes a
failure receipt, no bank, and exits nonzero. Explicit `--allow-partial` requires
a positive `--max-states` cap and labels the resulting bank `PARTIAL_SMOKE`;
it reports every missing stratum without substituting another stratum.

Effective weights are 0.5 for outside states; death-window states within 100
decisions of death have weight 1.0, and those 100–299 decisions away have weight
0.3. Each selected state retains every advisory-legal action, including branches
with no surviving alternative. `source_seed`, `source_decision`, and
`death_distance` provide row-level provenance. Outside death distance is `-1`
because eligibility is established online before any later death is observed.

`--return-frac F` samples `max(1, round(anchor_rows * F))` branch rows per active
update, with replacement, in addition to the ordinary batches. `F=0` is inactive
and consumes no bank RNG. `--return-weight W` multiplies only the extra loss.
Kappa may change because S is stored raw; gamma must match the bank. This first
implementation uses the eager learner and rejects active return regression with
`--learner-fast` or learner compilation. Existing fast/default paths are covered
by the unchanged-path traces. MPS execution itself has not been tested.

## Verification and commands

All verification uses CPU with one numeric thread
per process. The direct trainer smoke uses one learner, one actor, and Python's
resource tracker. Existing tmux sessions and other processes were not modified.

The two-world smoke used explicit fixture seeds `2026101601,2026101602`, with
`--max-states 2 --allow-partial --smoke-seeds ...`. Both heroes died below the
length threshold: 3,084 and 1,339 decisions, zero qualifying snapshots. The
collector correctly returned `NO_QUALIFYING_REAL_BRANCHES`, no shard, and all
unfilled outside-stratum quotas. World execution took **4.810266 s**, with
**0.634567 s** of setup. Explicit fixture seeds make no freshness claim and are
accepted only in a partial smoke of at most two worlds and four selected states.

For the real-format integration check, the same two seeded worlds were rebuilt
and their initial prepared states supplied three legal first actions each. Six
actual D300 branches executed **1,800 transitions**, all surviving. The written
bank loaded through `ReturnBank` with canonical packed observations and hashes.
Its provenance is `FORMAT_SMOKE_INITIAL_STATES`, with `family=2` reserved for
that fixture status. These low-length initial states are **not qualifying
outside/death-window samples**. The integration check did not extend collection
until qualifying states appeared.

The six-branch format check took **2.061970 s**, or **343.661590 s per 1,000
branches** by direct extrapolation. This includes world construction and is a
small CPU sample, not a production throughput estimate. Full matched collection
and MPS timing remain unmeasured.

The tiny direct CPU trainer smoke used that six-row real-format bank and
completed **512 transitions, 3 learner updates, and 192 sampled return rows**.
Mean weighted return loss was **0.7698238691**; halt was null, actor exit code
was zero, and ring overruns were zero. No experiment-length training ran.

Formula verification passed all **12** simulator cases. The final existing
default trace compared six modes over eight updates each against baseline
`91a6120`: online/target tensor hashes, NumPy/Torch RNG hashes, and every stable
statistic matched exactly, with no additional inactive telemetry fields.

The final focused regression suite passed **109 tests**, with four deselected,
in **67.19 s**. The exclusions were the three pre-existing long tests and the
legacy nested CLI smoke already exercised separately. Scoped flake8, Black,
isort using the documented Black profile and 100-character line length, and
`git diff --check` passed. Run the same bounded suite with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
./venv/bin/python -m pytest -q \
  tests/test_t4d_collection.py tests/test_t4d_return_bank.py tests/test_t4d_formula.py \
  tests/test_t3_label_bank_trainer.py tests/test_t4a_two_sided.py \
  tests/test_m3c_trainer.py tests/test_perf_learner.py tests/test_t4_restart_actor.py \
  tests/test_t3_death_window.py tests/test_t4_bank_tools.py tests/test_t4c_data.py \
  -m 'not slow' -k 'not bank_only_cpu_cli_smoke'
```

Artifacts are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4d-build-20261016/`:
`collection-smoke-c/`, `format-smoke/`, `format-smoke-receipt.json`,
`training-smoke/`, `default-final-comparison.json`, and `final-tests.log`.
The committed `t4d_build_validation_2026-10-16.json` binds the relevant artifacts
and source hashes. No checkpoint from this work is a promotion candidate.
Implementation commit: `e34c5ef938f83cf09b41d069ace5249693df212c`; the earlier
simulator-formula test commit is `8f3339c`. Each commit was followed by fetch,
merge of `origin/main`, then pushes to `main` and `t4d-20261016`.

Two earlier freshness preflights were stopped before any world ran: the inherited
Python scanner and an initial two-candidate ripgrep scan spent too long scanning
the large artifact tree. Production T4d planning now uses single-threaded
ripgrep, pinned ignore/config behavior, bounded candidate batches, and per-scan
timeouts; it retains the generated NI/dev-round exclusions and fails closed.
The fixture smoke intentionally bypasses freshness scanning and labels that fact.

Resource deviation: an early existing CLI smoke ran beneath pytest, adding the
pytest parent above the learner, actor and resource tracker; an initial formatter
also used its automatic worker pool. Those invocations exceeded the requested
three-process limit. Subsequent trainer smoke execution was direct, and subsequent
formatting used `--workers 1`. All task-owned runs exited; no unrelated process
or tmux session was signaled.

Exact CPU collection smoke command (executed):

```bash
T4D_ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
./venv/bin/python -m research.t4d_return_regression_20261016.collector \
  --student "$T4D_ART/redesign-m4b-mac-20261013/B8_s0/final.pth" \
  --out "$T4D_ART/t4d-build-20261016/collection-smoke-c" \
  --round build-smoke-20261016-c --worlds 2 --mixes scripted --depth 300 \
  --device cpu --record-returns --outside-returns \
  --match-bank "$T4D_ART/t4-prep-20261015/bank-v2" \
  --max-states 2 --allow-partial --smoke-seeds 2026101601,2026101602
```

Exact CPU trainer smoke command (executed after the format bank was written):

```bash
./venv/bin/python research/redesign_m3c_20261008/train_m3c.py \
  --seed 9 --init "$T4D_ART/redesign-m4b-mac-20261013/B8_s0/final.pth" \
  --demo-data "$T4D_ART/redesign-m2b-20261008/data" \
  --out "$T4D_ART/t4d-build-20261016/training-smoke" \
  --device cpu --actors 1 --worlds 2 --transitions 512 --warmup 64 \
  --replay 512 --label-capacity 512 --segment-frames 500 --frozen-frac 0 --mixed-frac 0 \
  --reward mass --mass-kappa 1 --kill-scale 0 --death-value -3 \
  --gamma-start 0.998 --gamma-end 0.998 --gamma-hold 0 --gamma-anneal-end 1 \
  --eps-end-at 1 --obs-version bv8 --anchor-mode v8margin --td-huber-delta 1 \
  --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1 \
  --onpolicy-labels --label-frac 1 \
  --return-bank "$T4D_ART/t4d-build-20261016/format-smoke" \
  --return-frac 0.5 --return-weight 1 --obs-pack --v61-backend numba --no-guard \
  --ring-dir "$T4D_ART/t4d-build-20261016"
```

The following future commands are **not run**. They use MPS as requested. They
are a build handoff, not a newly ratified experiment. The collection command
fails closed if 128 worlds cannot fill the reference strata; its receipt makes
the deficit explicit. Use a new round and output directory for any later run.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-t4d
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export SNAKE_DQN_DEVICE=mps
T4D_ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
T4D_OUT="$T4D_ART/t4d-return-20261016-production"

./venv/bin/python -m research.t4d_return_regression_20261016.collector \
  --student "$T4D_ART/redesign-m4b-mac-20261013/B8_s0/final.pth" \
  --out "$T4D_OUT/bank" --round production-b8-s0-01 \
  --worlds 128 --mixes scripted,scripted,scripted,mixed --depth 300 \
  --device mps --record-returns --outside-returns \
  --match-bank "$T4D_ART/t4-prep-20261015/bank-v2"

# One control and one return arm, sequential. Change both output paths for a rerun.
# Eager learner; one actor keeps the learner/actor/resource-tracker process count at 3.
for T4D_ARM in CTRL RETURN; do
  T4D_EXTRA=()
  if [[ "$T4D_ARM" == RETURN ]]; then
    T4D_EXTRA=(--return-bank "$T4D_OUT/bank" --return-frac 0.5 --return-weight 1.0)
  fi
  ./venv/bin/python research/redesign_m3c_20261008/train_m3c.py \
    --seed 10 --init "$T4D_ART/redesign-m4b-mac-20261013/B8_s0/final.pth" \
    --demo-data "$T4D_ART/redesign-m2b-20261008/data" \
    --out "$T4D_OUT/${T4D_ARM}_s10" \
    --device mps --actors 1 --worlds 128 --transitions 8000000 \
    --segment-frames 5000 --mixed-frac 0.25 \
    --reward mass --mass-kappa 1.0 --kill-scale 0 --death-value -3 \
    --gamma-start 0.998 --gamma-end 0.998 --gamma-hold 0 --gamma-anneal-end 1 \
    --eps-end-at 1 --obs-version bv8 --anchor-mode v8margin \
    --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1 \
    --onpolicy-labels --label-frac 1 --td-huber-delta 1.0 \
    --obs-pack --v61-backend numba --checkpoint-every 2000000 \
    --ring-dir "$T4D_OUT" "${T4D_EXTRA[@]}"
done
```
