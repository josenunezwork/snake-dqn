# T4a two-sided bank implementation — 2026-10-16

Implements `specs/t4a_two_sided_spec_2026-10-16.md`. The pre-registration and
prepared launch lanes were committed in `12e5920`, merged with then-current
main in `b9ed111`, and pushed to `main` and `t4a-20261016` before the real-bank
CPU integration smoke. The registered 8M experiment and evaluation have not
been launched. No MPS work or existing tmux/process operation was performed.

## Interface and loss

- `--label-bank-counter-mode {inert,single-action}` defaults to `inert`, the
  previous surviving-set loss. The new mode uses the single-action DQfD hinge
  on counters and retains the surviving-set hinge on decisive rows. Both
  sides retain row weights and use `margin * --label-bank-margin-scale`.
- `--label-bank-counter-ratio R` defaults to the stored uniform sampler.
  An explicit ratio samples each side with replacement; the counter count is
  the nearest integer to `bank_batch_rows * R / (R + 1)`. The registered
  128-row bank batch at R=1 has exactly 64 rows of each side. The ratio option
  applies independently of the loss mode. Invalid ratios and unavailable
  requested sides fail before actors start.
- Counter identity comes from B8's stored action surviving in the preserved
  `surviving_original` array, with original-bank compatibility falling back
  to `surviving`. Using the preferred v2 labels would misclassify 35 counters.
- Every active-bank 100k telemetry window adds `bank_agree_decisive` and
  `bank_agree_counter`: raw-six-action argmax in the decisive preferred set
  and equality to B8's stored action on counters, respectively. Missing
  sides produce null. Existing `bank_agree` retains preferred-set membership.
  The reference and fast learner paths use the same definitions.

The default bank-off path retains its existing numeric behavior. The inert
bank path retains its loss and original RNG sampling call; it adds the two
side diagnostics. The trace receipt explicitly distinguishes those additional
fields from baseline-common numeric statistics.

## Counter matching and seed audit

The immutable v2 bank has 10,885 decisive and 5,442 counter rows. Quantiles
are reported as decisive versus counter:

| Variable | p10 | Median | p90 | Absolute standardized mean difference |
|---|---|---|---|---:|
| Length | 768 / 767 | 1,001 / 982 | 1,303 / 1,259 | 0.10004 |
| Frame | 2,483 / 2,382 | 3,385 / 3,331 | 4,501 / 4,362.9 | 0.11102 |
| -max Q | -10.15690 / -9.33151 | -8.01986 / -7.82564 | 1.92312 / 4.88935 | 0.02485 |

No variable exceeds the declared descriptive threshold of absolute standardized
mean difference 0.25 or median difference 0.25 decisive IQR. The conditional
counter-matching flag is therefore not required and was not added. This does
not establish identical conditional state distributions. The Q p95 values
are 17.24560 / 18.93869 and p99 values 43.96055 / 42.60892; all quantiles and
KS distances remain in `t4a_counter_matching_2026-10-16.json`.

Rounds 65 and 66 each contain 96 unique world seeds and have no collisions
within the audited registered/observed scope. Round 65 supplies the initial
96/32/32 mixes and round 66 the one possible 96-world scripted extension.
Refresh the audit before evaluation; another task could use a namespace later.
The evidence binds the v2 shard, original provenance and label namespace.

## Launch artifacts

Prepared, not executed:

```text
/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4a-mac-20261016/lane0.sh
/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4a-mac-20261016/lane1.sh
```

Tracked copies are under `research/t4a_two_sided_20261016/`. Lane 0 owns
10/12/14 and lane 1 owns 11/13. `--check` verifies 19 registered input hashes
and prints the exact commands without training. Both scripts pass `bash -n`
and this preflight, and their parsed common arguments match all five existing
M6 CTRL summaries. The scripts refuse existing outputs/logs, prevent log
clobbering, record exits separately by lane, and stop on a failed seed.

Future experiment scripts specify MPS as in the Mac M6 recipe; validation used
CPU only. Two actors per learner and all existing guard behavior remain in
the prepared experiment commands. The scripts were never invoked without
`--check` by this task.

## Validation evidence

`t4a_default_trace_2026-10-16.json` compares the final source with baseline
`e6331c9`: eight deterministic updates in each of six bank-off/inert-bank,
reference/fast modes. It records online and target tensor hashes, NumPy and
torch RNG hashes, and baseline-common stable statistics at every update.
The reusable harness is `research/t4a_two_sided_20261016/default_trace.py`.

`research/t4a_two_sided_20261016/real_bank_smoke.py` uses the real bound B8_s0,
bank v2, demo margin and held-out demo shard. It requests only 512 transitions
on CPU with one actor/two worlds, and stores a source-bound receipt in its
isolated validation directory. Its timeout handler signals only the fresh
process group it owns; parent/child timeout cleanup has a focused test.
The short smoke does not reach the 100k log interval or warm the 10,000-row
ordinary-label buffer. Deterministic mixed-anchor tests cover the warmed
ordinary/bank combination. Neither the smoke nor these tests are gameplay or
promotion evidence.

Final checks passed on the implementation in `afa9b14`:

```bash
export SNAKE_DQN_DEVICE=cpu CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
./venv/bin/python -m pytest -q \
  tests/test_t4a_two_sided.py tests/test_t4a_smoke_runner.py \
  tests/test_t3_label_bank_trainer.py tests/test_m3c_trainer.py \
  tests/test_m4a_trainer.py tests/test_td_huber_delta.py \
  -m 'not slow' -k 'not mps'
# 41 passed, 2 deselected in 14.66 seconds.

./venv/bin/python -m pytest -q tests/test_perf_learner.py -k 'not mps'
# 13 passed, 1 MPS test deselected in 102.19 seconds (verification lane).

./venv/bin/python -m research.t4a_two_sided_20261016.real_bank_smoke \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4a-mac-20261016/validation/cpu-smoke-formatted
# 512 transitions, 3 updates, halt=null, actor exit [0], 5.252686 seconds.
```

Total: **54 CPU tests passed**, plus all six eight-update default trace modes
and the real-bank smoke. The smoke was repeated in separate create-only
directories after final telemetry and import-formatting changes; the latest
receipt, `t4a_cpu_smoke_2026-10-16.json`, binds the final source bytes. All
three tiny smoke checkpoints had the same SHA-256
`53f75549829df601bc172aa2ccd8b9e7480dd4c500a1d283b2f4327899604fd0`.
No `TWO_SIDED_s*` experiment output was created.

Scoped Black, isort with `--profile black`, flake8, `git diff --check`, both
shell syntax checks and input preflights passed. Independent read-only review
also verified mixed/all-counter/all-decisive loss and gradient formulas,
missing-side denominators, zero rounded bank rows and the v2 removed-boost
counter case. No actionable findings remain. CPU verification does not claim
MPS/CUDA execution coverage, 100k log cadence execution or an 8M research result.
