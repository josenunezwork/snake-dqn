# T4c implementation and validation — 2026-10-16

Implements `specs/t4c_tightened_spec_2026-10-16.md`. This task prepares the
TWO_SIDED_T experiment; it does not start registered training or evaluation,
change serving, or authorize promotion. The pre-registration was committed and
pushed to main and t4c-20261016 in `dd96474` before real-bank smoke execution.

## Default-off trainer behavior

`--label-bank-outside PATH` enables a separately loaded outside-counter bank.
It requires an active primary bank, `--label-bank-counter-mode single-action`,
and `--label-bank-counter-ratio 1`. The separate
`--label-bank-outside-margin-scale` defaults to 1.0 and the registered recipe
explicitly supplies 1.0: base margin 0.06319992542266846 Q. Primary decisive
and in-window counter margin remains base ×4.75 = 0.30019964575767517 Q.

The derived v3 bank attaches `row_weight` and `row_family` to the immutable v2
arrays. Effective weight is the original `weight` times `row_weight`. The original
weights are all 1. Decisive distance weights are 1.0 below 100 decisions and 0.3
for the remaining stored 100–299 decisions before death. All decisive rows are
retained. In-window counters retain weight 1; outside counters have weight 0.5.
The data document and manifests specify exact provenance and matching.

At a full anchor batch, the primary bank still supplies 64 decisive and 64
in-window counter rows; 64 outside counters are appended. The ordinary anchor
batch remains unchanged. The outside hinge contributes an additional weighted
sum normalized by the original primary-bank count; the ordinary/primary mixture
uses its original denominator. Adding outside rows therefore preserves the
existing ordinary and primary terms instead of weakening them by a larger
batch denominator. Sampling remains with replacement during training; the
stored outside bank itself uses unique observations without replacement.

Decisive rows retain the preferred surviving-set hinge. Both counter families
use the single-action hinge toward their stored B8 action over all six actions.
Agreement uses raw six-action argmax and independent family denominators:
`bank_agree_decisive`, `bank_agree_counter` (in-window), and
`bank_agree_outside`. Missing families produce null at telemetry flush. Existing
bank loss, agreement and row telemetry remain available. The same loss and
family definitions serve reference and fast learner paths.

## Verified default path and tests

Baseline commit `09162105972a1bcca2f5f7dfc45ef32fa3d5ea7d` was materialized
with `git archive`, without changing another checkout. The existing
`research/t4a_two_sided_20261016/default_trace.py` harness ran eight synthetic
CPU updates in each of six bank-off/inert-bank reference/fast modes. The
[original trace receipt](t4c_default_trace_2026-10-16.json) records exact equality of
online/target tensor hashes, NumPy/Torch RNG, and every old stable statistic.
No new telemetry fields appear in those off-path traces. This is bounded
synthetic verification, not an experiment run.

After the strict shard-set guard and metadata serialization fix were added, the
same six-mode/eight-update comparison was repeated on the final trainer sources
and again passed exactly; see the
[final trace receipt](t4c_default_trace_final_2026-10-16.json). The intermediate
strict-shard receipt is retained separately.

CPU tests used `SNAKE_DQN_DEVICE=cpu`, empty `CUDA_VISIBLE_DEVICES`, and
OMP/OpenBLAS/MKL/VECLIB/Numba thread limits of one, with no MPS invocation.

- 26 passed: `test_t4a_two_sided.py`, `test_t3_label_bank_trainer.py`, including
  nonzero outside hinge/gradient, independent scale and weight, family
  denominators, no dilution, and mixed ordinary/on-policy fast/reference checks.
  Tests also reject missing/unlisted shards and verify JSON-safe outside-path metadata.
- 33 passed, 4 deselected: `test_m3c_trainer.py`, `test_m4a_trainer.py`,
  `test_td_huber_delta.py`, `test_perf_learner.py`, with `-m 'not slow' -k 'not mps'`.
- 30 passed across the new T4c suites: 12 data, 6 diagnostic, 6 launch and 6
  freshness tests. The diagnostic artifact regression ran against the local
  saved evidence (it skips when those local artifacts are unavailable elsewhere).

Scoped Black, isort with Black profile/100-column limit, flake8 and
`git diff --check` passed. These checks exclude MPS and do not run the full
repository or any registered 8M experiment.

## Prepared launch and operating boundary

Tracked scripts are `research/t4c_20261016/lane0.sh` and `lane1.sh`. Their
standalone artifact copies run from this T4c worktree. Lane 0 owns seeds
10/12/14; lane 1 owns 11/13. Both are CPU-only, with two single-thread actors
per learner, and share a lock acquired before any Python hash preflight.
The second lane waits; `--check` refuses an occupied lock without starting
Python. This keeps learner, actors and Python resource tracker within four
processes. No other owner's lock, process or tmux session is controlled.

Launch bindings cover historical B8/CTRL identities, v2, both new bank manifests
and shards, the preregistration, changed trainer source and both scripts.
Existing seed output/log paths are refused; failures stop a lane. Neither
script is executed without `--check` during this task.

Both actual artifact scripts passed `--check`, validating all 29 bound files,
and are byte-identical to their tracked copies. Syntax checks passed. Six
launch tests cover lock ownership and extra/missing shard rejection. Strict
shard-set equality is enforced in the T4c loader, binder, smoke and both lane
preflights. The initial unrun binding is retained as
`launch_inputs_before_shard_guard.json`; the subsequent failed integration
smoke's binding is `launch_inputs_before_summary_fix.json`. `launch_inputs.json`
supersedes both and binds the final sources. No registered experiment was run
from any binding.

The seed audit scans repository research and all artifact-root JSON/JSONL,
including root-level files. It excludes only its own documentary JSON receipt,
never an entire output directory. Tests ensure actual own-study probe rows,
unusually named campaign directories, and root-level records are detected.
The audit makes no claim about unrecorded experiments, arbitrary binary files
or external systems, and must be refreshed before future evaluation.

The expanded scan also preserves malformed historical files as explicit evidence,
not silently skipped inputs. For them, a separate check records exact bytes,
SHA-256, size and parse error, and requires ASCII without NUL/Unicode escapes,
no complete reserved seed literal, and no explicit reserved-round metadata.
This establishes absence of those IDs in retained bytes only; missing or truncated
records cannot establish what unrecorded worlds were used. The limitation remains
part of the freshness receipt. No old artifact is repaired or overwritten.

The full freshness audit passed with no reserved-world collisions in its
recorded scope. It separately recorded and checked 67 malformed historical
files (62 empty, four truncated and one intentional invalid fixture), preserving
the retained-byte limitation above. Six audit regression tests passed.

## Published data and diagnostic evidence

The [data report](t4c_data_2026-10-16.md) records immutable manifests for the
16,327-row v3 bank and the 10,885-row outside bank. The outside rows match every
one of the 20 required strata, include 5,482 CTRL states and 5,403 B8 states,
and have no exact-observation overlap with v2. Both the data lane's real-array
audit and the root's [independent audit](t4c_data_audit_2026-10-16.json) passed.
The candidate writer now refuses existing payloads or receipts; original
capture and assembly sources remain archived under their original hashes.

The [drift diagnostic](t4c_drift_diagnostic_2026-10-16.md) reports separate
round-62 and round-65 same-init baselines and recomputes consequential changes
for all three historical arms using the original readout (c) standardization.
These diagnostic observations do not change the registered decision rule.

## Real-bank integration smoke

The [passing smoke receipt](t4c_cpu_smoke_2026-10-16.json) binds the final trainer
sources, real v3/outside shards, input ledger, summary and checkpoint. It
completed **512 transitions / 3 updates**, returned zero, reported no halt and
actor exit code zero. The CPU trainer's measured duration was 3.65 seconds.
It used one actor/two worlds and isolated validation output. The runner,
learner, actor and resource tracker fit the four-process budget, with other
numerical work paused throughout.

Earlier validation outputs are preserved: `validation/cpu-smoke` reached 512
transitions but failed writing summary JSON because the outside path remained
a `Path` in `m3c_flags`. A regression test and path serialization fix address
that defect. `validation/cpu-smoke-summary-fix` completed the trainer but exposed
a smoke-wrapper assertion that expected a default-valued flag in the sparse
non-default metadata. The wrapper now checks the complete CLI args and accepts
the omitted default. `validation/cpu-smoke-final` passed completely. These are
bounded integration attempts, not experiment evidence. The final checkpoint
is not a trained candidate or a promotion result.
