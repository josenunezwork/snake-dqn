# T3 rollout label bank — 2026-10-15

Development data from simulator rollouts; not training, evaluation, or promotion evidence.
The served policy remains a bare network. No training run or push is authorized in this task.

## Status

Part D is transitioning to the death-window addendum in the owned `tmux t3-run` session.
The target is at least 10,000 decisive rows plus `floor(0.5 * decisive)` counter rows,
with 12 single-threaded CPU workers on AC. Five completed-key replays passed before
relaunch. Completion and final throughput will be recorded after the independent audit.

The preserved dense ceiling is GO: 54/64 episode survival (0.843750), alive-frame
survival 0.942434, versus B8 38/64 (0.593750), alive-frame 0.861206.
This is the dense ceiling of record; the earlier sparse NO_GO remains reported.

## Preserved work and calibrated filter

The interrupted implementation was reviewed and preserved in commit `598c676` before
speed-up changes. The original three-worker collector was terminated; all 50 completed
worlds were retained byte-for-byte: 544 decisive rows and 1,424 counterweight candidates.
`bank/legacy-resume.json` pins their receipts, row shards, and original source identities.
No completed shard was orphaned at interruption.

The preceding filtered collector used the existing dense clock (every fifth decision from first
L>=700, with frame>=2000 and a complete D300 horizon) AND
**`-max(Q) > -9.8897066116333026`** over all six B8 Q values.
The strict threshold retains **517/544 decisive calibration states (95.036765%)**.
It is the next float64 value below the boundary order statistic, so equality at the
float32 boundary is retained.

Legacy Q vectors were recovered with exact rowwise B8 inference on the stored raw bv8
observations; length was recovered from scalar 1 times 1,000. Original shards were not
changed. A no-rollout replay of all 50 worlds matched every one of the 1,968 saved
Q/action/length/frame rows and every death endpoint. Across all 16,072 advisory-legal
dense batches, the fixed threshold would skip 2,233 (**13.893728%**). Replay took
40.991 wall seconds and
408.503 summed worker seconds.

The calibration sample contains b8_s0 and part of b8_s1, and no decisive frozen rows.
Its pooled 95% coverage is a calibration guarantee for those observed decisive rows,
not a generalization guarantee for all five finals or each mix. B8_s0 coverage is
400/424 (94.34%); B8_s1 is 117/120 (97.50%). The threshold stays fixed during collection.

## Death-window mode and preserved prefix

The filtered collector was stopped before changing collection code. All **134** completed
worlds remain byte-for-byte: **1,111 decisive rows and 3,768 raw counter candidates**.
`bank/death-window-resume.json` pins their receipts, raw shards, and source identities.
The original 50-world `legacy-resume.json` and calibration remain unchanged. No orphan shard
was present when the filtered collector stopped.

All five B8 finals play unchanged greedy trajectories on the previously verified fresh
namespace `t3-rollout-labels-dense/v1`, H5000. Each 200-world cycle has 40 worlds per final:
27 scripted, 9 mixed, 4 frozen (67.5%/22.5%/10% overall; scripted:mixed = 75:25 within
non-frozen worlds). Completed keys are skipped only after their hashes and roster validate.
All decisive source rows are retained; final source-world/decision pairs are unique.

The death-window worker keeps at most 60 prepared dense snapshots, every fifth decision
from first L>=700, with frame>=2000 and a full D300 horizon. After B8 dies, it evaluates
alternatives only at snapshots where `0 <= death_decision - decision < 300`, without a
Q filter. B8's actual trajectory supplies its fatal label. The surviving set and decisive
rule are unchanged. Every twentieth window batch also replays B8 and checks the exact
fatal decision; each check has a receipt. Unavoidable states produce no label.

A deterministic reservoir of up to 96 prefiltered snapshots per world supplies counter
candidates. After death, only sampled snapshots outside the strict window are rolled out,
with all legal actions including B8. Every such B8 branch must match the actual trajectory.
Survivor worlds incur play and snapshot costs only. Thus new counters are sampled from
out-of-window decisions in dying worlds; the preserved prefix also retains its original
counter candidates. The fixed Q threshold applies only to these new counter candidates.
The initial 32-snapshot death-window epoch was stopped at a full 400-world boundary:
3,021 decisive rows and 3,910 counter candidates. It added 1,910 decisive rows but only
142 counters, a marginal ratio of 0.0743. This projected a counter shortfall at the decisive
target. The reservoir was increased to 96 without changing the decisive-window rule.
`death-window-retained.json` preserves all 400 completed worlds; the original proof and
its exact source bytes remain under `source-history/`. The separate v2 five-key proof passed **262/262** decisive rows exactly in
233.934 seconds with five single-threaded replay workers; it gates the new epoch. The final audit checks both identities independently.

The final uniform counter sample uses seed 20261015 and contains exactly
`floor(0.5 * decisive_rows)` rows. Collection finishes at a full balanced cycle after both
row targets are satisfied.

Validation replays five completed dying-world keys from the original unfiltered prefix,
including a mixed world. Using that prefix avoids conflating removal of the Q filter with
parity: new unfiltered decisive rows could legitimately be absent from the filtered prefix.
The validation compares exact decisions, surviving sets, action, frame, and observation
bytes; source code, namespace, shards, and checkpoints are hash-bound. **PASS:** all
262 decisive rows matched across five keys, with no missing/extra decisions or byte
mismatches. Serial validation took 315.941 seconds. The collector independently verified
the proof against the frozen source and input files before admission.

## Part C interface (preserved)

The multi-actor trainer has three new flags. All defaults leave the previous numeric path
unchanged; no bank loading, sampling or bank telemetry occurs at fraction zero.

| Flag | Default | Meaning |
|---|---|---|
| `--label-bank PATH` | unset | Directory of sorted `.npz` shards |
| `--label-bank-frac F` | `0` | Rounded share of the configured anchor rows drawn from the bank |
| `--label-bank-margin-scale S` | `1.0` | Multiplier of the run's existing DQfD margin constant |

An active bank requires `--actors >=1 --obs-version bv8 --anchor-mode v8margin`. Bank-only
anchors use `--label-frac 0 --label-bank-frac 1`; `--onpolicy-labels` can be off. Combined
on-policy/bank anchors use `--onpolicy-labels` with fractions summing to one, for example
`--label-frac 0.5 --label-bank-frac 0.5`. The ordinary label portion retains the existing
minimum-buffer-size and retained-ring-row rules, so early batches can have fewer ordinary
anchors while that buffer warms. The bank is loaded and validated before output allocation,
shared memory or actor startup. The synchronous trainer rejects active bank flags.

The shard schema is:

| Array | Shape | Dtype |
|---|---|---|
| `local` | `(N, 7, 31, 31)` | `uint8` |
| `global` | `(N, 4, 37, 37)` | `uint8` |
| `scalars` | `(N, 60)` | finite `float32` |
| `mask` | `(N, 6)` | bool |
| `surviving` | `(N, 6)`, nonempty set per row | bool |
| `action` | `(N,)`, B8's chosen action, 0 through 5 | integer |
| `weight` | `(N,)` | finite, nonnegative `float32` |

With `--obs-pack`, a shard may replace `local` and `global` with `packed`, a `uint8`
`(N, train_layout().nbytes)` array. The loader unpacks using the trainer's layout and verifies
canonical bytes by repacking. Unpacked shards also work with packed training rings. Empty
banks, malformed shapes/dtypes, nonfinite values, empty surviving sets and invalid packed
payloads fail before actors start.

Bank rows receive only the specified surviving-set margin:

```text
row_loss = max_a(Q[a] + margin * scale * (a not in surviving))
           - max_a_in_surviving(Q[a])
bank_loss = mean(weight * row_loss)
```

Both maxima consider the specified six-action sets without applying the observation mask.
The stored B8 `action` is provenance; it is never an additional supervised target. Ordinary
on-policy anchors keep their existing single-action loss. Ordinary and bank means are combined
by their actual row counts. Reference and fast learner paths share these semantics, including
TD Huber delta, the lambda schedule, target synchronization and finite-loss handling.

Every 100k transitions, active-bank logs add `bank_rows_mean`, `bank_margin_loss` and
`bank_agree`. Agreement is the fraction of sampled bank rows whose raw six-action Q argmax
belongs to their surviving set. Bank-only runs safely retain `label_buffer=0`.
`--demo-data` remains required for the run's margin constant and hold-out Q-scale metadata.

## Trainer and hand-back

The existing default-off Part C flags are retained: `--label-bank`, `--label-bank-frac`,
and `--label-bank-margin-scale`. Bank rows use the weighted surviving-set DQfD margin;
fast/reference paths and bank-only anchors are supported. The required trainer and bank
suite passed 46 tests in 99.95 seconds. No research training was launched.

The shell template is `t3_lanes.sh.template` at the artifact root, with a repository copy
under `research/t3_rollout_labels_20261015/`. It provides CTRL/LABELS, five paired seeds
10–14 from B8_s0–4, 8M transitions, the M6 recipe, and literal `{TD_FLAGS}` awaiting T2.
LABELS uses bank-only anchors with the prescribed 2M hold and 0.1 floor. The optional
LABELS_RETURN name is reserved and fails explicitly because return targets were not
implemented by this spec. Both templates pass `bash -n`; neither was executed.

## Artifacts

Root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015`.

- `bank/namespace.json`: original seed allocation, exclusions and materialized rosters.
- `bank/legacy-resume.json`: original 50-world prefix and source identities.
- `bank/death-window-resume.json`: all 134 completed worlds before the mode change.
- `bank/death-window-validation.json`: original five-key replay proof.
- `bank/death-window-validation-v2.json`: 96-reservoir five-key replay proof.
- `bank/death-window-retained.json`, `bank/source-history/`: preserved first two cycles and original proof sources.
- `bank/prefilter.json`, `bank/calibration/`: threshold, enrichment, actual dense-score audit data.
- `bank/worlds/`: immutable completed world shards and receipts.
- `bank/progress.json`: live keys, cycles and decisive count.
- `speedup-calibration.*`, `speedup-bank.*`: previous-stage command and timing receipts.
- `death-window-*`: current-stage command, timing, validation, test and exit receipts.
- `t3_lanes.sh.template`: unexecuted Part E hand-back.
