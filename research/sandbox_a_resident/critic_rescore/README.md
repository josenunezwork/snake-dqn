# Shared critic re-score development packet — 2026-10-29

Status: preparation only. No dispatch, no critic refit, no actor update, no release claim.
Standing sandbox development authority and the owner's task authorize this diagnostic.

## Frozen design

Both saved iteration-zero critics are scored on the **same** completed lives from
released B8 s3, masked greedy, no serving veto. This deliberately fixes the rollout
policy instead of using either seed's calibrated stochastic actor. Inputs are the
lossless, original-byte-verified XZ snapshots from seed 1 attempt-11 and seed 2
attempt-04; only their separate critic states are loaded. Source digests, PopArt
state, and all encoders are retained; no optimizer is constructed.

Retain the admitted `a100-dev/sandbox-a-train01` root function, with **new index
ranges**: holdout `[9000000000, 9000003072)` and independent comparator bank
`[9100000000, 9100000096)`. These are distinct from training, calibration, critic
prep/holdout, proxy and nomination allocations in both seed packets (seed 1 offset 0, seed 2 offset 2000000000).
Their full source-manifest allocation ranges and metadata hashes are retained in
the seed audit. Mac admission
checks every new root's low-63-bit simulator alias against the frozen sandbox
exclusion inventory (including protected-bank exclusions), and refuses duplicates
and zero. This does not inventory opaque or unregistered external runs.

Collect E96 waves with 32 worlds of each existing scripted/frozen/mixed roster,
using the resident simulator-D natural reset, opponent and observation adapters.
Every life starts at its natural initial state and ends at death or H5000. There
is no injection, respawn credit or partial-life bootstrap. Target at each pre-action
row is the gamma-one suffix sum of the task's post-action logical mass / 1000,
including that row's transition. Critics receive the existing `(5000-age)/5000`
clock. Save length, target, clock and both raw predictions in each completed wave.
No observations need be retained because both critics score the identical state
before the action. Early rows need no critic forward; predictions there are zero
and excluded by the length filter in every critic statistic.

A late life is a completed life with at least one pre-action length >=800 row.
After each whole wave, stop when eight late lives exist in every mix. Select the
first eight per mix in fixed wave/slot order, giving **exactly 24** late lives in
the primary score bank. Disclose all collected support and selected IDs. The
support-based endpoint never uses critic values. Retain additional completed lives
for the old-gate and scarcity diagnostics, without including extra late lives in
the balanced primary estimate. The finite guard is 32 waves (base 16, effective
32 under the owner's 2x rule), plus one E96 independent comparator wave. Failure
to obtain the endpoint is `INCOMPLETE`, never a quality verdict.

## Mac numerical look

The runtime publishes support and mechanical progress only. The Mac aggregator
verifies the union of complete exported units, rejects duplicate/foreign life and
unit identities, then takes the numerical look:

- Late row-pooled EV = `1 - Var(prediction-target)/Var(target)`, population variance.
- Raw MSE and the same-bank **late-row-mean** baseline MSE (`Var(target)`), plus ratio.
- Each life's within-life late EV and support; single-row/zero-variance lives are
  explicitly undefined for the mean-per-life statistic.
- 2,000 deterministic bootstrap replicates, sampling whole lives with replacement
  independently within each mix. Percentile 95% intervals for row-pooled EV and
  mean per-life EV, and the same diagnostics separately for each mix.
- Old `critic_quality` function on **all completed holdout lives**, including its
  >=160 late-row, >=40 complete-life, >=8 lives/mix, EV>=.10 and MSE<baseline tests.
  **Original prep baseline coefficients were not saved**. Refit its time+length
  OLS comparator only on the independent E96 fresh bank, freeze it, and score it
  on the holdout. This is an explicitly labelled new-bank recomputation of the
  statistic, not reconstruction of either original quality verdict.
- 2,000 empirical resamples of 160 total rows and separately 160 late rows,
  uniformly without replacement. Count distinct late-life IDs and report the
  frequency below three plus Monte Carlo SE. Separately permute complete-life
  arrival order and apply the original >=160 late rows / >=40 lives / >=8 per-mix
  stopping rule. These are conditional finite-bank diagnostics, not estimates
  from independent new holdouts or binomial scientific confidence intervals.

## Resume and execution boundaries

Unit identity = namespace + phase + frozen wave index + seed offset. Filenames
include the full recipe/allocation fingerprint, allowing outcome-blind ledger
admission to refuse foreign allocations without opening numerical payloads. One compressed
JSON payload per full wave; deterministic gzip, atomic immutable publication, SHA
sidecar, completion marker last. An incomplete wave is not a unit. The first
attempt uploads an empty hash-bound ledger. `ledger` builds the Mac-verified ledger
using path/size/hash only. A completion attempt uploads those units under `prior/`
and supplies the ledger; runtime skips verified units and refuses foreign
namespace/allocation or corrupt payloads. An interrupted partial wave may be
recomputed, but a verified completed wave never is.

One original plus at most two outcome-blind completion attempts, only after a
verified mechanical failure, verified runtime stop, and missing units; fresh
manifest/output/receipt each time. No automatic completion retries. Keep the same
control/source/input allocation across attempts. The registered Mac aggregator
alone reads quality. No serving or promotion change follows this diagnostic.

H100-first / A100 fallback uses the transported preference mechanism. Native
architecture targets cover 8.0 and 9.0. Forecast 240 minutes; base runaway timeout
240 minutes, effective 480 minutes. Forecast unit amounts are informational:
H100 48/96 base/effective at provisional 12 u/h; A100 32/64 at 8 u/h. No unit-budget
stop. H100 fallback needs the existing transport's independent no-assignment proof;
there is no job retry for model quality or observed support. Mac tiny timing is
mechanical evidence, not a GPU throughput measurement. The full 32-wave guard is a
worst-case collection envelope; actual stop is only the registered support endpoint.

The packet retains TRAIN-01 sandbox lineage. A separate read-only critic-snapshot
input kind admits only the two owner-named original/derivative SHA pairs. Seed 2
has a failed-preparation snapshot without an actor export; this diagnostic kind
does not loosen the full-trajectory continuation requirements. The narrow ops extension routes
only `critic_rescore/job.sh` to this package's own offline preflight. Transport
implementation and training/prep/gate/serving algorithms remain untouched.

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
# After verified exports; paths include baseline-0000 and each holdout unit once:
$PY -m research.sandbox_a_resident.critic_rescore.run ledger \
  --inputs /absolute/committed-packet \
  --units /absolute/verified/rescore/*-{baseline,holdout}-*.json.gz \
  --output /absolute/fresh/verified-ledger.json
$PY -m research.sandbox_a_resident.critic_rescore.run analyse \
  --inputs /absolute/committed-packet \
  --units /absolute/verified/rescore/*-{baseline,holdout}-*.json.gz \
  --output /absolute/fresh/analysis.json
```
