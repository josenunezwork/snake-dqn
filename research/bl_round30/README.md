# Round 30: diagnosis first

Prepared implementation of Astra's **ADOPT_DIAGNOSIS_FIRST**, ruling SHA256
`f69c44863a8331028ade0dff20ee69eb9dc5959a92e8480a7329cb39d2b58d5c`.
Authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/JUDGMENT.md`.
Ruling and QUESTION, all three R30 memos, four R29 origin memos, R27/R28 judgments,
and the nice-15 interim policy were read. Advice is subordinate to the ruling.

**Qualification is source-bound.** The original preparation remained HOLD because
the exclusive launcher refused before pytest. The R30-only ordinary-W6 route
now supports the interim ruling; its immutable current-source receipt determines
Mac qualification. Generated manifests remain blocked until that receipt passes.
No Colab/Mac scientific jobs dispatched. No existing corpus admitted or opened. Claude
owns corpus reuse authority, account reservation, F4 terminal disposition and
dispatch. Codex-V reviews provenance before numerical data. Sol owns only this
new package and its tests. F4's registration/loss/factory, the A100 sprint, P1,
serving and other rounds are unchanged.

## Ordered decisions

1. R30-0: admit metadata and rehearse export → verified ledger → fused reload →
   reducer on synthetic inputs. One protected/foreign/duplicate parent or stale
   binding refuses access. No authorized corpus is HOLD; insufficient support is
   INSUFFICIENT. Salt is `t34-r30-parent-split-20261023-v1`, SHA256(salt + NUL +
   ultimate parent) mod10<2 is validation. All descendants inherit that parent.
   No redraw, replacement, extra collection or F4 reserve access.
2. R30-1: seed3403 original 2,000 batches, original 128+128 life sampler and
   original Adam; one bounded float64 L-BFGS solve of their fixed mean objective.
   Per-batch pair/KL denominators and duplicate occurrences are preserved.
   Independent original-loss value/gradient checks precede coefficient reduction.
   Seed3404 fixes separate training-audit batches, never validation. The solver
   uses at most 500 closure evaluations / 3,600 seconds. A separate directed
   Decimal50 gradient enclosure has at most 600 seconds and supplies numerical
   error in the strong-convexity certificate. Its conservative target is 1e-5,
   which implies the ruling's 1e-5*max(1,F(0)) target. No certificate is INCOMPLETE,
   never evidence of Adam adequacy. Report component gradients/conflicts,
   conditioning, real objective and reloaded Mac fp32 actions separately.
3. R30-2: prepare now; scientific fit/look waits for F4 terminal disposition,
   resolved R30-1 optimizer diagnosis, ≥32 training rescue parents with complete
   ≥2-survivor alternatives, ratio≥.10 and admitted natural archive. Add only
   complete decisive rescue survivor-mass pairs, coefficient one, gap cap four,
   separate pair denominator. Preserve ordinary growth, survival, KL, L2, seed,
   features and exact batch lists. Matched Adam if adequate; otherwise one
   independently verified frozen convergence procedure for both endpoints.
   R9 labels4–8 HOLD pending a new explicit Astra ruling. A flag such as
   TOP_PASSED_DOSE_FAILED never permits automatic continuation.

At the single validation look require positive control regret and ≥10% reduction
in parent-macro survivor-reference regret, life-macro rescue retention≥.70 and
≥control, and successor H=0. Dying selections retain their finite regret; UNKNOWN
is nonretention and cannot become a negative training label or a successful look.
The natural archive must include nonrescue parents, with frozen probabilities.
H counts changed B8-survive→successor-die and changed unknown selected/B8 outcomes.
Report control H, all new harms, unclipped design rates, clipped sensitivity,
ESS, override frequency and coverage. Missing archive is HOLD. Precedence:
INVALID → INCOMPLETE → INSUFFICIENT → HOLD → KILL_SUCCESSOR / GO_OFFLINE.
GO_OFFLINE funds preparation of a separately priced fresh Mac gameplay plan,
zero gameplay here. P7 drift/boost probes, strict/product/serving gates and owner
release approval remain necessary.

## Envelopes (doubled once, reserves included)

| Stage | Forecast | Base | Effective ceiling |
|---|---|---|---|
| R30-0, Mac | short checks | — | .25 aggregate core-hour, 0 units |
| R30-1, L4 | 1 allocated hour /1.54–2 units |1.2h/2.4u|**2.4h/4.8u**|
| R30-2, L4 conditional |2 allocated hours /3.08–4 units|2.4h/4.8u|**4.8h/9.6u**|
| Cloud combined |3h/4.62–6u|3.6h/7.2u|**7.2h/14.4u**|

Combined effective reserve is1.2h/2.4u, included. Additional light Mac work is
≤.25 core-hour per later stage; total≤.75. First local/combined wall/unit cap wins.
No gameplay/label allocation, CPU Colab allocation, A100, H100 or RunPod funding.
Rates/estimates/N/retries are not doubled. Planning envelopes are unmeasured for
real data; tiny synthetic times qualify plumbing only, not a real-corpus forecast.

Each exact-input native canary reserves **20min /2/3u** within its stage (base
10min /1/3u); main caps subtract that liability plus all prior spend. Reserves
within manifests are one sixth of caps. L4 manifest debit is1.54u/h; Claude must
confirm the actual admitted rate and unencumbered balance after other liabilities.
Unknown debit is never zero. Keep P1's priority and the three-account-assignment
limit. Queue R30 if capacity is full; never displace P1 or the A100.

## Metadata and staging

`freeze_inventory` consumes JSON only. See admission.py and its synthetic tests
for a complete fixture. Required inventory fields:

* schema `bl-round30-admission-inventory/v1`, fixed `split_salt`;
* corpus collection authority, source/label/checkpoint SHA256, explicitly reusable
  already-unsealed development authority and `pins_current:true`;
* corpus protection booleans `f4,protected,strict,retention,natural_archive,probe,
  gameplay,stale` explicitly false (these identify excluded sources, not the
  separately permitted historical ordinary archive);
* parents with ultimate_parent_id, rescue_bearing, previously_inspected boolean,
  prior_uses, explicit false `protected,foreign,f4,stale`;
* `partition_sha256.train` and `.validation` binding the frozen NPZ partitions,
  `continuation_sha256` binding H300 full-horizon/dead-zero semantics;
* optional `natural_ordinary_archive` with `explicitly_permitted:true`, validation
  ultimate_parent_ids, sampling_probabilities in(0,1], and a nonrescue parent.

At least32 validation parents/16 rescue-bearing parents are required, without
refill. Approval schema `bl-round30-codex-v-approval/v1` binds `approved:true`,
the canonical inventory_sha256 and allocation_sha256 from freeze_inventory.
Previously inspected data stays exploratory; the split does not make it fresh.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PIN=$(git rev-parse HEAD)
R=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30
B8=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
# First, use the existing quiet-gap route, or supply --factory-start and
# --factory-pid for a live fresh W6 D1024 attempt with its own frozen baseline.
# Both routes refuse baseline/rearming/W12 sanity and never retry automatically.
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 "$PY" -m research.bl_round30.local_check \
  --output "$R/sol-verification/gap-check-01"
# Caller must already be nice>=15. Both routes own the shared serial lock;
# the ordinary route preserves the factory lease and checks its live identities.
# CPU/thread caps
# and 120-second complete deadline. A refusal remains HOLD; no automatic retry.
# Only after Claude's reuse authority and Codex-V metadata approval:
"$PY" -m research.bl_round30.prepare --output "$R/admitted-r301-01" \
  --inventory "$R/authorized-inventory.json" --approval "$R/corpus-approval.json" \
  --train "$R/authorized-train.npz" --b8 "$B8" --commit "$PIN"
```

Training NPZ follows residual.data.validate_rows, which is consumed read-only.
Only training outcomes are uploaded. The Mac validation NPZ is never copied by
the pack builder. All admitted parents must match the frozen partition. Natural
archive row probabilities equal the frozen parent probability in fp32.

`series.json` must bind ruling_sha256, attempt0, verified cumulative stage_hours,
stage_units, combined_hours, combined_units, unencumbered_reservation_verified,
canary_allowance_hours=1/3, canary_allowance_units=2/3. Include setup, exports,
canaries and earlier failures. Completion attempts1/2 additionally require
mechanical_failure_verified, stop_verified, missing_units_verified and failure_kind
transport/interruption/export. Scientific failure or solver cap is never a retry.
No automatic completion launch; original plus at most two mechanical completions.

## Qualified path: Claude dispatches

The synthetic smoke pack is only a consumer rehearsal. **Its input inventory
cannot qualify a real-data job.** After metadata admission, build the exact-input
canary and blocked main template. The same bytes (including series liability)
must be used when binding its successful native receipt to the main job.

```bash
"$PY" -m research.bl_round30.pack --output "$R/r301-native-01" --commit "$PIN" \
  --b8 "$B8" --inputs "$R/admitted-r301-01" --series "$R/series-r301-01.json" \
  --preflight-receipt "$R/sol-verification/gap-check-01/preflight.json"
./tools/ops/dispatch.sh r30-1-canary-01 "$R/r301-native-01/canary-job.json" \
  --broker "$PWD" --cwd "$PWD" --state-dir "$R/broker-r301-canary-01"
# Require native PASS, verified immutable exports, terminal broker receipt and
# confirmed runtime disappearance. Use its actual create-only receipt path:
"$PY" -m research.bl_round30.pack --output "$R/r301-main-01" --commit "$PIN" \
  --b8 "$B8" --inputs "$R/admitted-r301-01" --series "$R/series-r301-01.json" \
  --preflight-receipt "$R/sol-verification/gap-check-01/preflight.json" \
  --canary-receipt "$R/broker-r301-canary-01/ACTUAL-RUN-DIR/receipt.json"
./tools/ops/dispatch.sh r30-1-main-01 "$R/r301-main-01/job.json" \
  --broker "$PWD" --cwd "$PWD" --state-dir "$R/broker-r301-main-01"
```

Run supervisor `validate` before dispatch, using
`research/colab_cli/launch.sh --state-dir FRESH-STATE validate ABS-JOB.json`.
The dispatch wrapper repeats qualification/validation and refreshes live account
assignments under its allocation lock. No supervisor source was changed. Current
pin and complete sparse dependency closure must match; clean-import preflight can
be run through `tools/ops/import_preflight.py` in an admitted quiet check, never a
substantial local build during factory windows. Native canary proof is compulsory. The synthetic canary reloads its actual
immutable fused endpoint and executes a six-action masked fp32 argmax on the
manifest device, so the qualified observer can attribute a real CUDA decision.
A CPU fixture checks the same call and fallback semantics without claiming CUDA.

R30-2 uses identical commands with `--stage r30-2`, new directories/names, and
`prepare --gate ABS-successor-gate.json`. Its sibling diagnosis.json must be the
verified immutable R30-1 Mac result. The gate binds that SHA and corpus allocation,
F4 terminal/R9 disposition, archive/trigger/optimizer status, allow_successor:true,
optimizer and convergence_procedure (both adam or both lbfgs). A material optimizer
requires independent verification of the exact convergence procedure frozen from
R30-1; no tolerance/optimizer selection from validation.

## Mac reducer and resume

`analyze --unit UNIT --table NPZ --binding BINDING --stage r30-1|r30-2 --output RESULT`
reloads each fused checkpoint and executes one CPU row at a time. The source closure
must match binding.source_commit. Binding includes ruling_sha256, source_commit,
unit_sha256, table_sha256, analyzer_sha256, stage, partition(train-audit/validation),
mac_rowwise_admission:true, inventory_path/SHA, approval_path/SHA, corpus_allocation,
allocation(the worker control SHA), salt, and canonical look_path:
`<R>/looks/<corpus_allocation>-<stage>.json`. R30-2 additionally binds successor_gate
and diagnosis_path. The canonical fence is created before outcomes open and cannot
be moved by choosing another result filename. Missing/foreign identity refuses.
Keep remote diagnostics sealed until this entrypoint's allowed look.

Scientific mains use resume-by-default. Both synthetic and exact-input native
canaries use the explicit 20-minute `short-canary` exemption; their synthetic
units cannot enter scientific completion ledgers.

Namespace+frozen-index complete units export atomically with SHA and marker last.
`resume --input-dir ORIGINAL --prior VERIFIED-OUTPUT --output NEW-INPUT/verified-ledger.json`
constructs the ledger from path/size/hash, never unit outcomes. It can recover a
complete unit even if the worker terminal receipt was interrupted; that does not
prove runtime release. Copy original control/metadata inputs unchanged into the
new input directory and use a fresh manifest, output, state and receipt. Only one
eligible complete unit exists per packet. Duplicate/foreign/canary units refuse.
Incomplete fits are not exact snapshots; no partial weights are resumed. A final
fit unit contains parameters, optimizer state, sampler RNG/counters and fixed plan.

Mac work must be nice≥15, CPU only, all numerical pools1, one shared serial check
lock and ≤120s including cleanup under the interim policy. No tests during
baseline/rearming/W12 sanity; no local scientific fits or substantial builds.
Strict, timing, calibration and serving remain exclusive regardless of priority.

## R30-3 deferred design (no execution funded)

Only after offline diagnosis and separate admission: one frozen hash-selected root
per eligible parent,64 parents without replacement, first action{teacher,B8} ×
continuation{B8,frozen student}, paired root/RNG/reacting opponents; H1440 with H300
prefix, at most368,640 branch decisions plus reconstruction. Compare teacher−B8
advantage within each continuation. ≥10-point teacher-first survival loss or≥20%
loss of positive mass advantage supports repair; otherwise close this next
investment. Relative loss is undefined for nonpositive baseline advantage;
<64 parents is INSUFFICIENT. Forecast4 Mac core-hours/hard8 is unreserved and
unfunded here. DAgger, hidden-RNG panels, new losses/heads/architectures and gameplay
are deferred or rejected exactly as the ruling specifies.
