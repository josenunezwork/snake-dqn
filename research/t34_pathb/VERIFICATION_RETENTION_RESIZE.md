# T34 retention pre-data resize verification — 2026-10-20

Implementation of [the operator ruling](../../docs/research/specs/2026-10-20_t34_retention_resize.md).
No retention efficacy, reused-T3 pre-check or timing gameplay was run. Package
status remains **DRAFT_PENDING_OPERATOR_RATIFICATION**, unauthorized.

Worktree `/Users/josenunez/Projects/ml/snake-dqn-retention-resize`, branch
`t34-retention-resize-20261020`, clean initial revision
`655a6686`. Frozen executable/source-inventory revision, published on origin/main:
`b7ac811ecf415a1d0a0dff5e8cc952be97f6d71a` (implementation `9a613c65`, followed by
concurrent main integration). Subsequent JSON/document-only commits preserve it.

## Implemented design

- 48 paired terminal H5000 worlds per mix, 144 total; 32 complete paired
  continuous-respawn mixed Watch6 sessions; one disjoint timing seed.
- Caps 40 elapsed h / 60 CPU-h / $0; first cap stops INCONCLUSIVE.
- Runner, analyzer, profile, draft registration and OC agree on N and caps.
  Decision thresholds, controller identity, comparator, one-look discipline,
  original T3 source bank and the separate operator ratification remain fixed.
- All 96 original terminal rows, their rosters, the original 16 product rows and
  timing row are byte-equivalent JSON values to baseline `655a6686`.
  The 48 additional terminal rows follow the original block; 16 product rows
  append to their original prefix. No prior seed was removed or replaced.
- `registration_retention_v2 resize` derives the next namespace indices through
  the frozen allocator and audits all 64 additions in one scan before writing
  create-only output candidates. A collision rejects the entire extension.
  The scan returned zero hits across docs/research/src/scripts/artifact text and
  seven concrete historical FRP exclusion inventories, including the retained
  113-seed retention bank and Round 0 draft bank in the research scan. Audit
  metadata records the prior bank/audit hashes, embeds prior audit evidence,
  and explicitly binds `research/t34_round0/BANK_ROUND0.json` by SHA-256
  `76863477c4e47bee40c19980ca0dd01983f1bf49ea473c8b2fd4018958d0c2ae`.
  There are 177 unique retained-plus-added seeds, no collisions. Prior hashes
  resolve in Git at `resize.previous_commit`; they describe historical bytes.
- Minimum paced exposure if all terminal heroes survive:
  `(144+32)*2*5000/24/3600 = 20.370 h`, plus timing/warmup (~.361 h), pre-check,
  setup and descriptive unpaced work. These caps are limits, not runtime promises.

## Development-only OC

PASS, 20,000 replicates with seed 3420261020. Maximum observed protected-regression
joint pass rate **0.0262**, below the separately recorded **0.06** limit, zero
failing protected cells. The complete receipt retains all original proxy/stress
models and the protected scenarios, 120 Path B power cells and four product cells.

| Historical Watch baseline | 32-session product power, 20% fewer deaths | 40% fewer deaths |
|---|---:|---:|
| T15 mixed Watch6 | 0.93755 | 0.99950 |
| T13b mixed Watch6 | 0.96850 | 0.99990 |

B8 s3 T3-shaped joint GO planning probabilities, using terminal N48/mix:

| Mass gain / scripted no-death gain | T15, 20% fewer deaths | T13b, 20% | T15, 40% | T13b, 40% |
|---|---:|---:|---:|---:|
| +40 / +0.1 | 0.43740 | 0.44960 | 0.46420 | 0.46450 |
| +60 / +0.1 | 0.72770 | 0.75015 | 0.77430 | 0.77465 |
| +92 / +0.1 | 0.77920 | 0.80385 | 0.82950 | 0.82985 |
| +40 / +0.25 | 0.45065 | 0.46305 | 0.47825 | 0.47860 |
| +60 / +0.25 | 0.74905 | 0.77255 | 0.79730 | 0.79770 |
| +92 / +0.25 | 0.80360 | 0.82905 | 0.85545 | 0.85585 |

At T3-like +92 / +.25 and 20% death reduction, joint GO power is
**0.80360 / 0.82905** (T15 / T13b); both exceed .5, so
`t3_like_power_review_required.required` is **false**. N remains 48/32 with no
further resize. This is a prospective planning result, not retention efficacy.

All joint powers assume compliant timing and boost plus unchanged product growth.
T3 uses 32 shared physical seeds across two checkpoints, each source arm resampled
separately to 48 draws; scripted shapes proxy frozen/mixed variability. Product
power uses separate T15/T13b baselines with optimistic binomial death thinning.
The full receipt carries Wilson intervals, proxy limitations and +40 mass power.
Its 24/48 sequential cost illustration is descriptive only; fixed N is retained.

## Verification and exact commands

The final full retention selection passed **106 tests in 11.72 s**, no skips or
deselections: the required original 103 tests plus three new allocation tests.
The first pass exposed three old-N synthetic fixture expectations; expected
stratum df and test-only uncertainty spread were adjusted to exercise the same
registered boundary at N48. No decision threshold changed. The final Black check
covered 158 Python files; isort and flake8 pass. Read-only independent source and generated-artifact
review found no actionable issue; preserved rows, 177 seeds, audit provenance,
90 development inputs and all 604 rate/count rows were independently checked. Draft validation reports
`{"valid": true, "authorized": false, "mode": "draft"}`.

All commands run on the Mac with numerical pools limited to one thread. No
parallel worlds or efficacy worker was started; no service/tmux action occurred.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-resize
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-resize-20261020
# Already executed against original N32/16 bank; do not repeat on resized inputs.
"$PY" -m research.t34_pathb.registration_retention_v2 resize \
  --out-bank "$E/bank-01.json" --out-audit "$E/seed-audit-01.json"
# Create-only audited candidates were then copied to the authorized draft paths.
# Implementation committed and published before OC and source-pinned draft build.
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-02.json" --state "$E/oc-02-state.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# PASS receipt copied to RETENTION_OC_V2.json, then:
"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-02.json" --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
# Draft candidate copied to REGISTRATION_RETENTION_V2.json, then:
"$PY" -m research.t34_pathb.registration_retention_v2 validate
"$PY" -m pytest -q \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py \
  tests/test_t34_pathb_retention_v2.py \
  tests/test_t34_pathb_analysis_v2.py \
  tests/test_t34_pathb_timing_v2.py \
  tests/test_t34_retention_fixes_analyzer.py \
  tests/test_t34_retention_fixes_power.py \
  tests/test_t34_retention_fixes_runner.py
CHECKS=(src web/backend \
  research/t34_pathb/{analyze_v2,registration_retention_v2,retention_v2,retention_oc_v2}.py \
  tests/test_t34_pathb_{analysis_v2,registration_retention_v2,retention_v2}.py \
  tests/test_t34_retention_fixes_{analyzer,power}.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
git diff --check
```

A separate read-only comparison to `git show 655a6686:research/t34_pathb/RETENTION_BANK_V2.json`
confirmed every retained row/roster and timing identity. The allocation audit,
OC and draft build used fresh create-only artifact paths; the original draft
versions remain in Git. No simulation resume or second resize was needed. The initial source-frozen
OC (`oc-01.json`) passed; concurrent main additions changed the full source
inventory, so development-only OC was repeated at the same N/seed/inputs into
fresh `oc-02` paths after integration. All 604 rate/count rows were identical.
Both source-bound receipts and both draft candidates are preserved. This was
a source-binding refresh, not a gameplay retry or response to scientific data.

## Package SHA-256 bindings

| File under research/t34_pathb | SHA-256 |
|---|---|
| RETENTION_BANK_V2.json | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| SEED_AUDIT_RETENTION_V2.json | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |
| RETENTION_PROFILE_V2.json | `7cc3361709ba998e40b9860f8c04ec503544c7b8412608dbda529de1e7f260ea` |
| PREREGISTRATION_RETENTION_V2.md | `250153225614b424b90a6458597151fb26419700bebf8e25655df9a65134200d` |
| RETENTION_OC_V2.json | `4bb241e72d2ec245e55e9a6b11a9f15f55fe6ff5db3a7e0b373afef4b4489daf` |
| REGISTRATION_RETENTION_V2.json | `849ea2d8ca305c336dfbfec60286bb1fdfd00d88621a4dd3939768ab6b946cf3` |

## Remaining operator work

The draft validator passes without authorization. Actual final-source trace
receipt, operator ratification, published ready registration, quiet-host timing,
reused-T3 pre-check and retention efficacy remain separate operator steps in
[README_V2](README_V2.md#retention-registration-package). None ran here; Phase 0
was left undisturbed. No cloud spend, training, serving change or release.

Only the explicitly requested pre-data draft preregistration/profile/bank/audit/
OC/registration were refreshed. Historical and ratified governance documents,
create-only gameplay receipts, released checkpoints, serving pins and all
services/tmux sessions were left untouched.
