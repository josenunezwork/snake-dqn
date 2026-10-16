# T4 update cadence and diagnostic checkpoint curve — 2026-10-15

Implements [the T4 preparation spec](specs/t4_update_every_and_curve_spec_2026-10-15.md).
This is development diagnosis only. **The curve never changes the T3 screen decision**,
uses no T3 screen worlds (rounds 62/63), and grants no promotion authority.

## Part 1: update cadence

`--update-every M` is implemented and defaults to 1;
one learner update consumes credit for `M * 128 * batch_scale` ingested agent transitions.
Batch size is unchanged. Target synchronization retains the existing `round(1000 / batch_scale)`
learner-update interval (500 at K=2), and actor publication retains its existing K-adjusted
update interval. Neither interval is multiplied by M; increasing M therefore increases
transitions between synchronization/publication events. Nondefault M is written to log rows
and checkpoint/summary nondefault options. M < 1 and nondefault M with zero actors are rejected.

The A/B training is deferred. No A/B or study training was run by this task; the required small CPU integration test ran as validation.
The complete one-seed B8_s0, 8M, LABELS_KEEPV8_LO command appears below for a future quiet Mac.

Validation: **40 passed, 1 deselected in 118.88 seconds** across `test_update_every.py` and all
four specified test files. Only the hardcoded MPS test
`test_perf_learner.py::test_mps_fast_and_fp16_stay_finite_and_close` was deselected; the CPU
multi-actor integration test passed. The new cadence tests exercise reference and fast
learner updates, default intervals, same-K batch/target/publication invariance, CLI validation,
and nondefault metadata. All numerical tests used CPU and single-thread limits.

[Hash-bound validation receipt](../../research/t4_curve_20261015/results/update_every_validation.json):
clean baseline commit `48ab33fd7d64890fc7972dee14826be932faa8b2` versus the modified source;
both default traces have SHA256 `597219dea8a97ac05dad6dd1f041da6f152c3fce0d39af43e6d3a8484746e6ab`.
All four b/bv8 × reference/fast cases preserve exact network tensors, loss/counter traces,
and RNG state. An independent reviewer reproduced exact equality against the same baseline.
This establishes deterministic default learner parity, not asynchronous actor scheduling identity.

## Part 2: diagnostic protocol (declared before outcomes)

- 15 finished T3 runs: LABELS, LABELS_KEEPV8, LABELS_KEEPV8_LO, seeds 10–14.
- Checkpoints: `ckpt_2000k.pth`, `ckpt_4000k.pth`, `ckpt_6000k.pth`, `final.pth`.
- Reference: five M6 CTRL seeds 10–14, `ckpt_4000k.pth` and `final.pth`.
- Each checkpoint: the same 32 scripted worlds, round 64, horizon 5000, no veto,
  T2 probe path, CPU rowwise, advisory mask, numba vector61, fast anchors.
- `tmux t3-curve` owns at most four single-thread CPU probe children. It reads training
  status and artifacts only. No calls target the protected sessions or existing processes.
- Only terminal successful runs with complete summaries are eligible. Probe failures stop
  the watcher without automatic retry. Saved rows are validated against checkpoint SHA256,
  settings, exact world identities, and survival accounting.
- Every arm/checkpoint uses equal-weight paired training-seed means versus **CTRL final**
  for the common-reference curve. Also report arm4M minus CTRL4M, and CTRL4M minus CTRL8M.
  No CTRL2M/6M measurements are inferred.
- Episode survival is the fraction alive at H5000; alive-frame survival is alive frames / 5000;
  mass is the full-horizon mass integral. Seed-paired differences are computed on identical worlds.
- Primary 4M retention is `(arm4M - CTRL8M) / (arm8M - CTRL8M)` in episode survival.
  A positive 8M effect and at least 80% retained at 4M are required; nonpositive 8M effects
  cannot establish a shorter-cycle benefit. Alive-frame and mass retention are secondary.
  The matched4M comparison is reported alongside this fixed-reference calculation.

The pre-first-probe [seed audit](../../research/t4_curve_20261015/results/seed_bank_audit.json)
passed: round64 seed digest `374f99d569e6b4d6ba75b8a6d84347ca6f62a3a82781e475efbce45a14804102`,
32 unique seeds, zero collisions. Scope: registered NI/distillation worlds, reserved 62/63,
observed JSON probe and repository research receipts, and manifest-bound T3 label namespaces.
This does not claim knowledge of unrecorded experiments or arbitrary binary files.

Artifacts: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t3-curve-probes-20261015/`.
Reproducible watcher: `research/t4_curve_20261015/watch.py`; seed audit: `banks.py`.
All 70 valid probe rows and all 15 T3 runs are complete; results and evidence are below.

## Completed curve results

**Complete: 70/70 validated probes, all 15/15 T3 runs, all five M6 controls.**
Every training summary is unhalted with two successful actor exits; all T3 status receipts are exit 0.
The watcher launched exactly 70 probes, with maximum concurrency four, zero failures, zero retries,
and zero remaining owned probe children. Final validation rechecked every row against its checkpoint
SHA256, successful summary, exact round-64 world set, settings, and survival accounting.

The following deltas use the same **CTRL final (8M)** at every candidate checkpoint,
paired by training seed 10–14 on the same 32 worlds. Values are mean (SE across five seed-paired deltas);
survival is a fraction, and mass is the full-horizon mass integral.

| Arm | Length | Episode survival delta | Alive-frame survival delta | Mass delta |
|---|---:|---:|---:|---:|
| LABELS | 2M | -0.0187 (0.0437) | -0.0112 (0.0314) | -167.07 (24.67) |
| LABELS | 4M | -0.0625 (0.0431) | -0.0416 (0.0306) | -136.11 (29.48) |
| LABELS | 6M | +0.0250 (0.0334) | +0.0200 (0.0212) | -85.13 (25.53) |
| LABELS | 8M | -0.0187 (0.0403) | -0.0190 (0.0177) | -95.84 (20.70) |
| LABELS_KEEPV8 | 2M | -0.0437 (0.0480) | -0.0237 (0.0261) | -149.59 (25.49) |
| LABELS_KEEPV8 | 4M | -0.0875 (0.0348) | -0.0049 (0.0350) | -92.27 (36.24) |
| LABELS_KEEPV8 | 6M | +0.0125 (0.0254) | +0.0092 (0.0101) | -76.54 (12.50) |
| LABELS_KEEPV8 | 8M | +0.0063 (0.0362) | +0.0222 (0.0223) | -55.09 (25.90) |
| LABELS_KEEPV8_LO | 2M | -0.1562 (0.0261) | -0.0525 (0.0330) | -150.41 (32.24) |
| LABELS_KEEPV8_LO | 4M | -0.0688 (0.0643) | +0.0021 (0.0326) | -81.61 (33.39) |
| LABELS_KEEPV8_LO | 6M | -0.0250 (0.0536) | +0.0237 (0.0277) | -62.56 (23.85) |
| LABELS_KEEPV8_LO | 8M | -0.0875 (0.0554) | -0.0249 (0.0332) | -93.24 (28.59) |

### Controls and matched-duration 4M comparison

| CTRL checkpoint | Episode survival | Alive-frame survival | Mass |
|---|---:|---:|---:|
| 4M | 0.44375 | 0.77627 | 521.10 |
| 8M | 0.58750 | 0.82165 | 578.43 |

CTRL4M minus CTRL8M is **−0.14375 episode survival, −0.04539 alive-frame survival,
and −57.33 mass**. The control changes materially with duration, so the reference choice matters.

| Arm at 4M vs CTRL4M | Episode survival delta | Alive-frame survival delta | Mass delta |
|---|---:|---:|---:|
| LABELS | +0.0813 (0.0606) | +0.0038 (0.0311) | -78.78 (28.21) |
| LABELS_KEEPV8 | +0.0563 (0.0182) | +0.0405 (0.0159) | -34.94 (17.03) |
| LABELS_KEEPV8_LO | +0.0750 (0.0581) | +0.0475 (0.0167) | -24.28 (20.22) |

### 4M verdict

**No arm demonstrates at least 80% retention of a positive 8M episode-survival effect under
the declared fixed-CTRL8M comparison.**

- LABELS: 8M delta is −0.01875; there is no positive 8M effect to retain.
- LABELS_KEEPV8: 4M delta is −0.08750 versus +0.00625 at 8M. Retention is −14.0×,
  below 0.8×; alive-frame retention is −0.2195×. The 8M episode effect is small
  relative to its between-seed SE of 0.0362.
- LABELS_KEEPV8_LO: 8M delta is −0.08750; there is no positive 8M effect to retain.

**Matched-duration sensitivity:** using arm4M−CTRL4M instead of arm4M−CTRL8M,
LABELS_KEEPV8 has +0.05625 / +0.00625 = **9.0× (900%)** episode retention, and
1.8284× alive-frame retention. It meets the numerical 80% condition under that alternative
reference. LABELS and LABELS_KEEPV8_LO still have nonpositive 8M effects. The 9.0× ratio
compares against a much weaker CTRL4M and divides by a tiny 8M effect; it does not establish
that 4M preserves final-control-relative performance. All arms have negative mass deltas
at every measured duration versus CTRL8M, so mass retention of a positive gain is undefined.

These data do not provide a stable basis to shorten T4 to 4M. The duration conclusion is
diagnostic only: **it never changes the T3 decision**, selects no new training run, and
confers no promotion or serving authority.

### Evidence and reproduction

- [Raw 70 probe rows](../../research/t4_curve_20261015/results/probes.json).
- [Analysis with absolute metrics, every seed-paired delta, SEs, and both retention definitions](../../research/t4_curve_20261015/results/analysis.json).
- [Completion receipt with checkpoint, row, summary, source, and result hashes](../../research/t4_curve_20261015/results/completion.json).
- [Watcher event log with every command and owned child PID](../../research/t4_curve_20261015/results/watcher.jsonl).
- Curve-tool tests: **14 passed** (analysis, seed audit, watcher).
- Independent read-only data audit: **PASS**, with zero mismatches across all 240 paired per-seed
  metric values, exact recomputation of the saved analysis, all 70 checkpoint hashes verified,
  all 20 successful training summaries verified, and the four-worker/no-retry event history checked.

Recompute the saved analysis without running probes or training:

```bash
SNAKE_DQN_DEVICE=cpu ./venv/bin/python - <<'PY'
import json
from pathlib import Path
from research.t4_curve_20261015.analyze import analyze, markdown
root = Path("research/t4_curve_20261015/results")
result = analyze(json.loads((root / "probes.json").read_text()))
assert result == json.loads((root / "analysis.json").read_text())
print(markdown(result))
PY
```

## Deferred paired A/B command (not executed)

Run sequentially on a quiet Mac only, after separately authorizing training/device use.
Both arms use seed 10, B8_s0, 8M transitions, batch-scale 2, and LABELS_KEEPV8_LO.
`AB_DEVICE` must be explicitly set for that future session; this preparation task uses CPU
only and does not set or invoke an accelerator device. Do not change any other flag between arms.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-t4prep
: "${AB_DEVICE:?Set the authorized device for a future quiet-Mac A/B session}"
artifacts=/Users/josenunez/Projects/ml/snake-dqn-artifacts
export SNAKE_DQN_DEVICE="$AB_DEVICE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for m in 1 2; do
  out="$artifacts/t4-update-every-ab-20261015/M${m}_s10"
  test ! -e "$out" || exit 1
  ./venv/bin/python research/redesign_m3c_20261008/train_m3c.py \
    --seed 10 --init "$artifacts/redesign-m4b-mac-20261013/B8_s0/final.pth" \
    --out "$out" --demo-data "$artifacts/redesign-m2b-20261008/data" \
    --transitions 8000000 --device "$AB_DEVICE" \
    --actors 2 --worlds 128 --segment-frames 5000 \
    --reward mass --mass-kappa 1.0 --kill-scale 0 \
    --gamma-start 0.998 --gamma-end 0.998 --gamma-hold 0 --gamma-anneal-end 1 \
    --eps-end-at 1 --mixed-frac 0.25 --anchor-mode v8margin \
    --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1 \
    --obs-version bv8 --v61-backend numba --learner-fast --obs-pack --batch-scale 2 \
    --checkpoint-every 2000000 --tight-telemetry --td-huber-delta 1.0 \
    --onpolicy-labels --label-frac 0.5 \
    --label-bank "$artifacts/t3-rollout-labels-20261015/bank" \
    --label-bank-frac 0.5 --label-bank-margin-scale 1.0 --update-every "$m"
done
```

The identical recipe is saved as
[`ab_update_every.sh.template`](../../research/t4_curve_20261015/ab_update_every.sh.template).
It was checked with `bash -n` only; no A/B training was run. Set `AB_DEVICE` explicitly
for a future quiet-Mac session, then invoke `bash research/t4_curve_20261015/ab_update_every.sh.template`.
