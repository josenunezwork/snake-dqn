# T4 update cadence and diagnostic checkpoint curve — 2026-10-15

Implements [the T4 preparation spec](specs/t4_update_every_and_curve_spec_2026-10-15.md).
This is development diagnosis only. **The curve never changes the T3 screen decision**,
uses no T3 screen worlds (rounds 62/63), and grants no promotion authority.

## Part 1: update cadence

Implementation and required CPU tests are in progress. `--update-every M` defaults to 1;
one learner update consumes credit for `M * 128 * batch_scale` ingested agent transitions.
Batch size is unchanged. Target synchronization remains counted in learner updates, so
increasing M increases transitions between target synchronizations.

The A/B training is deferred. No A/B or other training is authorized by this task.
The complete one-seed B8_s0, 8M, LABELS_KEEPV8_LO command will be recorded here after
flag validation; it is intended for a quiet Mac, not the active T3 lanes.

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
Completion requires all 70 valid probe rows and all 15 T3 runs. Results pending.
