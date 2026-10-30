# H5000 supported-bank downstream C/T package — 2026-10-30

Prepared at the owner's 2026-10-30 request under the adopted development sandbox.
This is prospective branch/fit/assessment work, simulator-D only. No downstream
production branches, fitting or registered look has occurred for this packet.
Independent Argon review precedes operator dispatch. No strict, Watch6-v2,
release or serving claim or authority is created.

## Immutable input and inherited science

Consume exactly the completed H5000 bank stage-0001.pt, 126,341,023 bytes, SHA256
112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0.
Collection pin 9fdc4cb655431686a60c6a548a03d1b9a05b1bd5, allocation
8cb3af9a7611ce469d1ec6d6474f493ec97d6d644647779d8fc9794e18164b93.
Collection receipt PASS, stop verified; terminal SUPPORT_COMPLETE. Its source
control, summary/support report, receipt hash and all verified immutable exports
are retained in bank-verification.json. No new worlds, anchor selection, death
selection, repartition, root replacement, collection, rescoring of support, or
success-driven horizon extension. Repeat support validation for integrity only.

Adopt CT_1028.md §1–2, the original ruling §4–5, and H5000 PREREGISTRATION.md plus
PREREGISTRATION_FIXES_20261029.md for the already collected bank. This package
changes orchestration, immutable input delivery, identity and operational caps.
The original frozen ct_branching.py, ct_fit.py, ct_models.py, ct_stats.py,
ct_mechanics.py and ct_topology.py remain unchanged. Inherited task: mechanics-v1,
terminal hero, in-step food refill, natural roots, bare B8 s3 and pinned opponents;
zero Watch6-v2 transfer claim. All parent-world partitions and root UIDs remain
those of the immutable bank. Branch RNG remains the frozen original action-free
root-UID/suffix scheme; fresh H5000 root UIDs already separate it from t34-cf-1028.
The new sandbox world-0 salt witness is not a new data-collection allocation.

- All 2,560 roots: every legal canonical action, H300, four selection suffixes
  0..3 and four independent scoring suffixes 4..7; same CRN consumers. Freeze
  Q_sel = S_sel + 0.25*M_sel/1000, a*, F_sel gap 0.50 before scoring suffixes.
- Three unchanged arms: frozen_phi, full_bv8, full_bv8_plus_topology. Same train
  roots/40k train anchors, 2,000 minibatches per arm, searched/anchor batches
  256/768, Adam trunk/head LR 1e-5/1e-4, seed 20261028, alpha .15, margin .25,
  anchor KL beta .05, identical schedule and detached train-only danger probe.
- Held-out only: 32 deaths from 32 worlds, 256 ordinary roots and 10k anchors.
  Parent-world bootstrap: 10,000 resamples, seed 3432. The original ct_stats
  function alone computes one ordered registered evaluation after all fits.
  INVALID/INCOMPLETE_SUPPORT precedes (a) >=25% rescuable deaths (<15% closes
  rescue-target; 15–25% inconclusive), (b) selected-action H300 mass benefit
  LB90>0, (c) full_bv8 retention >=30% and >=15pp above frozen_phi, (d) <=1%
  anchor flips and <=1 lethal ordinary flip. Topology additionally needs >=15pp
  over full_bv8 and cross-world D100–D300 AUC>=.83. Exact rational boundaries and
  Holm m=2 remain in original code. No interim pooled assessment or selective fit.
  No tuning, new threshold, changed N, extra suffix or alternative endpoint.

## Namespace, resume and mechanical completion

Study t34-ct-h5000-stage2-1030; namespace a100-dev/t34-ct-h5000-stage2-1030/stages.
Allocation = canonical SHA256 of full control (bank descriptors, source pin,
science and fixture flag). Initial ledger empty. Eight contiguous, deterministic
320-root chunks are stages 1..8, all three complete arm predictions stage 9,
registered evaluation stage 10. Atomic immutable payload+SHA+completion, then
bound header+SHA+completion. Bank is input-only and never republished as a stage.
A verified stage is skipped exactly, including final fits/evaluation. Interrupted
chunks restart only their missing unit; unfinished fits restart all three arms
from the unchanged initializer/schedule. No weights-only mid-fit resume.

Mac resume.py builds union ledgers using path/size/hash, refusing conflicting,
foreign, duplicate or gapped coverage; runtime rechecks all bytes and headers.
Completed stages remain immutable across attempts. Mechanical triggers are the
existing qualified timeout/lease/credential-expiry or acknowledged supervisor
transport stops after stop_verified and missing units, with scientific/fault
precedence checked across receipts and retained polls. One original plus at most
two manual completion attempts; every attempt gets fresh job/output/receipt and
preserves all prior evidence. No automatic completion retry. A scientific result
or fault never authorizes retry; a watchdog failure is not mechanical completion.
The initial packet refuses completion metadata. Any future completion needs a
separate sealed provenance-bound packet, full admission and current-pin canary;
resume.py alone supplies no dispatch authority. No series cap increase. Each
attempt uses the frozen 6h base/12h effective guard (maximum three allocations;
18h base/36h effective aggregate runaway envelope), not a doubled retry count.

## Forecast and operational guards

Original expected full package ~6.30 A100-hours included ~35.3min collection.
Now collection is excluded. Retain projected branching 5.45h (26.5M compacted
rows at 1350/s, unmeasured for this mature bank), fits ~8min, and allow ~25min for
bootstrap/source upload, bank verification, transfer, final assessment/export and
cleanup: **6h forecast/base, 12h effective timeout**. Original conservative branch
projection 8.90h is below this guard; it is not a guarantee. Maximum branch charge
36,864,000 rows; 6,000 fit minibatches = 6.144M sampled rows. Do not double measured
rates, N, horizon, minibatches or estimates. H100-first/A100-fallback use identical
6/12h envelopes: no assumed H100 speedup. Provisional planning rates H100 12u/h,
A100 8u/h: forecast 72/48 units, effective informational guards 144/96 units.
At previous observed A100 6.73u/h, 6h prices at 40.38 units. These are planning
inputs; the operator records fresh admission rates. No unit budget or rationing.

Real bank 120.49MiB plus ~22.66MB models, sidecars, ledger and ~1.1MB source feeds
the retained measured-conservative staging model (810734 B/s + 3.67s per file +
5.29s setup). transfer-projection.json must remain <300s whole-input staging.
Bank input upload is charged; the old 100MB private continuation-stage bound is
not applied to the new fixed bank input. Downstream exports originally ~22.4MiB;
256MiB per stage, 512MiB recursive cap and 768MiB supervisor transfer bound allow
headroom based on this real bank size without reexporting it. Caps refuse overflow,
never truncate data. Future completion inputs must reprice all retained stages
and pass the same whole-input bound, or be separately prepared before dispatch.

Use only tools/ops/dispatch.sh / qualified supervisor; no new transport changes,
Drive, RunPod or direct colab operations. Three-assignment account cap, leases,
<=900s meaningful-work watchdog, SIGUSR1 fence, <=60s phase/progress publication,
all single-thread environment caps, fp32/TF32-off and H100/A100 native geometry
settings retained. Every manifest requires resume skip_verified=true, retry max=2.

Mac admission executes the real status entrypoint on four fixed real roots and
16 anchors with H2/one minibatch per arm/32 bootstrap resamples. Full bank support
is validated first; the slice is fixture-only, cannot satisfy production admission,
returns PREFLIGHT_ONLY, and its gate output is sealed diagnostic only. It supplies
no scientific endpoint or representative CUDA throughput evidence. Execution-time
source-delivery policy is the same pure function called by guarded dry-run.
