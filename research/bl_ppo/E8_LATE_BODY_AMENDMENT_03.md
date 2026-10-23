# E8_LATE_BODY_AMENDMENT_03 — exact-return critic convergence

Status: **DRAFT_PENDING_ASTRA_AND_FABLE_JUDGMENT**, 2026-10-22 PT.
Prospective for a new recipe and new canary/source, not a completion attempt or
retry of the refused recipe. This draft, implementation, offline replay and
Codex-V review authorize **no dispatch or spend**. Publish on branch
`p1-snapbound-20261022`; main stays unchanged pending Fable judgment.

## 1. Authority and historical premise

Authority: external `big-lever/exp/JUDGMENT_P1_Y3_CRITIC.md`, especially
sections 3–5, and its authorized diagnostic's `offline-critic-diagnostic-01/DECISION.md`
under `bl-p1/y3/source-latebody-05-snapshot-bounds-staging-repin-20261022/`.
I0 verified 162 natural held-out late rows and baseline MSE 561,561.60.
D1 found the registered affine `[z,h]` head, fit to convergence on the natural
exact-return fit reservoir, reaches EV 0.5867 / MSE 307,567.7 / bias −105.1.
This is evidence of an optimization shortfall of the previous recipe, not a
qualification or evidence that a future source will pass.

The previous valid scientific refusal remains NOT_READY(P1 critic gate), with
its original receipt and metrics. No historical gate is rescored or relabelled.
Registrations, amendments 01/02, erratum 03, resume amendment 05 and all past
preparations/receipts/judgments stay byte-identical. This sibling supersedes
**only the final critic convergence step before qualification**, when explicitly
enabled. No amended-recipe Colab data exist at drafting.

## 2. Frozen prospective convergence recipe

After the existing natural critic-refit prefix and support construction, before
the single accuracy look, replace the one bounded Adam exact-target pass with
one CPU float64 least-squares solve. Keep the rollout/GAE critic refit unchanged.

1. Select **all natural completed FIT reservoir rows**, in retained reservoir
   order, at most the existing 10,000-row cap. The target is the existing raw
   gamma-one complete-life suffix return, closed only at death/H5000. Do not
   select only L≥800 rows or reweight rows. Held-out, fixture, supplement and
   unfinished-life rows never enter this solve or its PopArt moments. No new
   collection is authorized by convergence.
2. Form `X=[1,z,h]` (258 columns) without changing the frozen 256d core or clock.
   Solve `min ||X c - y||²` once with NumPy `linalg.lstsq`, CPU float64,
   `rcond=None`, one BLAS thread. Use the deterministic SVD minimum-norm solution
   even if rank deficient; report rank, count and fit error. No ridge, feature
   interactions, optimizer sweep, stopping criterion or model selection.
3. Set PopArt moments from the same fit targets: mean `μ=mean(y)`, second
   moment `mean(y²)`, scale `σ=sqrt(max(var(y),1e-8))`. Initialize moments with
   the existing output-preserving affine rescaling, then install normalized
   weights `c[1:]/σ` and bias `(c[0]-μ)/σ` in the existing fp32 head. The solve
   is CPU-only; the existing model's dtype, device and CUDA operations remain
   unchanged. Subsequent PopArt updates keep the existing output-preserving
   semantics and beta=.999.
4. Clear Adam moments for the two analytically replaced critic parameters only.
   Preserve all other optimizer entries, actor/core/encoder/teacher bytes,
   RNG states and transition/update counters. This is one analytic head solve,
   not an additional PPO update. A restored pre-gate snapshot deterministically
   repeats that solve on its restored fit reservoir; an actor-enabled snapshot
   retains its qualified head and receipt without refitting. Existing snapshot
   identity checks distinguish amended and unamended runs.
5. Validate finite nonempty inputs, finite coefficients/moments and finite
   installed predictions. Nonfinite failure retains the existing nonfinite
   tripwire; invalid inputs or a solver exception refuse admission. Export a convergence report in
   the critic-refit receipt, including the amendment identity. Evaluate accuracy
   exactly once after the step; finite fit refusals preserve available numeric
   evidence while nonfinite failures retain the existing tripwire precedence.

Enable only with explicit boolean P1 amendment flag; disabled means the old
one-pass recipe and old identity remain unchanged. Bind the enabled flag and
this amendment's SHA-256 into run identity and the reviewed executable closure.
New controls must opt in; no historical control is modified. The implementation
and control spelling are recorded in `P1_E8_03_HANDOFF.md`.

## 3. Gate and scientific scope remain frozen

EV ≥0.10 **and** MSE strictly below the float64 SVD `(1,h,L)` comparator,
on natural completed held-out L≥800 rows only. Baseline fits all natural
completed fit rows. Split remains whole-world `world_id % 5 == 0`.
Natural retained support remains fit ≥160 late lives / ≥640 late rows;
held-out ≥40 late lives / ≥160 late rows / ≥8 late lives per mix. The support
check at ≥2M fit-world transitions, at most one support-only continuation to
≤4M, supplement eligibility and accounting, fixture isolation, one-look rule
and refusal/nonfinite precedence are unchanged. No extra accuracy-driven fit.

Actor/environment/reward/features, seed 20261021, opponents, natural actor
resets, gamma1/lambda.995, 100M endpoint, PPO learning rates/epochs, CUDA path,
bank exclusions, Mac-only strict/serving gates and timing windows are unchanged.
The 1M ≤4u qualification still requires an actual Y3 PASS. B8 s3 remains served;
release authority remains with the owner. No protected study directory is read.

## 4. Forecast proposed for judge pricing — no admission

See `P1_E8_03_FORECAST.json` and `P1_E8_03_HANDOFF.md`. Frozen total 25.50u;
conservative charged-to-date 7.26u; residual 18.24u remains frozen. Proposed:
one new family canary plus one source, L4 ceiling 1.54u/h, zero rate tolerance,
first applicable wall/unit/transition cap wins, no unpriced GPU substitute.

| Item | Estimate (min) | Base timeout / effective timeout (min) | Base cap / effective cap (u) | Base reserve / effective reserve (u) |
|---|---:|---:|---:|---:|
| Canary | 15 | 15 / 30 | 1.00 / 2.00 | .20 / .40 |
| Source | 75 | 150 / 300 | 4.81 / 9.62 | .96 / 1.92 |

Caps, reserves and timeouts already incorporate the owner's 2× rule once;
estimates/rates/N are not doubled. Expected nominal 90min / 2.31u including
existing setup/export allowance. Conservative cap-plus-reserve liability uses
the effective caps above (reserves included): 11.62u, leaving 6.62u after the
7.26u historic charge. Even holding the unchanged, **unadmitted** 3.86u
replacement reservation leaves 2.76u. The 1M phase is separately blocked;
this forecast does not admit it or promise all future phases fit the residual.
No automatic completion or scientific retry is authorized.

The observed predecessor elapsed 3,349.5s (55.825min), gross charge 1.71u;
the unchanged source estimate provides 19.175min headroom. The replay records
local solve time; it is not an L4 end-to-end timing measurement. Until the new
exact-pin canary, allow up to 5min of that headroom for the ≤10k×258 CPU solve
and telemetry. A canary that cannot support the existing estimate/caps blocks
source admission; do not raise caps or shorten the registered work to fit them.

Before any future dispatch: Astra review, then Fable ratification and pricing;
current-code/current-pin family canary; fresh hashes, namespace/allocation,
empty or verified resume ledger and control/manifest; reviewed source receipt;
fresh account assignment inventory and debit reconciliation. The prior unexplained
assignment discrepancy remains disclosed; `{}` is not a live inventory. Use
the qualified supervisor through `tools/ops/dispatch.sh`, operator only. This
task does not create dispatchable manifests or call provider tools.

## 5. Offline validation and next disposition

Offline replay uses only the already verified Y3 chunk-index exports and
hash-verified D1 diagnostic. It calls the production convergence step, installs
the existing fp32 PopArt head and compares its natural held-out late metrics
to D1. Fix tolerances prospectively at EV absolute difference ≤1e-4 and
relative MSE difference ≤1e-3; these check numerical reproduction only and do
not change any gate. Require I0 count=162 and baseline within .5% of
561,561.60. No feature/head/threshold selection uses replay results.

Tests are light CPU fixtures, single process, `-p no:xdist`, all numerical
threads one; no full suite, training trajectory, timed Mac run, CUDA run or
Colab data. Independent Codex-V receipt reports scope, hashes, commands,
findings and unresolved limits. Review establishes software preparation only.
Fable may reject this proposal; without approval it stays blocked. A future
scientific gate refusal is final for this registered recipe, with no automatic
retry, extension, alternate head or best-checkpoint substitution.
