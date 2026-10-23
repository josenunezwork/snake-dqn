# F4 registration — nested rescue-life dose study for a frozen-B8 residual (round-26 judgment)

Status: registered by the round-26 judge (Opus stand-in for Fable), 2026-10-23. Binding for factory-label fits once the approval JSON
binds this file's sha256 and the post-change FitRecipe digest. Tier-1 development evidence only (Claim A, fixed checkpoints). No
outcome here promotes, serves or releases anything.

## R1. Question, comparator, claim boundary
Does growing selected rescue-bearing natural lives 1,024 -> 4,096 make the same frozen-B8 residual stronger in fresh paired H5000
gameplay, and does the top dose beat bare B8 s3 (sha256 3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8)? Primary
endpoint: per-world full-horizon terminal-hero mass integral over H5000 (dead frames zero). Inference is about three fixed checkpoints
under one fit seed (P7 section 1), not a data-scaling law; a null closes this recipe, feature set, update budget and dose range only.

## R2. Data, ancestry, doses
- Unit: a rescue-bearing life (one natural hero death with >= 1 exact H300 survivor). Roots, rows, branches, audits, H1440 extensions and
  any descendant stay on their ultimate parent's side. Lives are clustered by parent; every reserve statistic is parent-clustered.
- Folds: `world_fold(parent)` (sha256 of `bl-residual-f3-split-v1/<parent>` mod 5); fold 0 = held-out side.
- Reserve (fixed at D256, reused unchanged): the first 16 fold-0 S-Watch6 rescue lives then the first 48 other fold-0 rescue lives by
  append-only `allocation_index`, skipping any life whose parent already has 4 reserve lives. Report distinct parents and streams.
- Doses = selected support including the reserve: D256 / D1024 / D4096 = 192 / 960 / 4,032 fit lives (fold 1–4 allocation-index
  prefixes, nested, append-only). D1024/D4096 tranches should pre-filter parent IDs to folds 1–4 (outcome-blind); unused lives are reported.
- Composition: every tranche manifest uses the D256 tranche's S-Watch6 : S-gate parent ratio and gate roster ratio; realised shares are
  reported, never adjusted after outcomes.
- Cap reached without the selected support -> INSUFFICIENT for that dose; no refill, resize or cap raise.

## R3. Frozen fit recipe (identical for every dose; one final checkpoint each)
- Start: B8 s3 frozen; residual `Linear(256,6)` on `a100_a2.model.b8_features`, weight and bias exactly zero; export folded into B8's
  value/advantage head (`fuse_residual`); no teacher, search, veto or opponent ID at decision time.
- Optimiser: Adam, fp32, lr 0.001, 2,000 updates, seed 3403, batch 256; no early stop, checkpoint choice, seed swap, sweep or
  data-dependent threshold. Small doses repeat the deterministic sampler; larger doses get no extra updates.
- Batch: 128 rows from the doom stratum (exact rows with >= 1 exact survivor and >= 1 exact dier among legal, served-selectable actions)
  + 128 rows from the ordinary stratum; within each stratum draw a life uniformly, then a row uniformly within that life.
- Survival loss (both strata): every exact survivor preferred to every exact dying legal, served-selectable action, softplus(Q_die - Q_surv),
  pair weight 1, mean over pairs.
- Growth loss (ordinary stratum only): pairs of exact survivors with mass gap g > 0 (mass = mean hero length over H300),
  weight min(g,4)/4, loss = sum(weight * softplus(Q_poor - Q_good)) / max(1, number of eligible pairs in batch).
- Retention: KL(softmax(Q_B8) || softmax(Q_corrected)), temperature 1, all six actions, ordinary rows, coefficient 1.
- Residual L2 1e-4 (mean square of weight and bias). Coefficients: survival 1, growth 1.
- Prevalence weights are not used in any training loss. NOT_EVALUATED and UNKNOWN actions are never dying labels or negatives.
- Served action: B8's advisory mask with the normal-only fallback when empty.

## R4. Tripwires (P7 section 4; "previous student" = B8 s3 for every dose)
- Unconditional fault, any dose: non-finite loss/Q/gradient/parameter, illegal action, > 90 % one-action share in any 100k-decision
  window -> KILL_FIT for that dose.
- Fit-drift, binding for D4096 only (descriptive for D256/D1024): on a fresh 32-world H5000 probe bank (11 scripted / 11 frozen /
  10 mixed, own collision-audited namespace) with greedy D4096 and B8 trajectories: (a) at each D4096-visited state, served-argmax
  disagreement with B8 > 5 % in any hero-length decile (deciles from pooled D4096 probe decisions) -> KILL_FIT; (b) D4096 probe boost
  share > 1.5 x B8 probe boost share -> KILL_FIT; (c) "held-out decisive agreement" = life-macro rescue retention on the reserve
  (R5) < 0.70 -> KILL_FIT.

## R5. Offline reserve looks (admission and funding only; never efficacy evidence)
- Read by the reloaded exported fused checkpoint, rowwise, on the Mac, after that checkpoint is immutable. Look R-A: D256 and D1024
  jointly, after D1024 is frozen. Look R-B: D4096, after it is frozen. Each look is create-only and happens once.
- Life-macro rescue retention: per reserve life, fraction of its decisive rows (bare exact-dies, >= 1 exact survivor) where the served
  action is an exact survivor (UNKNOWN or NOT_EVALUATED selection counts as not retained); mean over lives. Row-level retention reported.
- Archive: every natural-stream ordinary row of every life (rescue-bearing or not) whose parent is fold 0 and whose allocation_index is
  <= the largest reserve allocation_index; parents need not contain a rescue. Census frozen and hashed before Look R-A.
- Harm screen (empirical, zero-event; not a population confidence bound): H = unweighted count of archive rows where the served action
  changes from B8 and either (selected exact-dies and B8 exact-survives) or (selected or B8 outcome not EXACT). Pass iff H = 0. Report N,
  weighted rates with unclipped design weights 1/sampling_prob, clipped-at-5 rates, ESS, unknown coverage, regret, override frequency,
  and margin-binned harm/retention as descriptives.
- Admission (D4096): harm screen pass and R4(c). Failure of the harm screen -> STOP_ADMISSION.

## R6. Staging and stop/advance rules
1. D256: collect (tranche-1 caps base 40 / effective 80 u). Fit. Unconditional fault -> STOP (recipe defective; KILL_FIT; no D1024
   purchase). Otherwise proceed to D1024 collection automatically.
2. D1024: collect under its cap. Fit. Unconditional fault -> KILL_FIT, no D4096 purchase. Then Look R-A.
3. D4096 purchase: if F2's published disposition does not strike volume (25 -> 100 % selected-action-survival change LB > 0), buy.
   If F2 strikes volume, buy only if, on Look R-A, mean over reserve lives of [retention(D1024) - retention(D256)] >= 0.05 and its
   parent-cluster bootstrap LB90 > 0 (10,000 parent resamples, seed 3403, percentile) and D1024 passes the harm screen; else STOP_FUNDING
   (terminal; no gameplay; recipe closed). If F2 has not published before Look R-A, the F2-strikes branch applies.
4. Before D4096 dispatch, project cost = measured Colab units per selected rescue life (D256 + D1024 receipts) x 3,072 x 1.2. If the
   projection exceeds the D4096 base cap on every qualified placement -> INSUFFICIENT (budget); no gameplay.
5. D4096: collect, fit, probe tripwires, Look R-B admission. Any failure -> its terminal label; no gameplay.
6. Gameplay (Mac only, permitted gaps, one look): H5000 screen, then Watch6 only if every H5000 GO bar passes.
Units not run because a rule stopped the path are NOT_RUN_BY_RULE, never INCOMPLETE.

## R7. Caps (Colab units; effective = 2 x base, applied once; first cap wins; all attempts/setup/canaries/exports included)
D256 tranche 40 / 80. D1024 increment 120 / 240. D4096 increment 480 / 960. Fits, whole series 10 / 20 (L4; A100 only after a cached
fit measures >= 5x speed-up). Gameplay, probe and reserve looks: Mac, 0 u. Cap exhaustion never buys N, a refit or a retry.

## R8. Gameplay development screen
- Bank: one fresh collision-audited Tier-1 namespace, 128 H5000 terminal-hero worlds per mix (scripted, frozen, mixed), seeds/IDs/parents
  unique across the bank, disjoint from label, archive, probe and strict namespaces. Arms: B8 s3, D1024, D4096 (D256 not played). Each arm
  starts from an exact clone of the same initial state, roster, hero slot and wrapper. Fixed N, one look, no extensions. Delivered action
  traces recorded for every arm.
- Per world w in mix m: d_top = M(D4096) - M(B8); d_dose = M(D4096) - M(D1024).
- Estimator per contrast: G = (mean_s + mean_f + mean_m)/3; V = sum_m s_m^2 / (9 * 128); df Welch-Satterthwaite
  (sum_m a_m)^2 / sum_m (a_m^2 / 127) with a_m = s_m^2/(9*128); one-sided 90 % bounds G -/+ t_{df,0.90} sqrt(V). Per-mix bounds use
  t_{127}. Hierarchical paired bootstrap: sensitivity only.
- Efficacy family (Holm, one-sided, alpha 0.10, exactly two hypotheses): H_top: G_top <= 0; H_dose: G_dose <= 0. Holm rejection of both
  is required for GO.
- Point bars: G_top >= +40; G_dose >= +20.
- Guards (all required; intersection-union; unadjusted): (a) every mix: per-mix one-sided UB90 of mean d_top >= 0 (not < 0);
  (b) survival guard A = survival band v2 `pooled_ni_continue`, single look, band_alpha 0.05, pooled margin 0.05, mix margin 0.075,
  world order = bank order, D4096 vs B8; (c) survival guard B: scripted no-death fraction D4096 - B8 >= +0.03 (point);
  (d) delivered boost share of D4096 over all H5000 decisions in [0.5 %, 10 %] (point).
- Futility / closure on G_dose (unadjusted one-sided 90 % UB): UB <= 0 -> KILL_DOSE_CLOSED; else UB < +20 -> KILL_DOSE_FUTILE.
- Product (only if every H5000 GO bar passes): 128 fresh paired continuous-respawn mixed Watch6 sessions, 14,400 frames each, hero vs
  5 x FRP3-s12+v8, dense rule off, arms B8 and D4096, exact-clone start. Deaths ratio R = (sum deaths / sum hero decisions)_D4096 /
  (same)_B8 - 1; mass ratio Q = mean full-session hero mass integral D4096 / B8 - 1. Paired-session bootstrap (10,000 session resamples,
  seed 3403, percentile). Pass iff two-sided 95 % interval of R lies entirely below 0 and lower 95 % bound of Q >= -0.05 (both
  required; intersection-union). The 20 % reduction is a reported target. Median/p95 length and late large-body deaths reported.
- Reported, not decision-bearing: D1024 vs B8 contrasts and guards, all per-mix estimates, SEs, bounds and p-values.

## R9. Ordered result labels (immutable analyzer; complete ledger; one look)
1. INVALID: bad digest/lineage/seed audit/parity, foreign or duplicate unit, unsealed output, wrong controller identity, unregistered recipe.
2. INCOMPLETE: a unit required by the path actually taken is missing (never for NOT_RUN_BY_RULE units); do not shrink N.
3. INSUFFICIENT: a dose's selected support or its budget projection (R6.4) cannot be met under the caps.
4. KILL_FIT: an R4 tripwire on the dose where it is binding (D256/D1024 faults stop the path at that stage).
5. STOP_FUNDING: R6.3 F2-strikes branch failed.
6. STOP_ADMISSION: D4096 harm screen failed.
7. KILL_DOSE_CLOSED: UB90(G_dose) <= 0.
8. KILL_DOSE_FUTILE: UB90(G_dose) < +20.
9. GO_TOP_DOSE_DEV: Holm rejects both efficacy nulls, both point bars, all four guards and the product bars pass.
10. INCONCLUSIVE: every other complete result.
Labels 4–8 close this recipe/feature/dose-range lane: no more labels, refits, heads or seeds under F4. If D4096 passes every top-vs-B8 bar
but the dose contrast fails, the label stands and the flag TOP_PASSED_DOSE_FAILED is reported; any use of that checkpoint needs a new
advisor-reviewed, judged registration. GO buys only a separately registered fresh Mac strict-gate plan.

## R10. Power and honesty
128/mix is a fixed falsifier, not an 80 %-powered trial. Historical L2-prime per-mix SEs rescaled to 128/mix give a pooled mass SE
near 8 for a single arm-vs-B8 contrast; the dose-contrast SE is unmeasured. At SE 8–15, a true +20 dose effect passes its point bar about
half the time; a true zero effect triggers KILL_DOSE_FUTILE about 70–90 % of the time, a true +20 about 10 %. The harm screen is an empirical
zero-event filter over roughly 10^3 archive decisions; no population harm rate below 1/1,000 is claimed. Judgment prior P(GO) ~ 0.10.

## R11. Unchanged
Exact CPU simulator only (snapshot/restore, clone_world_exact, full_digest, exact_stop.run, native v8); no learned simulator, smaller
maps, synthetic bodies or prefix mutations. Dispatch only via tools/ops/dispatch.sh on the qualified CLI tree. Strict gate stays exact and
Mac-only. B8 s3 stays served; release needs a strict pass, serving qualification and the owner's OK.
