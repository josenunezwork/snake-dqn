# T34 Claim A amendment: corrected projection objective, frozen update dose, D0 diagnostics, R1 labels

**Status: APPROVED WITH MECHANICAL CONDITIONS C1–C6 by independent Fable 5.1, 2026-10-20 PT.**
Prospective and pre-data. The ruling is preserved in
`docs/research/advice/2026-10-20_colab_lanes/CLAIMA_AMENDMENT_JUDGMENT.md`.
Execution still requires the ruling's conditions, recording in CAMPAIGN_LOG and merge to main.
Release and serving authority stays with the owner.

- **Date:** 2026-10-21 (PT). **Author:** Claude (operator). Code: branch `t34-claim-a-amendment-20261021`.
- **Rules on:** JUDGMENT §7a–§7d, lane rows L1/L2 and §5 item 2
  (`docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md`). Also RULING_round0_go_file.md (same folder) and
  findings 1–6 of the Claim A learning-design review (`snake-dqn-artifacts/fable-panel-claima/design.md`).
- **Pre-data statement:**
  - Nothing under `snake-dqn-artifacts/t34-round0-study-01/` was opened.
  - No Round 0, Path B or R1 outcome is known to the author.
  - The only data this document uses are the A5 v2 exports, which are already excluded from every claim, and the
    review's published counts.

## 0. What this amends and what it leaves alone

**Amends** (Claim A only):

- the Claim A projection objective and row rule: FINAL_PLAN §3 "Student fit", bullets 1–2, and the frozen
  `research/t34_exit_b8_fit/loss.py` + `data.py` as used for a production fit;
- "epochs" as the fit's unit, replaced by an explicit update dose;
- the meaning of "Round 0 GO" for R1, restating the ruling;
- the R1 shape;
- the T34 unit cap.

**Unchanged:**

- every FINAL_PLAN §8 tripwire, including the **70% held-out decisive agreement** (not weakened, not moved to a
  later epoch, not redefined as improvement over baseline);
- the ratified package `research/t34_exit_b8_fit/` (no file in it is edited; the corrected objective lives in
  `research/t34_claim_a_loss/losses.py` and reuses the ratified KL/adv-MSE arithmetic);
- teacher_v2 and the label job;
- the strict gate, survival band v2, serving qualification and P7 statistics;
- §5 Amendment conditions;
- the owner's release authority.

## 1. Facts this rests on (measured; not evidence about any student)

| Fact | Source |
|---|---|
| A5 v2 exports: 39 worlds, 190,779 labelled rows. Of these: 190,483 `shadow_confirm`, 225 decisive `horizon_rescue`, 71 `all_die_delay` (label weight 0). 21/39 worlds carry decisive rows. | design.md "Measured facts" |
| The frozen loss shares one `sample_weight` between KL (τ=0.05), the margin-0.1 hinge and adv-MSE. In a free-Q probe (each decisive row's 6 Q values optimised directly, 5,000 Adam steps) it flips **0.4%** of decisive rows to a\*, and 97.8% stay bare; the result is identical at weight 10. | `research/t34_exit_b8_fit/loss.py:43-64`; design.md finding 1 |
| The frozen 1:1 searched:unsearched rule gives ~1 optimizer step/epoch on 39 worlds and ~60 steps per fit on 800 worlds. | `data.py:80-91`; design.md finding 2 |
| `all_die_delay` rows count as searched and so claim anchor quota while training nothing. | `replay_source.py:329-332`; design.md finding 6 |
| The default 10% world split on A5 gives 0 held-out decisive rows. | design.md finding 5 |
| Sparse teacher_v2 labels ≈ 0.036 u/world; ≈ 5.8 decisive rows/world (A5: 225/39 = 5.77). | JUDGMENT §0, §1(a) |
| Diagnostic cache regeneration on the Mac: 1 A5 world (2,105 kept rows) replays hash-exact in 3.5 s; max \|Q_recorded − Q_B8\| = 7.2e-5; 13.5 KB/row. | this branch, `cache.build_world` smoke (one world, 1 thread) |

## 2. §7a: the corrected objective (L3 2×2)

### 2.1 Row classes (from the ratified `normalize_row`)

| Class | Rows | Trains? |
|---|---|---|
| **Decisive** | `horizon_rescue` (bare dies, a survivor exists, F ≠ ∅) | Decisive stream only |
| **Anchor** | every `shadow_confirm` row, plus searched rows whose bare action survives | Anchor stream only |
| **Delay** | `all_die_delay` | **Never.** Excluded from both streams and from every quota. |

### 2.2 Objective

τ = 0.05, margin = 0.1, adv weight = 0.01 (the ratified `LossConfig`). The KL and adv-MSE terms are computed by
the ratified `exit_fit_loss` arithmetic: centred log-partition KL, with exactly zero loss and gradient at
Q = Q_prev.

- Anchor stream, mean over the anchor rows in the batch:
  L_A = mean_i [ KL( softmax(Q_prev/τ) ‖ softmax(Q_θ/τ) ) + 0.01 · adv-MSE(Q_θ, Q_prev) ]
- Decisive stream, mean over the decisive rows in the batch:
  L_D = mean_i [ τ · CE( softmax(Q_θ/τ), a\* ) + Σ_{f∈F} max(0, Q_θ(f) − Q_θ(a\*) + 0.1) ]
- Total: **L = L_A + λ · L_D**, with λ = 1.

Rules:

- The decisive stream has no KL and no adv-MSE. The anchor stream has no CE and no hinge.
- Each stream is normalized by its own row count, so neither stream's weight depends on how many rows the other
  has. This is the "separate, normalized stream" of §7a.
- One coefficient variant is registered: **`split_tau2`**, which multiplies the anchor KL by τ². This puts the
  KL gradient on the same scale as the hinge (otherwise the KL gradient is 1/τ = 20× larger).
- D0 (§4) picks between `split` and `split_tau2` and their step size per trunk. λ, τ, margin and the 0.01 adv
  weight are not tuned.

### 2.3 Batches, anchors, dose

- **Batch:** 1,024 rows = **128 decisive + 896 anchor**.
- **Decisive rows:** drawn by cycling random permutations over all training decisive rows.
- **Anchor rows:** drawn the same way over the **full** training shadow-confirm pool (§7a), never a 1:1 quota.
- **DAgger age (R2+ only):** older-round rows are intended to keep weight 0.5 inside their stream's fixed-count
  mean. **Not implemented on this branch; required by any R2+ registration**, with tests before use.
  The split loss currently uses `sample_weight` only as a positive-row mask. R1 has only current-round rows.
- **Dose:** an explicit number of optimizer updates U\*, run in **4 blocks** of U\*/4 updates. Block ends replace
  "epochs" for tripwire checks (`train.py:281-288`, `diagnostics.py`). L3 uses the student at U*;
  block ends do not select a student (§10 answer 4).
  - Starting proposal: U\* = 2,048, so 512 per block.
  - D0 freezes U\* ∈ {512, 1,024, 2,048, 4,096} by the rule in §4.4, using excluded A5 data only.
  - The optimizer is Adam: 2.5e-5 on pretrained parameters and 1e-4 on new ones (ratified), unless D0 selects
    the registered alternative for that trunk.

### 2.4 The 2×2 (registered for L3; L3 still needs its own registration under §7h)

|  | trainable B8 (2.5e-5) | frozen B8 trunk + zero-init residual head R(h) (1e-4) |
|---|---|---|
| **frozen loss** (ratified `exit_fit_loss`) | control | control |
| **corrected loss** (§2.2, D0-selected variant and lr) | arm | arm |

- The residual head is `Q = Q_B8 + W₂·relu(W₁·h)`. Here h is B8's fused 256-D feature, and W₂ is zero-initialised,
  so the step-0 Q equals B8's exactly.
- The **frozen-loss cells are controls.** They use the same streams, batch and dose as the arms, so the contrast
  isolates the loss. Their shared-weight loss uses decisive `sample_weight = 7.0` (= 896/128) and anchor
  weight 1.0: the shared weighted mean is 0.875 times the sum of the two per-stream means, with equal stream
  balance. This is a normalization equivalence, not an equivalence of the frozen and corrected objectives.
  They stop at their tripwire. This shows the correction-vs-retention conflict on data rather than asserting it.
- **Tripwires are unchanged** (FINAL_PLAN §8), evaluated at every block end:
  - non-finite loss;
  - more than 90% one-action share;
  - disagreement with the previous student above 5% at any length decile;
  - boost share above 1.5× the previous student's;
  - **held-out decisive agreement below 70%** (raw six-action argmax == a\* on the R1 held-out bank, §6).
- A data-shape failure (held-out support below §6.3) is `PRECONDITION_FAILED`, never `FIT_STOPPED`
  (design.md finding 5).

## 3. Why the dose is frozen in updates

- The A5 bank gives each D0 fold ~150 training decisive rows, so 2,048 × 128 draws ≈ 1,750 passes over them.
- R1 gives ~4,600 training decisive rows, so the same 2,048 updates ≈ 56 passes.
- The anchor pool also scales with the number of worlds.
- Updates fix the anchor drift budget (KL steps taken against B8), which is what the tripwires guard. Passes over
  decisive rows are reported for each dose but are not the frozen unit (§10 answer 2).

## 4. L1 D0 diagnostics (excluded A5 data only)

### 4.1 Data

- The 39 A5 v2 world exports listed above. The packer refuses any world outside `t34-exit-a5/v2/`, any carrier or
  rollout other than B8 s3 (`3e649783…`), and any export whose bytes differ from its `.sha256` sidecar.
- They are packed into one deterministic tarball. On this branch's packer: 16.8 MB, sha256 `82fd839d…`.
- Each world is regenerated by the ratified A2 replay. Every saved state hash and the recorded Q must match within
  1e-3 + 1e-4·|q|. Any rejected world makes the run `PRECONDITION_FAILED`.
- **Anchor stride = 1** (the full shadow-confirm pool).
- Folds: 3 world-level folds, stratified on decisive-bearing worlds and dealt by sha256(world id)
  (`assign_folds`). Held-out metrics use worlds the fit did not see; the permissive diagnostic uses the
  training fold, and the free-Q probe uses all decisive rows.

### 4.2 Registered measurements (`research/t34_claim_a_loss/harness.py`)

**Free-Q probe** for `frozen`, `frozen_tau2`, `split`, `split_tau2`: on all decisive rows, 5,000 Adam steps at
lr 1e-2.

**Cells** (3 folds each):

| Cell | Loss | Trunk | lr | Row rule | Checkpoints (updates) |
|---|---|---|---|---|---|
| frozen_recipe | frozen | trainable | 2.5e-5 | ratified 1:1, 5 epochs | epoch ends |
| frozen_trainable | frozen | trainable | 2.5e-5 | streams | 0, 32, 128, 512, 1024, 2048 |
| frozen_residual | frozen | residual | 1e-4 | streams | same |
| split_trainable | split | trainable | 2.5e-5 | streams | 0, 32, 128, 512, 1024, 2048, 4096 |
| split_trainable_lr1e-4 | split | trainable | 1e-4 | streams | same |
| split_tau2_trainable | split_tau2 | trainable | 2.5e-5 | streams | same |
| split_residual | split | residual | 1e-4 | streams | same |
| split_residual_lr1e-3 | split | residual | 1e-3 | streams | same |
| split_tau2_residual | split_tau2 | residual | 1e-4 | streams | same |

The `frozen_trainable` and `frozen_residual` stream controls use decisive `sample_weight = 7.0` and anchor
weight 1.0. With 128 decisive and 896 anchor rows, their shared weighted mean equals 0.875 times the sum
of the two per-stream means (§2.4). `frozen_recipe` retains the full ratified 1:1 recipe and weights.

At each checkpoint the harness records:

- training-fold and held-out-fold decisive agreement (raw argmax == a\*, row-pooled and world-weighted);
- argmax == bare, argmax not forbidden, a\* margin;
- on held-out anchors: agreement with B8, the maximum length-decile disagreement, the boost-share ratio and the
  one-action share.

### 4.3 Kill rule (JUDGMENT L1)

The projection is closed if either holds:

- no corrected loss flips **≥ 70%** of decisive rows even with free Q; or
- no free-Q-eligible corrected cell reaches **≥ 70%** pooled training-fold decisive agreement at any checkpoint
  (the permissive diagnostic).

Free-Q eligibility is stricter than the first prong: ≥ 99% (§4.4). If all corrected losses reach free-Q
agreement in [70%, 99%), none is eligible and the result is also `KILL`, regardless of in-sample agreement.

Claim A then rests on the tower question only (L6), as JUDGMENT §2 states.

### 4.4 Selection rule (frozen; `research/t34_claim_a_loss/decide.py`)

1. **Free-Q eligibility.** A corrected loss is eligible only if its free-Q flip rate is ≥ 99% ("free-Q ≥ 99%
   train flips").
2. **Permissive gate.** As in §4.3.
3. **Guards,** on held-out anchors, every fold, at every checkpoint up to the dose:
   - maximum length-decile disagreement ≤ 5%;
   - boost ratio ≤ 1.5;
   - one-action share ≤ 90%.
4. **Dose.**
   - The default is 2,048.
   - If the guards fail at or before 2,048, the dose is the largest guarded value in {1,024, 512}. If none is
     guarded, the cell is ineligible.
   - If 2,048 is guarded, 4,096 is guarded, and pooled held-out decisive agreement at 4,096 is at least 0.05
     higher than at 2,048, the dose is 4,096.
5. **Per trunk,** pick the eligible cell with the highest pooled held-out decisive agreement at its dose. Ties go
   to the smaller dose, then the smaller lr, then `split` before `split_tau2`, then the name.
6. **Outcome.**
   - `PASS` if at least one trunk is selected. A trunk with no eligible cell drops its corrected arm from L3.
   - `KILL` if neither trunk keeps the guards for ≥ 512 updates.
   - The frozen-loss cells are reported as controls and are never selected.

The held-out agreement here is a cross-fold *trend* on ~225 rows across ~21 worlds. It chooses among corrected
cells and is reported. The 70% held-out bar is applied, unchanged, to the production fit on the R1 held-out bank.

### 4.5 Execution and cap

- **Job:** one L4 job under the qualified supervisor `research/colab_cli`, built by
  `research/t34_claim_a_loss/make_manifest.py` (entrypoint `d0.sh`, driver `run_d0.py`).
- **Inputs:** the pinned public catalog (B8 s3 + league checkpoints for replay) and the uploaded tarball.
- **Runtime:** cache on CPU worker processes (nproc − 1). Cells on CUDA in fp32, with TF32 off.
- **Outputs:** `d0_timing.json`, `cache_manifest.json`, `d0_report.json`, `D0_DECISION.json`. No weights are
  written or exported.
- **Sizing:** est 100 min, timeout 240 min, unit cap 8.0 u (240 min × 2.0 u/h). The **lane cap is 12 u**, which
  leaves room for at most one judge-approved mechanical rerun with a ≤ 4 u cap. There are no automatic retries.
- **Before enqueue,** the operator runs the real entrypoint on the Mac on a tiny input (Colab job preflight rule),
  using `d0.sh` in local mode.
- **Order:** D0 may run from the merged main commit once this amendment and the ruling's preflight/dispatch
  conditions are satisfied. It does not wait for Round 0 or Path B, because A5 data is excluded and this is
  design work; it is not a Round 0-conditioned lane.
- **Recording:** `D0_DECISION.json` is recorded in CAMPAIGN_LOG before any L3 registration. L3 binds its SHA-256.

## 5. §7b: "Round 0 GO" binding (restates the ruling; no new rule)

For R1 and every Round 0-conditioned lane, "Round 0 GO" means
`<study_root>/amendment-boost-floor/decision.json` with all of the following:

- `schema = t34-round0-decision-amended/v1`, `governing: true`, `decision: "GO"`,
  `evidence_authority: "TIER_1_DEVELOPMENT"`;
- a sealed `phase_terminal.json` (`t34-round0-amend-terminal/v1`, `status: COMPLETE`) whose `files` map binds that
  decision's SHA-256;
- that terminal's SHA-256 in an `amendment-boost-floor` line of
  `research/t34_round0_amend/ledger/<study_root>.jsonl` on `origin/main`.

The registered `analysis/decision.json` never governs R1. The only fallback in the ruling is a recorded late-
publication `decide` refusal. It is not automated here: it needs a CAMPAIGN_LOG record and a judge ruling first.

Implementation: `research/t34_claim_a_r1/make_manifest.py::governing_round0` refuses anything else.

- Every R1 chunk manifest embeds the binding: decision and terminal SHA-256 and the ledger path.
- The manifest builder refuses any binding that is not a GO from that function.
- The label seal refuses the same.

## 6. §7c: R1 shape

### 6.1 Banks

- **800 training worlds + 200 whole-world held-out worlds.** They come from different namespaces, so the split is
  fixed by seed allocation before any label exists and never depends on label content:
  - training: `t34-claim-a/r1-labels-v4/train/s{0..7}/a0`;
  - held-out: `.../heldout/s{0,1}/a0`;
  - 100 worlds each.
- Attempt `a1` namespaces are audited for a judge-approved mechanical rerun only.
- **Seeds:** `research/t34_claim_a_r1/SEED_AUDIT_R1.json`.
  - Status PASS, with the full-history scan.
  - Disjoint from the Round 0 (242), Round 0-D (272) and retention v2 (177) banks, with 0 collisions each, and
    pairwise disjoint across R1 namespaces.
  - The namespace family is `r1-labels-v4` because earlier families failed the fail-closed freshness scan on
    incidental numeric matches.

### 6.2 Job

Each chunk is the ratified `research/t34_exit_b8/label.sh` v2 label job, unchanged:

- bare B8 s3 + teacher_v2 (shadow-LAD v2), **β = 1**, confirm interval 8, H5000;
- gate-matched 50/25/25 mixes, CPU process fan-out on one L4.

Only four things are set by `research/t34_claim_a_r1/make_manifest.py`:

- `T34_NAMESPACE`;
- `job_id`;
- `claim = R1_LABEL_DATA_ONLY_NOT_EVIDENCE`;
- `budget.other_assignments = {}`.

### 6.3 Held-out support minimum

- The held-out bank must have **≥ 60 decisive-bearing worlds and ≥ 300 decisive rows**. Otherwise R1 is
  **`PRECONDITION_FAILED`**: it is reported, there is no top-up without a new registration, and it is not a
  learning result.
- Expectation at A5 density: ~108 bearing worlds and ~1,150 rows, so failure would itself be a finding about the
  teacher.
- The check is applied by `research/t34_claim_a_r1/seal.py` on the Mac after every chunk is terminal.
- The seal also writes the training and held-out world-set hashes, so a fit can bind the training set and can
  never read a held-out world.

### 6.4 Ordering (§5 Amendment, unchanged)

- R1 is registered after a published governing GO and dispatched after its own approval. It may run in parallel
  with the Path B retention phases.
- Its outputs are label data only until the Path B disposition.
- No Mac work for R1 runs while a retention phase runs or is admitted. The seal requires an explicit attestation.
- R1 is never cited in the retention verdict.

## 7. §7d: caps

- The FINAL_PLAN §1/§6 hard cap (≈ 270 u) is raised to **the sum of the admitted lane caps**. This amendment admits
  only:

  | Lane | Cap | Covers |
  |---|---|---|
  | L1 | **12 u** | D0 (§4) |
  | L2 | **60 u** | R1 labels (§6) |

- R1 admission is checked per chunk, `research/t34_claim_a_r1/plan.py::admissible`:
  settled units + every open chunk's cap + the new chunk's cap ≤ 60.
  - The ratified per-chunk cap is 12 u (360-min timeout at 2.0 u/h).
  - So at most four chunks are open at once early on. The JUDGMENT's limit of three concurrent assignments
    account-wide binds first.
  - Expected spend is ≈ 1,000 × 0.036 ≈ 36 u.
- Every other lane (L3–L9) adds its cap only through its own approved registration.
- Units from a killed lane do not transfer.
- Every admission is checked against the live `colab usage` balance minus outstanding liabilities.

## 8. Routing

| Event | Next |
|---|---|
| D0 `PASS` | L3 registration binds `D0_DECISION.json` (loss variant, lr, U\*, per trunk). It may be drafted now, and fits wait for the R1 bank and the PB disposition. |
| D0 `KILL` | Close projection (no L3 corrected arms; no production fit). Claim A rests on L6 (tower) only. R1 labels are still a valid bank for L6 if R0 is GO. |
| D0 `PRECONDITION_FAILED` / `FAILED` (replay mismatch, crash) | Mechanical: diagnose; at most one judge-approved rerun inside the 12 u lane cap. |
| Governing R0 GO | Register R1 (§6) → review → one judge approval → dispatch. |
| Governing R0 not GO | No R1. Route per the ruling's table (0-D, bug branch or mechanical branch). |
| R1 held-out support below §6.3 | `PRECONDITION_FAILED`. No fit on R1, and a new registration is needed for any top-up. |

## 9. Implementation map (branch `t34-claim-a-amendment-20261021`)

| File | Role |
|---|---|
| `research/t34_claim_a_loss/losses.py` | `frozen` (ratified, imported), `frozen_tau2`, `split` (§2.2), `split_tau2` |
| `research/t34_claim_a_loss/cache.py` | Hash-exact replay → per-world npz (rasters, q_prev, fused h, row class); Q identity check |
| `research/t34_claim_a_loss/harness.py` | Free-Q probe, stream/ratified samplers, trainable/residual cells, world-level folds, metrics; `D0_CELLS` |
| `research/t34_claim_a_loss/decide.py` | The §4.4 rule → `D0_DECISION.json` |
| `research/t34_claim_a_loss/inputs.py` | Deterministic, hash-bound A5 tarball (pack on the Mac; verify and unpack on the runtime) |
| `research/t34_claim_a_loss/run_d0.py`, `d0.sh` | D0 driver and L4 entrypoint |
| `research/t34_claim_a_loss/make_manifest.py` | Pure D0 manifest builder (L4, ≤ lane cap; never dispatches) |
| `research/t34_claim_a_r1/plan.py` | 800 + 200 plan, lane admission, held-out support rule |
| `research/t34_claim_a_r1/make_manifest.py` | Governing-GO binding + ratified label job retargeting (refuses without GO) |
| `research/t34_claim_a_r1/seal.py` | Mac-side label inventory, support minimum, world-set hashes |
| `tests/test_t34_claim_a_loss.py`, `tests/test_t34_claim_a_d0.py` | Tiny-fixture tests |

## 10. Six binding answers from the independent judge

These answers are binding under the approval's C1–C6, preserved in `CLAIMA_AMENDMENT_JUDGMENT.md` §3.

1. **Permissive diagnostic:** in-sample training-fold decisive agreement ≥ 70%, as implemented. The small A5
   folds cannot support a 70% held-out requirement without testing data starvation. Held-out trend remains
   reported and selection-only; the production R1 held-out bar stays unchanged.
2. **Transfer unit:** freeze dose in optimizer updates, not decisive passes. Updates bound anchor drift;
   freezing passes would multiply R1's dose about 30×. Report `decisive_passes_at_dose` alongside U*.
3. **Control cells:** L3 frozen-loss controls use the new streams and dose to isolate the loss. Their decisive
   `sample_weight = 7.0` gives the equal stream balance scaled by 0.875 (§2.4). D0 alone retains
   `frozen_recipe` as the full ratified-recipe control.
4. **No block selection in L3:** the student is the model at frozen dose U*. Block ends check tripwires only;
   a breach yields `FIT_STOPPED`. Read the held-out bank once per block for tripwires and once at U* for the
   report. Run the 32-world probe once, at U*. A later registration seeking selection must add its own
   training-side world split.
5. **World balance:** draw decisive rows uniformly, as implemented; D0 freezes dose under this sampler.
   Report `flip_world_weighted` beside the pooled rate. The unchanged 70% R1 held-out bar is row-pooled;
   the world-weighted rate is reported, not gated.
6. **Order:** D0 may run as soon as this amendment is merged, before any Round 0 or Path B verdict, subject
   to C2's preflight and dispatch requirements. Its only input is excluded A5 data, and it emits no student
   evidence. D0 is not a Round 0-conditioned lane. L3 fits still wait for the R1 seal (`SUPPORTED`) and the
   published Path B disposition.
