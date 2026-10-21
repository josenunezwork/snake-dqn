RULING: APPROVE

# JUDGMENT: T34 Claim A L3 amendment (§7a–§7d) — one independent judge, prospective, pre-data

- Judge: Fable 5.1 (independent; wrote none of the spec, code or governing documents). 2026-10-20 PT.
- Target: branch `t34-claim-a-amendment-20261021` @ 71de7645; spec
  `docs/research/specs/2026-10-21_t34_claim_a_amendment.md`; code `research/t34_claim_a_loss/`,
  `research/t34_claim_a_r1/`, `tests/test_t34_claim_a_loss.py`, `tests/test_t34_claim_a_d0.py`.
- Governing: JUDGMENT.md §7a–§7d + lane rows L1/L2/L3; RULING_round0_go_file.md; design.md findings 1–6.
- Pre-data statement: nothing under `snake-dqn-artifacts/t34-round0-study-01/` was opened. No Round 0, Path B or R1
  outcome is known to me. The only data file I opened is one excluded A5 v2 export
  (`t34-claim-a-20261020/smoke-05-final-9230b150/engine-export/worlds/a5-s0/world-000002.json`) to confirm key
  names that `seal.py` reads; I read no label content beyond counts and key names.
- Method: read every file in the diff vs `origin/main` (18 files, +3,620, no ratified file touched), the ratified
  `loss.py` / `data.py` / `interface.py` / `label.sh` / `make_manifest.py` they build on, and ran
  `pytest tests/test_t34_claim_a_loss.py tests/test_t34_claim_a_d0.py` (exit 0, light, 1 thread, nice 10).

## 1. Verification in code of §7a–§7d (what I checked, where)

| Requirement | Verified in code | Result |
|---|---|---|
| Objective by row type: KL(prev‖θ)+0.01 adv-MSE on anchor rows only; τ·CE(a\*)+margin-0.1 hinge on decisive rows only; no cross terms | `losses.py:64-103` (`_anchor_term` reuses ratified `exit_fit_loss` arithmetic with forbidden mask zeroed and a\*=argmax q_prev; `_decisive_term` has no KL/adv); `batch_tensors` (`harness.py:179-193`) sets `forbidden = forbidden & decisive`, `kl_anchor_only = kind==ANCHOR` | PASS |
| Separate, normalized streams; λ=1; τ/margin/adv weight = ratified | `losses.py:35-42, 83-86, 97-102` (each stream a mean over its own rows; `SplitConfig` defaults from ratified `LossConfig`) | PASS |
| `all_die_delay` excluded from both streams and every quota | `cache.py:66-73` → `KIND_DELAY`; `StreamSampler` (`harness.py:239-241`) draws only `KIND_DECISIVE`/`KIND_ANCHOR`; `split` masks `weight>0` and `ratified_epoch` gives delay rows weight 0 via ratified `sample_weights` (`data.py:62-64`). Only the `frozen_recipe` control still counts delay rows toward the 1:1 quota, deliberately (it is the ratified rule verbatim) | PASS |
| Anchors drawn from the full shadow-confirm pool | `run_d0.py:33` `REGISTERED_ANCHOR_STRIDE = 1`, manifest env `T34_D0_ANCHOR_STRIDE=1`, `timing.registered` false otherwise | PASS |
| Batch 1,024 = 128 decisive + 896 anchor; 4 blocks; dose ∈ {512,1024,2048,4096} | `Cell` defaults (`harness.py:357-358`), `D0_CORRECTED` checkpoints, `decide.py:48-56, 98-110` | PASS |
| Dose frozen from EXCLUDED A5 data only; no outcome leak | `inputs.pack` refuses any world not under `t34-exit-a5/v2/`, any carrier/rollout ≠ B8 s3, any sidecar mismatch (`inputs.py:70-80`); `unpack` is SHA-bound; `cache.build_world` requires hash-exact replay and recorded-Q identity (`cache.py:113-141`). The D0 job has no input path to Round 0 / Path B / R1 data. All `decide.py` constants are literals quoted in the spec | PASS |
| Tripwires and the 70% held-out bar unchanged | `decide.py` guards = ≤5% any length decile, boost ≤1.5×, one-action ≤90% (the FINAL_PLAN §8 set); the 70% held-out bar is explicitly left to the production fit on the R1 held-out bank (`decide.py:28-31`, spec §4.4 last paragraph). Nothing in the branch edits `research/t34_exit_b8_fit/` | PASS |
| Frozen-loss cells are controls, never selected | `decide.py:130, 163-164` (`CORRECTED_LOSSES` filter), `frozen_recipe`/`frozen_trainable`/`frozen_residual` in `D0_CELLS` and reported under `controls` | PASS |
| GO binding = governing `amendment-boost-floor/decision.json` only | `r1/make_manifest.py:75-122` checks path, schema `t34-round0-decision-amended/v1`, `governing: true`, `decision: GO`, `evidence_authority: TIER_1_DEVELOPMENT`, COMPLETE `phase_terminal.json` whose `files["decision.json"]` equals the decision SHA, and the terminal SHA in the ledger on `origin/main` (`git show origin/main:…`, no fetch). `make_chunk_manifest` and `seal` both refuse without that binding. Test `test_r1_binds_only_the_published_governing_round0_go` | PASS |
| R1 = 800 train + 200 whole-world held-out, disjoint namespaces, audited seeds | `plan.py` (8+2 shards × 100; `r1-labels-v4`), `SEED_AUDIT_R1.json` status PASS, full-history scan, 0 collisions vs Round 0 (242) / 0-D (272) / retention v2 (177), a0 and a1 audited; `verify_receipt` recomputes the seed digest. L4 label jobs run the single-process `device=cpu` branch of `label.sh` (no `T34_DEVICE` in the ratified label env), so the audited un-suffixed namespace is the one that seeds worlds (`trajectory.py:73`); the `/lane{n}` suffix branch is not taken | PASS |
| PRECONDITION_FAILED minimums (≥60 bearing worlds, ≥300 decisive rows), seal binds world sets, no held-out read by a fit | `plan.heldout_support`, `seal.py:120-138`; seal requires `--attest-no-retention-phase` and the GO binding. Key names (`decisive`, `beta`, `mix`, `teacher_v2.kernel`, `rollout_policy_sha256`) confirmed against a real A5 v2 export | PASS |
| Caps: L1 12 u, L2 60 u, per-chunk 12 u, raised T34 cap = sum of admitted lanes | `loss/make_manifest.py:27-66` (priced = timeout×2.0/60 ≤ cap ≤ 12; default 240 min → 8.0 u); `plan.admissible` (settled + open caps + new ≤ 60); ratified chunk cap 12 u at 360 min | PASS, with condition C4 (admission has no runtime caller; it is an operator step) |

Other checks: `trunk_forward` (`cache.py:182-190`) reproduces `Ego2sNet.forward` line for line, and the residual
trunk's `h` is verified transitively by the recorded-Q identity check. Pooled held-out predictions and indices are
built in the same ascending order (`harness.py:391, 465-473`). Folds are world-level, stratified on decisive-bearing
worlds, dealt by sha256 of the world id. Decile edges use anchor lengths only (no outcome). No weights are written.

## 2. Is D0's KILL rule decision-useful?

Yes, but only its third prong does real work, and the spec should say so:

- Prong 1 (free-Q < 70%) cannot fire for `split`/`split_tau2`: with free Q and no KL, CE+hinge flips every row. It is
  a sanity check (and the 99% eligibility rule is the same check, stricter).
- Prong 2 (in-sample ≥ 70% at any checkpoint) is weak for the trainable trunk: 4,096 × 128 draws over ~150 training
  decisive rows is ~3,500 passes, which a 1.6 M-parameter net will mostly memorize unless the anchor KL on ~127k
  anchor rows forbids it. It is a fair test for the residual trunk.
- Prong 3 (no trunk keeps the anchor guards for ≥ 512 updates) is the cheap stop: if correcting ~150 rows already
  breaks ≤5% decile agreement with B8 by 512 updates on excluded data, R1 with ~30× more decisive rows cannot do
  better at the same dose, and the projection line stops for 8 u instead of 120 u. That is what L1 is for.

So D0 PASS is weak evidence and D0 KILL is strong evidence. That asymmetry is correct for a pre-screen whose
positive case still has to pass the unchanged 70% held-out bar and tripwires in L3.

## 3. Answers to the open questions (binding)

The spec's §10 lists five questions, not six. I answer those five and the implicit sixth in §4.5 ("Order").

1. **Permissive diagnostic = in-sample training-fold decisive agreement ≥ 70% (as implemented).** Reason: the L1
   row says "even on the permissive diagnostic"; ~150 training rows per fold cannot support a 70% cross-fold bar,
   so a held-out requirement would KILL for data starvation, not for the loss. Held-out trend stays reported and
   selection-only.
2. **Dose frozen in optimizer updates (as proposed).** Reason: the tripwires bound anchor drift away from B8, which
   accrues per update, not per decisive pass; freezing passes would multiply R1's dose ~30× and break both the drift
   budget and the L3 cap. `decisive_passes_at_dose` must be reported next to the dose (it already is in
   `D0_DECISION.json`).
3. **L3 frozen-loss controls use the new streams and dose (as proposed).** Reason: a 2×2 must vary one factor per
   edge; the full ratified recipe is already known to be starved (design.md finding 2) and D0 keeps it as
   `frozen_recipe` for the record only. Note for the spec (C1): under the ratified shared-weight loss the stream
   control carries `sample_weight = 7.0` on decisive rows (`harness.py:442, 263-265`), which equals the split's
   per-stream means scaled by 0.875; that is the right balance and must be written down.
4. **No block selection in L3.** The student is the model at the frozen dose U\*; block ends are tripwire checks only
   (a breach → FIT_STOPPED, as ratified). The held-out bank is read once per block for the tripwires and once at U\*
   for the report; the 32-world probe runs once, at U\*. Reason: D0 freezes the dose, so there is nothing left to
   select; this removes the optimistic max-over-blocks bias on the tripwire bank without spending training worlds on
   a selection split. Any later registration that wants selection must add a training-side world split itself.
5. **Decisive rows drawn uniformly (as implemented).** Reason: D0 freezes the dose under this sampler and a later
   sampler change would void the transfer; the separate stream means already remove cross-type mass dependence;
   world concentration stays visible because `flip_world_weighted` is reported next to the pooled rate. The 70% bar
   stays row-pooled as ratified; the world-weighted rate is reported, not gated.
6. **(§4.5 Order) D0 may run as soon as this amendment is merged, before any Round 0 or Path B verdict.** Reason:
   its only input is the excluded A5 bank, the packer refuses anything else, and it emits no evidence about any
   student. It does not count as a Round 0-conditioned lane.

## 4. Non-blocking defects found (none can yield a wrong verdict, leak, overrun or integrity failure)

- D1. `decide.py:186-196`: when a corrected loss flips between 70% and 99% of free-Q rows, the eligibility rule
  (≥ 99%) makes every cell ineligible and the KILL reason string says "reaches 70% … in-sample", which is not the
  cause. Outcome is correct under the frozen rule; the text is wrong. Document or fix the string before D0 runs
  (string-only change; no rule change).
- D2. Spec §2.3's DAgger-age weight 0.5 is not implemented: `split` uses `sample_weight` only as a `> 0` mask
  (`losses.py:93-95`). R1 has only current-round rows, so no effect in L1–L3; an R2+ registration must implement and
  test it first.
- D3. `loss/make_manifest.py:29-31` sizes the report phase at "~43k CUDA updates"; the registered cell set is
  6×3×4,096 + 2×3×2,048 + `frozen_recipe` ≈ 86k updates, ~61k of them through the 1.6 M-parameter trunk at batch
  1,024. The 240-min timeout and 8.0 u cap still bound the spend inside the 12 u lane; a timeout is a FAILED with one
  ≤ 4 u rerun. The Mac preflight should report measured seconds per update so the operator can re-price the timeout
  (inside the lane cap) before enqueue if the extrapolation exceeds ~180 min.
- D4. `plan.admissible` has no production caller; lane admission is an operator step (condition C4).
- D5. The spec's §10 says five questions; the task brief said six. Recorded above.

## 5. Conditions (mechanical; no further review)

- C1. Before merge, spec-text-only edits: (i) §2.4/§4.2 state the 7.0 decisive `sample_weight` of the frozen-loss
  stream controls and its equivalence (D3 of §3 Q3 above); (ii) §2.3 marks the 0.5 DAgger-age weight "not
  implemented on this branch; required by any R2+ registration"; (iii) §10 replaced by the six answers in §3 of this
  judgment; (iv) §4.3 notes that free-Q in [70%, 99%) also ends in KILL via eligibility (D1), and the
  `decide.py` reason string is either fixed to say so or the note stands. No code semantics change.
- C2. D0 dispatch only from the merged main commit, manifest built by `research/t34_claim_a_loss/make_manifest.py
  --validate` with est 100 / timeout 240 / cap 8.0 u (or a re-priced timeout whose cap stays ≤ 8.0 u), after the
  `d0.sh` local-mode Mac preflight receipt (tiny input, timing, per-update seconds, no `/Users` deps) is attached to
  the CAMPAIGN_LOG row. At most one rerun, ≤ 4 u, only on PRECONDITION_FAILED or FAILED, never on a decision.
- C3. `D0_DECISION.json` is recorded in CAMPAIGN_LOG with its `report_sha256` and `registered_run: true` before any
  L3 registration; a decision with `registered_run: false`, or from any cell set / stride / fold count other than
  the registered one, is void.
- C4. Each R1 chunk dispatch record states `plan.admissible(settled, open_caps, 12.0) == True` with the numbers used
  and the live `colab usage` balance minus outstanding liabilities; manifests come from
  `research/t34_claim_a_r1/make_manifest.py --validate` with the governing GO binding embedded; the seal runs on the
  Mac with `--attest-no-retention-phase` and only after every chunk is terminal; no fit, screen or metric reads any
  world in `heldout_world_set_sha256`.
- C5. The L3 registration binds: the `D0_DECISION.json` SHA-256 (loss variant, lr, U\*, per trunk), answers Q1–Q6
  above (student at U\*, no block selection, updates as the unit, stream-rule controls, uniform decisive draws), the
  unchanged FINAL_PLAN §8 tripwires with the 70% held-out decisive bar row-pooled on the R1 held-out bank (world-
  weighted rate reported), and the L3 cap of 120 u. L3 fits still wait for the R1 seal (`SUPPORTED`) and the
  published Path B disposition.
- C6. Release and serving authority stay with the owner; nothing here changes the strict gate, survival band v2,
  serving qualification or P7.

Under C1–C6 the amendment implements §7a–§7d faithfully, the D0 KILL rule can stop the projection line for ≤ 12 u,
and no outcome data from any claim-bearing study can reach a frozen choice.
