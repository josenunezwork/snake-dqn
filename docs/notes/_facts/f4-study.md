# FACTS: F4 frozen-B8 residual dose study (registration → D256 → D1024 → R-A STOP_FUNDING → post-mortem)
Sources read: big-lever/round-26/JUDGMENT.md, big-lever/round-27/30/31/32/33/36 JUDGMENT.md (grep only), big-lever/exp/f4/{F4_REGISTRATION.md, judgment.json, JUDGMENT_D4096_START_20261024.md, JUDGMENT_D1024_FIT_TRANSPORT_20261024.md, HANDOFF_D1024_FIT_20261024.md + ADDENDUM_01/02, CLAUDE_V_STANDIN_D1024_FIT_REVIEW(_02).json, d256/d1024 preflight-*.json, d256-device-validation-03/validation.json, */pack.json, */inputs/life-allocation.json, canary receipt status fields, r-a-archive-census-01/census.json, r-a-astra-f9c08082-01/funding-look.json}, big-lever/exp/astra-20261024/{ACCEPT_F4_*.md, FIX_FIRST_F4_POSTMORTEM_*.md, ACCEPT_RA_PACKET_f9c08082_20261025.md, JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md, JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md, ACCEPT_F2_IMMUTABLE_SUCCESSOR_20261024.md, JUDGMENT_F2_POSTRUN_adef78ec_20261024.md}, repo research/bl_residual/{README.md, F4_REGISTRATION.md, F4_HANDOFF.md, HANDOFF.md, D256_*_HANDOFF_*.md, F2_I2_HANDOFF_20261024.md, I3_*_HANDOFF_20261024.md, ledger/*, ledger/F4_R_A_20261025/*, f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md}, research/bl_labels/D4096_I1_HANDOFF*.md, docs/research/final_run_20261019/CAMPAIGN_LOG.md (lines 157–309, 792–1037, 1336–1429), git show --stat 6cc4580c c13d8b28 d4290939 69b14c95 03bf2f40   (sheet written 2026-10-25 PT)

Path abbreviations: `AR` = /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever ; `LOG` = docs/research/final_run_20261019/CAMPAIGN_LOG.md ; `BR` = research/bl_residual (repo-relative).

## Verdict sequence (HOLD / ACCEPT / STOP, in order)
- [2026-10-23] Round-26 judge (Opus stand-in for Fable) ruled RATIFY-WITH-CHANGES on the F4 draft, pre-data — src: AR/round-26/JUDGMENT.md:1-12
- [2026-10-23 01:25 PT] F4 APPROVED (pre-data): judgment sha 0fe1a103…8362, registration sha 66388c79…2fd65, FitRecipe sha 49d12f48…3a5a, all 8 code edits at 8d6df609 (Codex-V PASS, 151 tests), approval artifact big-lever/exp/f4/judgment.json — src: LOG:157 | AR/exp/f4/judgment.json:1-7
- [2026-10-24 00:00 PT] D256 'prepare' step failed (Colab manifest lacked explicit device) → fixed at c2c82ff2 → prepare-02 PASS; canary bl-f4-d256-canary-02 dispatched — src: LOG:234
- [2026-10-24 00:08 PT] F4 D256 canary-02 INCOMPLETE (UPLOAD_CAP on factory-verified-ledger.json; no fit ran) → operator paused new Colab dispatches — src: LOG:236 | AR/exp/f4/d256-pack-02/canary-output-l4-01/receipt.json (status INCOMPLETE, failure_class UPLOAD_CAP)
- [2026-10-24 00:53 PT] F4 D256 metadata canary PASS (bl-f4-d256-metadata-canary-02, bundled inputs); main manifest d256-metadata-pack-02/main.json sha 69df54af…be4a sealed; registered D256 fit bl-f4-d256-metadata-fit-02 dispatched on L4 — src: LOG:240
- [2026-10-24 01:02 PT] Registered F4 D256 FIT COMPLETE (EXIT=0); outcome not opened — src: LOG:241
- [2026-10-24 01:20 PT] Round-27 T-EQ = FAIL (sealed, equiv-02) → D1024 uses OLD W6 fallback — src: LOG:242
- [2026-10-24 01:21 PT] D256 fit PUBLISHED and VERIFIED (local == origin; Sol f4publish 053b4c71; publish 6deb9846); no D256-only look — src: LOG:243
- [2026-10-24 01:47 PT] D1024 collection RUNNING on the Mac (bl-f0-d1024-old-eq02-rebind-01) — src: LOG:245
- [2026-10-24 09:15 PT] JUDGMENT_D1024_FIT_TRANSPORT: KEEP_QUALIFIED_UPLOAD_FOR_LABELS_MOVE_B8_TO_EXISTING_PUBLIC_URL — src: AR/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:1-3
- [2026-10-24] Claude-V stand-in D1024 fit review 01 = ACCEPT (0 blocking; at efa01291); review 02 = ACCEPT (at fa07bd75) → "D1024 fit pack READY" — src: AR/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW.json:63 | AR/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW_02.json:50 | LOG:291
- [2026-10-24 ≈11:30 PT] JUDGMENT_D4096_START: (A) ADMIT_D4096_OLD_W12; (B) ADMIT_SEALED_EARLY_COLLECTION; (F2) BUILD_NOW — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:1-3 | LOG:309
- [2026-10-24 12:35 PT] D4096 I1 build: Astra Codex-V re-review conditional ACCEPT; I3 fix 9aad2ce4; I2 one P1 left → sol-f2i2fix2 — src: LOG:1339
- [2026-10-24 13:36 PT] I2 (F2 tooling) Astra Codex-V ACCEPT at 92f26cc4 after four fix rounds; I3 L4-accounting fix da4dbf25 ACCEPT — src: LOG:1356
- [2026-10-24 16:13 PT] D1024 collection COMPLETE; final union published c55d3c66 (sha f5ec94cc…) — src: LOG:1365
- [2026-10-24 17:00 PT] D1024 fit HOLD for transport repair (excluded L4 canary -01 and authorized retry -02 both INCOMPLETE UPLOAD_CAP); sealed D4096 STARTED on the Mac — src: LOG:1366
- [2026-10-24 18:28 PT] Transport repair 0bd6f058 Astra ACCEPT (JUDGMENT_F4_TRANSPORT_REPAIR_REVIEW_0bd6f058, a634442e…), merged f628c1af — src: LOG:1391
- [2026-10-24 18:50 PT] Requalification acceptance-1792890832 REFUSE (transfer 207.3 s > 150 s) → Astra JUDGMENT_F4_REQUALIFICATION_FAILURE_0bd6f058: HOLD — real transport-performance failure — src: LOG:1392
- [2026-10-24 19:09 PT] ACCEPT_F2_IMMUTABLE_SUCCESSOR (recorded 2026-10-25T02:09:33Z) — src: AR/exp/astra-20261024/ACCEPT_F2_IMMUTABLE_SUCCESSOR_20261024.md:4-6
- [2026-10-24 19:37 PT] F2 post-run ACCEPT: COMPLETE, volume_struck = true (recorded 2026-10-25T02:36:53Z) — src: AR/exp/astra-20261024/JUDGMENT_F2_POSTRUN_adef78ec_20261024.md:1-13
- [2026-10-24 20:05 PT] Throughput-fixed transport 55439bff ACCEPT + R-A timing amendment (JUDGMENT_F4_TRANSPORT_55439bff_RA_TIMING, fcf67afc…) — src: LOG:1394
- [2026-10-24 20:12 PT] ACCEPT_F4_KERNEL_REPLACEMENT_PLAN_1792897885 (plan delta only; recorded 2026-10-25T03:12:29Z) — src: AR/exp/astra-20261024/ACCEPT_F4_KERNEL_REPLACEMENT_PLAN_1792897885.md:3-5
- [2026-10-24 20:45 PT] Requal #3 REFUSE (transfer 221.7 s); Astra JUDGMENT_F4_D1024_PROSPECTIVE_TRANSFER_EXCEPTION (f57080bd…): ADMIT D1024-only transfer bar ≤300 s, "no B6 now" — src: LOG:1395
- [2026-10-24 ~22:18 PT] ACCEPT_F4_D1024_TRANSFER_IMPLEMENTATION_de5bde2d → live requalification acceptance-1792902941 PASS → ACCEPT_F4_D1024_LIVE_REQUALIFICATION_1792902941 → ACCEPT_F4_D1024_PRECANARY_ATTEMPT04 → canary-04 PASS → ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db → registered fit bl-f4-d1024-fit-04 DISPATCHED — src: LOG:1396 | AR/exp/astra-20261024/ACCEPT_F4_D1024_TRANSFER_IMPLEMENTATION_de5bde2d.md:5 | AR/exp/astra-20261024/ACCEPT_F4_D1024_LIVE_REQUALIFICATION_1792902941.md:5 | AR/exp/astra-20261024/ACCEPT_F4_D1024_PRECANARY_ATTEMPT04_de5bde2d.md:5 | AR/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:5
- [2026-10-24 22:23 PT] Registered D1024 fit bl-f4-d1024-fit-04 PASS (365 s, stop verified); fit phase/binding published (merge 3395b8cb); R5 census published (924f77fa) — src: LOG:1403
- [2026-10-24 ~22:28 PT] R-A prepare (bl-f4-ra-prep-astra-924f77fa-01) FAILED (attempt-path bug: MAIN_OUTPUT "d1024-output-l4-01" vs actual attempt 04) — src: LOG:1403
- [2026-10-25 ~10:22 PT] ACCEPT_RA_PACKET_f9c08082 (packet acceptance, not a funding outcome; reviewed 2026-10-25T17:22:40Z) — src: AR/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:3-5
- [2026-10-25 10:23–10:40 PT] Look R-A bl-f4-ra-astra-f9c08082-01 ran once → verdict STOP_FUNDING — src: LOG:1404 | AR/exp/f4/r-a-astra-f9c08082-01/funding-look.json:1
- [2026-10-25] Astra (independent judge; "Fable stopped") JUDGMENT_RA_STOP_FUNDING: "UPHOLD STOP_FUNDING" — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:1-7
- [2026-10-25 10:50 PT] B6 executed: D4096 sealed d4096-not-run-by-rule.json (terminal_label STOP_FUNDING, terminal_stage r-a); "F4 closes" — src: LOG:1404 | research/bl_residual/ledger/F4_R_A_20261025/d4096-not-run-by-rule.json:1
- [2026-10-25 16:50 PT] Astra JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025 (8c4480a5…): ALLOW ONE DIAGNOSTIC EXPORT; STOP_FUNDING REMAINS FINAL — src: LOG:1427 | BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:1-5
- [2026-10-25 17:09:41 PT] FIX-FIRST on post-mortem exporter (impl a31a1d54, execution pin 7f3c0857; recorded 2026-10-26T00:09:41Z) — src: AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_7f3c0857_20261026T000941Z.md:1-7
- [2026-10-25 17:24:06 PT] FIX-FIRST re-review at pin 1fe60c99 (handoff 03bf2f40; recorded 2026-10-26T00:24:06Z) — src: AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:1-5
- [2026-10-25 17:29:51 PT] ACCEPT_F4_POSTMORTEM_69b14c95 (recorded 2026-10-26T00:29:51Z), conditional on F5 ADOPTED and published before diagnostic exposure; "F4 remains STOP_FUNDING" — src: AR/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:1-5
- [2026-10-25 17:40 PT] Log row: exporter ACCEPTED (gated); F5 adoption HOLD on R10a annex gaps — src: LOG:1429 | git 6cc4580c

## Registration (round-26 judgment, F4_REGISTRATION.md)
- [2026-10-23] Round-26 judge was an "Opus stand-in for Fable (Fable rate-limited)", written ≈ 00:45 PT, light reads only — src: AR/round-26/JUDGMENT.md:3-5
- [2026-10-23] Judge verified defects D1–D10 in the draft recipe, including constant pair weights (D1) and growth pairs rewarding dying actions (D2) — src: AR/round-26/JUDGMENT.md:20-34
- [2026-10-23] The draft's only null rule (UB90(G_dose) ≤ 0) was judged to fire ≈10% of the time under a true zero effect — src: AR/round-26/JUDGMENT.md:14-15
- [2026-10-23] Registration = bytes between marker lines; judge-computed sha256 66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65 (129 lines, 0 UNRATIFIED) — src: AR/round-26/JUDGMENT.md:46-57,198-200
- [2026-10-23] Binding sequence: Sol implements §4 (8 code changes) → Codex-V verifies → operator writes judgment.json with registration + recipe sha; no label fitted before step 3 — src: AR/round-26/JUDGMENT.md:59-62,203-227
- [2026-10-23] judgment.json: decision APPROVE, judgment_sha256 0fe1a103…8362, registration_sha256 66388c79…fd65, recipe_sha256 49d12f486bffa455adcedbc318e88f1524f5c502894d0c1ace47daa84b9d3a5a — src: AR/exp/f4/judgment.json:1-7
- [2026-10-23] Question (R1): does growing selected rescue-bearing natural lives 1,024 → 4,096 make the same frozen-B8 residual stronger in fresh paired H5000 gameplay, and does the top dose beat bare B8 s3 (sha 3e649783…eb4f8) — src: AR/exp/f4/F4_REGISTRATION.md:8-11
- [2026-10-23] Primary endpoint: per-world full-horizon terminal-hero mass integral over H5000 (dead frames zero); inference about three fixed checkpoints under one fit seed — src: AR/exp/f4/F4_REGISTRATION.md:9-11
- [2026-10-23] Unit = rescue-bearing life (one natural hero death with ≥1 exact H300 survivor), clustered by parent; folds by sha256 of `bl-residual-f3-split-v1/<parent>` mod 5, fold 0 held out — src: AR/exp/f4/F4_REGISTRATION.md:14-16
- [2026-10-23] Reserve fixed at D256: first 16 fold-0 S-Watch6 rescue lives then first 48 other fold-0 rescue lives, ≤4 reserve lives per parent — src: AR/exp/f4/F4_REGISTRATION.md:17-18
- [2026-10-23] Doses D256/D1024/D4096 = 192/960/4,032 fit lives (nested fold 1–4 prefixes) plus the 64-life reserve — src: AR/exp/f4/F4_REGISTRATION.md:19-20 | BR/README.md:35-37
- [2026-10-23] Frozen recipe (R3): B8 s3 frozen; zero-init residual Linear(256,6) on b8_features fused into B8 head; Adam fp32 lr 0.001, 2,000 updates, seed 3403, batch 256 (128 doom + 128 ordinary, life-uniform then row-uniform) — src: AR/exp/f4/F4_REGISTRATION.md:25-31
- [2026-10-23] Losses: survival softplus(Q_die − Q_surv) pair weight 1; growth among exact survivors weight min(g,4)/4 normalised by pair count; retention KL(softmax(Q_B8)‖softmax(Q_corrected)) coefficient 1; residual L2 1e-4; no prevalence weights — src: AR/exp/f4/F4_REGISTRATION.md:32-38
- [2026-10-23] Tripwires (R4): non-finite/illegal/>90% one-action share → KILL_FIT any dose; fit-drift (>5% disagreement per length decile, boost >1.5× B8, reserve retention <0.70) binding for D4096 only on a 32-world probe bank (11/11/10) — src: AR/exp/f4/F4_REGISTRATION.md:41-48
- [2026-10-23] Look R-A = D256 and D1024 jointly after D1024 frozen; Look R-B = D4096; each create-only, once — src: AR/exp/f4/F4_REGISTRATION.md:51-52
- [2026-10-23] Life-macro rescue retention = per reserve life, fraction of decisive rows where served action is an exact survivor, mean over lives — src: AR/exp/f4/F4_REGISTRATION.md:53-54
- [2026-10-23] Harm screen H = unweighted count of archive rows where served action changes from B8 and (selected exact-dies and B8 exact-survives) or (selected or B8 outcome not EXACT); pass iff H = 0 — src: AR/exp/f4/F4_REGISTRATION.md:57-58
- [2026-10-23] R6.3 funding: if F2 does not strike volume, buy D4096; if F2 strikes (or is unpublished before R-A), buy only if mean Δretention(D1024−D256) ≥ 0.05, parent-cluster bootstrap LB90 > 0 (10,000 resamples, seed 3403) and D1024 H = 0; else STOP_FUNDING (terminal, no gameplay, recipe closed) — src: AR/exp/f4/F4_REGISTRATION.md:67-70
- [2026-10-23] R6.4 D4096 cost projection = measured units per selected rescue life × 3,072 × 1.2 vs D4096 base cap — src: AR/exp/f4/F4_REGISTRATION.md:71-72
- [2026-10-23] Caps (R7, base/effective Colab units): D256 tranche 40/80; D1024 increment 120/240; D4096 increment 480/960; fits whole series 10/20 (L4; A100 only after ≥5× measured speed-up); gameplay/probe/reserve looks Mac 0 u — src: AR/exp/f4/F4_REGISTRATION.md:77-79
- [2026-10-23] Gameplay screen (R8): 128 H5000 worlds per mix × 3 mixes, arms B8 s3/D1024/D4096; Holm two-hypothesis family alpha 0.10; point bars G_top ≥ +40, G_dose ≥ +20; four guards; Watch6 product 128 sessions × 14,400 frames only if every H5000 bar passes — src: AR/exp/f4/F4_REGISTRATION.md:82-102
- [2026-10-23] Ten ordered labels: INVALID, INCOMPLETE, INSUFFICIENT, KILL_FIT, STOP_FUNDING, STOP_ADMISSION, KILL_DOSE_CLOSED, KILL_DOSE_FUTILE, GO_TOP_DOSE_DEV, INCONCLUSIVE; labels 4–8 close the recipe/feature/dose-range lane — src: AR/exp/f4/F4_REGISTRATION.md:105-118
- [2026-10-23] Registered judgment prior P(GO) ~ 0.10; 128/mix is "a fixed falsifier, not an 80 %-powered trial" — src: AR/exp/f4/F4_REGISTRATION.md:121-124
- [2026-10-23] Pre-data freezing: any change to the registered block = new registration → advisors → judge, before data — src: AR/round-26/JUDGMENT.md:62
- [2026-10-23] F4 implementation handoff: 151 tests passed in 18.60 s; Codex-V receipt CODEX_V_F4_REVIEW.json; synthetic fixtures cover all ten R9 labels; no real labels opened — src: BR/F4_HANDOFF.md:4-9,18-21,64-66,90
- [2026-10-23] Repo copy research/bl_residual/F4_REGISTRATION.md is byte-identical to big-lever/exp/f4/F4_REGISTRATION.md (sha 66388c79…fd65) — src: BR/README.md:3-5 | AR/exp/f4/F4_REGISTRATION.md:1
- [2026-10-23] Predecessor F3 package (pin 560e6526) was "F3 built; F4 unratified" — src: BR/HANDOFF.md:3-8

## Science pins (111c4781 vs bf8ac97d)
- [2026-10-23] D1024 was originally planned at speed pin bf8ac97dc66d1b4515e4c5510db3ad5886fa695d with wrapper 3c5fc403 — src: LOG:865-867
- [2026-10-24 00:45 PT] Round-27 speed-equivalence harness bug fixed at 111c4781 (re-run equiv-02) — src: LOG:239
- [2026-10-24] Owner relayed Operator Claude's YES to rebind a fresh D1024 control package to repaired speed pin 111c47818a097c57451386ef4f8a7009616f8f77, because the sealed equiv-02 T-EQ FAIL binds 111c4781 and original control bf8ac97d cannot admit it — src: LOG:993-997
- [2026-10-24] Fresh D1024 package d1024-old-w6-equiv-02-pack-01, control sha 9952387d…37f7; only source_commit and mac_gate.source_commit changed; namespace (still containing bf8ac97d), 2,160 parents, 144,000 s (40 h) cap, W6, calibration 61900af2 unchanged — src: LOG:999-1007
- [2026-10-24] Operational pin 587ff73e (fixes mac_gates.LEGACY_FACTORY_COMMIT AttributeError); science remains frozen 111c4781, collector OLD 61900af2 at W6 — src: LOG:1016-1021
- [2026-10-24] D4096 collector is OLD (61900af2 closure, science pin 111c4781, as D1024) for the whole tranche; NEW is not re-qualified; the original sealed pack-01 bound NEW bf8ac97d — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:22,42-44
- [2026-10-26 file] F5 annex records original D256 commit 61900af2 and D1024 commit 111c4781 — src: AR/exp/astra-20261024/F5_R10A_GAP_CLOSURE_ANNEX_20261026.md:15

## D256 tranche
- [2026-10-23 23:18 PT] D256 collection COMPLETE: 8 attempts, 1,566 sealed units, 864 parents, cumulative factory 16,333.70 s (4.54 h), 0 thermal trips; union ledger sha 30919209…eb809 — src: LOG:232 | LOG:794-810. Collection mechanics → see label-factory.
- [2026-10-23] "D256 final union publication — outcome-blind": published before any F4 partition prep; recipe 49d12f48 and registration 66388c79 unchanged; no fit/look run — src: LOG:792,814-819
- [2026-10-23] F4 D256 post-P1 preparation: union pushed 483c8f56; F4 implementation 6549f84d; qualified source pin e01f3a96 using CLI tree d0d92bdb; Codex-V static receipt bf64e7ec…6cae — src: LOG:824-845
- [2026-10-23] D256 fit reservations: canary 0.5/1 u, main 2/4 u (+reserves), total 6 u within whole-fit-series 10/20 u — src: LOG:855-858
- [file 2026-10-24] d256-preflight-01 PASS at e01f3a96; d256-preflight-02 PASS at c2c82ff2; both fits_run false, study_outcomes_opened false — src: AR/exp/f4/d256-preflight-01.json:1 | AR/exp/f4/d256-preflight-02.json:1
- [2026-10-24] Device fix: frozen F4 source c2c82ff2 declares cuda:0 in both manifests; intermediate pin 1629fb25 superseded; failed d256-pack-01 kept unchanged — src: BR/D256_DEVICE_FIX_HANDOFF_20261024.md:3-16
- [file 2026-10-24] d256-device-validation-03: canary PASS (cuda:0), main-template EXPECTED_HOLD (needs SHA-bound family canary), dispatched false — src: AR/exp/f4/d256-device-validation-03/validation.json:1
- [2026-10-24 00:45 PT] Input bundling 13af8afa: F4 staged files 18 → 5 (one metadata tar per job) — src: LOG:239
- [file 2026-10-24] Metadata-only packs: d256-metadata-pack-01 (commit e655caac) and -02 (commit 6313fe8f) repack frozen d256-pack-02 inputs (16 → 3 staged files), outcome_reporting none — src: AR/exp/f4/d256-metadata-pack-01/pack.json:1 | AR/exp/f4/d256-metadata-pack-02/pack.json:1
- [2026-10-24] Metadata post-fit handoff: continues only d256-metadata-pack-02 at fit pin 6313fe8f with publish-fit then stage-d1024 — src: BR/D256_METADATA_POSTFIT_HANDOFF_20261024.md:3-15
- [2026-10-24] D256 fit checkpoint fit-0000.pth, 6,588,730 bytes, sha 103d02e9…0360 (D256 fit phase ledger) — src: BR/ledger/F4_D256_FIT_PHASE_20261023.json:1 | AR/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:12
- [2026-10-24] D256 fit main ran 676 s wall in total — src: AR/exp/f4/HANDOFF_D1024_FIT_20261024.md:39
- [2026-10-24] D256 main fit receipt (c038eb8f) transfer total 260.0 s including three failed 120.0 s exec control calls; D256 canary transfer 72.6 s — src: AR/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:17-23 | BR/ledger/F4_D256_METADATA_FIT_BINDING_20261024.json:1
- [file 2026-10-24] D256 allocation: 192 train lives / 43 distinct parents, all S-Watch6 (0 S-gate); reserve 64 lives / 28 parents (49 S-Watch6, 15 S-gate); 590 available fit rescue lives, 398 unused at dose 256 — src: AR/exp/f4/d256-metadata-pack-02/inputs/life-allocation.json:1

## D1024 tranche (collection, transport, fit)
- [2026-10-24 01:47 PT] D1024 ran on the Mac as registered OLD W6 fallback after T-EQ FAIL; check CHECKED (W6, deadline 144,000 s) — src: LOG:245
- [2026-10-24 04:12 PT] D1024 PREEMPTED at 03:50 PT (7,585 s, 62 parents done, 2,092 pending) by a nice-0 Colab supervisor after the renicer died; resumed 04:24 PT (bl-f0-d1024-resume-old-eq02-01 @7b7b4ca2, 136,415 s cap remaining) — src: LOG:252-253
- [2026-10-24 06:25 PT] D1024 cap projection ON_TRACK: 4.15 h used, 124/2,160 parents; projected total 13.3 h (pessimistic 16.0, extreme 21.5) — src: LOG:267
- [2026-10-24 11:45 PT] D1024 at 609/2093 parents after 7.2 h, ~280 parents/h → projected finish ≈ 17:00 PT — src: LOG:1336
- [2026-10-24 16:13 PT] D1024 COMPLETE (2,092 parents; pending 0; lease released) — src: LOG:1365
- [file 2026-10-24] D1024 allocation: 960 train lives / 372 parents (753 S-Watch6, 207 S-gate); same 64-life reserve; 2,260 available fit rescue lives, 1,300 unused at dose 1024 — src: AR/exp/f4/d1024-pack-04/inputs/life-allocation.json:1
- [2026-10-24] D1024 fit handoff (code efa01291; impl 91e25453) built final union, fit pack, F4_DOSE canary, R5 census and R-A packet tooling; PREPARED_NOT_RUN — src: AR/exp/f4/HANDOFF_D1024_FIT_20261024.md:1-24
- [2026-10-24] D1024 fit caps: excluded canary 0.5/1 u, 10/20 min; main 2/4 u, 60/120 min; prior series liability 7.2 u + 6 u = 13.2 u, 6.8 u uncommitted — src: AR/exp/f4/HANDOFF_D1024_FIT_20261024.md:26-37
- [2026-10-24] Transport ruling: per-file model projects clean canary transfer ≈110 s (B8 via URL) vs 150 s bar; refused publishing label data, Drive, splitting, Mac-side fit, raising the bar — src: AR/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:1,25-28
- [2026-10-24] Addendum 01 (pin 48056754): only b8.pth becomes HTTPS row (6,562,005 B, sha 3e649783…); 198 tests passed — src: AR/exp/f4/HANDOFF_D1024_FIT_20261024_ADDENDUM_01.md:3-18
- [2026-10-24] Addendum 02 final pin fa07bd75 (48056754 → b8416c00 → fa07bd75) supersedes Addendum 01 pin; fixes Claude-V review-01 L1/L3/L4 lows — src: AR/exp/f4/HANDOFF_D1024_FIT_20261024_ADDENDUM_02.md:3-16
- [file 2026-10-24] d1024-preflight-01 PASS at c55d3c66; -02 PASS at 6deda139; -04 PASS at de5bde2d; all fits_run false — src: AR/exp/f4/d1024-preflight-01.json:1 | AR/exp/f4/d1024-preflight-02.json:1 | AR/exp/f4/d1024-preflight-04.json:1
- [2026-10-24 16:33 PT] Fit prepare-01 at c55d3c66 refused in clean-venv import preflight; Sol 3e6a1f0d + runbook suffix 6deda139 → preflight-02 PASS — src: LOG:1365
- [2026-10-24 17:00 PT] Excluded L4 canary -01 and authorized retry -02 (JUDGMENT_F4_D1024_CANARY_MECHANICAL_RETRY dc7bb384…) both INCOMPLETE UPLOAD_CAP (control-call stalls ~240 s) — src: LOG:1366 | AR/exp/f4/d1024-pack-02/canary-output-l4-01/receipt.json | AR/exp/f4/d1024-pack-02/canary-output-l4-02/receipt.json
- [2026-10-24 17:00 PT] Astra JUDGMENT_F4_TRANSPORT_REPAIR_D4096_B1 (03f74bee…): Q1 ADMIT qualified L4 transport repair; Mac fit stays unapproved (R7 L4, R11 qualified dispatch) — src: LOG:1366 | LOG:1368-1375
- [2026-10-24] Transport repair branch sol-colab-transport-repair-20261024: 30 s control calls, 60 s recovery within 300 s staging; 150 s bar unchanged; tests 765 passed / 7 failed / 3 collection errors — src: LOG:1370-1378
- [2026-10-24 20:05 PT] 55439bff: 2 parallel chunk workers, heartbeat renewal; offline replay 91 s (stress 128 s) vs 150 s — src: LOG:1394
- [2026-10-24 20:45 PT] Requal #2 L4 arm failed kernel readiness; requal #3 transfer 221.7 s (~200 KB/s Contents path), staging 235.5 s — src: LOG:1395
- [2026-10-24] Transfer-exception implementation de5bde2d (transport tree c1484312): L4 ≤300 s for exact F4 D1024 inventory only; CPU/other lanes keep 150 s; focused tests 196 passed — src: AR/exp/astra-20261024/ACCEPT_F4_D1024_TRANSFER_IMPLEMENTATION_de5bde2d.md:9-26
- [2026-10-24] Live requalification acceptance-1792902941: all 12 required gates OBSERVED; CPU transfer/staging 44.6/55.1 s, L4 48.0/57.5 s; CPU 0.08 and L4 1.54 u/h — src: AR/exp/astra-20261024/ACCEPT_F4_D1024_LIVE_REQUALIFICATION_1792902941.md:13-16
- [2026-10-24] Pre-canary attempt-04: pack d1024-pack-04 sha ee84de22…; union f5ec94cc…; training NPZ 6906bffc… 45,397,643 B; control d55e4665…; 128.8 s / 50,014,845-byte projection diagnostic only — src: AR/exp/astra-20261024/ACCEPT_F4_D1024_PRECANARY_ATTEMPT04_de5bde2d.md:12-21
- [2026-10-24] Canary-04 PASS: 600.79 s, transfer 136.40 s ≤ 300, staging 183.58 s ≤ 300; excluded synthetic canary runs 3 updates and never opens train-1024.npz — src: AR/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:12-17 | AR/exp/astra-20261024/ACCEPT_F4_D1024_PRECANARY_ATTEMPT04_de5bde2d.md:19
- [2026-10-24] Pre-fit ACCEPT: sealed main.json sha 9c2b88db…; one fit-0001 checkpoint, "Never retry or refit"; series liability 9.6 u + 6 u = 15.6 ≤ 20, remaining 4.4 — src: AR/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:5,11,24-25
- [2026-10-24] D1024 fit phase ledger sha 1de13756…; binding sha b44ba54e…; fit pin de5bde2d; checkpoint fit-0001.pth 6,588,730 B sha da1d2656…5093 — src: LOG:1399 | BR/ledger/F4_D1024_FIT_PHASE_20261024.json:1 | BR/ledger/F4_D1024_FIT_BINDING_20261024.json:1
- [2026-10-24] GPU label-path ruling: F4 R7 A100 cached-fit measurement NOT_FUNDED (D256 fit 676 s mostly overhead); D1024 fit stays on L4 — src: LOG:293

## What the fit measured (recorded result labels)
- [2026-10-24] Each fit produced one final fused checkpoint under the frozen recipe; no fit outcome was opened before R-A; unconditional fault tripwires did not stop the path (D256 and D1024 fits both completed and proceeded to R-A) — src: LOG:241,1403 | AR/exp/f4/F4_REGISTRATION.md:64-66
- [2026-10-25] Fits' only registered read was Look R-A (life-macro retention on the 64-life reserve and archive harm screen), "admission and funding only; never efficacy evidence" — src: AR/exp/f4/F4_REGISTRATION.md:50-58
- [2026-10-25] No gameplay, probe or R-B ran; skipped units are NOT_RUN_BY_RULE — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:39

## R5 archive census
- [2026-10-24] Census r-a-archive-census-01 frozen before Look R-A: receipt sha 44577f8a…30f0, table sha 34ad5395…7b4d, schema bl-residual-natural-archive/v2, outcome_reporting none — src: LOG:1402 | AR/exp/f4/r-a-archive-census-01/census.json:1
- [2026-10-24] Census counts: 5,936 rows, 560 natural rows, 71 eligible lives (3 non-rescue), 29 parents, 64 reserve lives, reserve max allocation_index 321000000 — src: AR/exp/f4/r-a-archive-census-01/census.json:1
- [2026-10-25] Packet reviewer independently recomputed shared 64-life reserve, ancestry and 71 eligible lives — src: AR/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:14

## F2 (volume test) and immutable successor
- [2026-10-24] F2 pinned spec: T5 table, 1,481 decisive rows / 269 worlds; A2 c0 frozen-linear D300 at 25% and 100%; Δ = mean survived(100%) − survived(25%); one-sided LB90 world-cluster bootstrap 10,000 resamples seed 3403; volume_struck = (LB90 ≤ 0) — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:120-133
- [2026-10-24] F2_I2 handoff: f2_inputs.py metadata-only T5 ancestry check; any count other than 1,481/269 is INVALID and strikes volume — src: BR/F2_I2_HANDOFF_20261024.md:8-16
- [2026-10-24] First run-sealed refused pre-lock (input ledger bound pre-hardening f2.py; look not consumed) — src: LOG:1393
- [2026-10-24] JUDGMENT_F2_IMMUTABLE_SUCCESSOR (Astra; sha f8671bc7…): "ADMIT ONE IMMUTABLE SUCCESSOR, FIX FIRST", path (b): one create-only successor ledger f2-inputs-20261024-r2.json (ac0d32c2…) and one-line INPUT_LEDGER edit; statistic block 23c86e2b… unchanged — src: AR/exp/astra-20261024/JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md:1-37 | AR/exp/astra-20261024/ACCEPT_F2_IMMUTABLE_SUCCESSOR_20261024.md:17-21
- [2026-10-24] Successor ruling: F2 must be published, verified, reviewed before R-A and before 2026-10-25 18:00 PT; any failure → strikes branch — src: AR/exp/astra-20261024/JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md:51-54
- [2026-10-24 19:24:24 PT] F2 result published (push 47679069356, adef78ec, 2026-10-25T02:24:24Z) — src: AR/exp/astra-20261024/JUDGMENT_F2_POSTRUN_adef78ec_20261024.md:57-61
- [2026-10-24] F2 result: paired Δ(100%−25%) = −0.02903443619176232, one-sided LB90 = −0.04213373082537578, 1,481 rows / 269 worlds → volume_struck = true; receipt sha b7d43e66…6842 — src: AR/exp/astra-20261024/JUDGMENT_F2_POSTRUN_adef78ec_20261024.md:13-18,43-44

## D4096 (sealed, never opened)
- [2026-10-24] Ruling (B) defines "purchase" in R6.3 as committing D4096 to fit/probe/R-B/gameplay or any Colab unit, not outcome-blind Mac collection — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:83-85
- [2026-10-24] Conditions B1–B6: start after D1024 COMPLETE and fit Colab jobs terminal/HOLD; B2 Mac-h projection ≤130 h; attempts carry funding_state PENDING_SEALED; no Colab-unit D4096 work before PASS; census and R-A preempt the factory; B6 discard on STOP_FUNDING etc. — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:97-112
- [2026-10-24] R-A timing rule: if sealed D4096 started by Sun 10-25 06:00 PT, R-A runs at earlier of verified F2 or Sun 10-25 18:00 PT; never re-run R-A after F2 — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:137-144
- [2026-10-24] Defect named: no producer for collection/projection receipts and job.py `projected > 0` rejects 0-u Mac placement; I3 to fix by Mon 10-26 23:59 PT — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:19-20,163-165
- [2026-10-24 11:45 PT] I1/I2/I3 commissioned to Sol (sol-d4096i1, sol-f2i2, sol-receiptsi3) after old implementer agents died — src: LOG:1336
- [2026-10-24] D4096 I1 pack d4096-old-equiv-02-pack-01 binds science pin 111c4781, collector OLD 61900af2, allocation sha 74b1c5db…, cap 468,000 s; 7,020 fold-1–4 rows occupy the 8,792-index span — src: research/bl_labels/D4096_I1_HANDOFF.md:8-15
- [2026-10-24 12:10 PT] Owner chose option B: tie-tolerant L4 admission; D4096 described as nested mixed-platform (owner decision 003399e7…) — src: LOG:1338 | research/bl_labels/D4096_I1_HANDOFF_FIX_FIRST_20261024.md:8-14
- [2026-10-24] I3 FIX-FIRST: job.py zero-cost admission when measured charges are zero requires projection exactly zero plus Mac/B2 checks — src: BR/I3_FIXFIRST_HANDOFF_20261024.md:1-16
- [2026-10-24 17:00 PT] B1 satisfied; B2 Mac-h projection sealed 66.6 h ≤ 130 h; D4096 dispatched (mac-bl-f0-d4096-old-w12-01, pin c2681b0d), funding_state PENDING_SEALED; actual placement Mac-only (L4 lane closed for October) — src: LOG:1366
- [2026-10-24 16:55:00 PT] Earliest D4096 start receipt epoch 1792886100.35 — src: AR/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:19
- [2026-10-25] Seven D4096 attempt-start receipts are bound in the funding look — src: AR/exp/f4/r-a-astra-f9c08082-01/funding-look.json:1
- [2026-10-25] D4096 output-mac-07 sealed PREEMPTED; cumulative wall 18,267.14 s recorded as sunk operational time — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:27
- [2026-10-25] d4096-not-run-by-rule.json: terminal_label STOP_FUNDING, terminal_stage r-a, unrecoverable false, judgment ccf1e0cd…, ledger copy sha f4d535e2… — src: research/bl_residual/ledger/F4_R_A_20261025/d4096-not-run-by-rule.json:1 | BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:17
- [2026-10-25] Any non-F4 use of the D4096 corpus needs a NEW pre-data ruling naming it before any byte is read — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:43 | AR/exp/f4/JUDGMENT_D4096_START_20261024.md:111-112

## Look R-A and STOP_FUNDING
- [2026-10-25] R-A prepare fixes: e238b337 (attempt-04 output path from sealed binding), f9c08082 (ra_guard spaced-JSON vs compact-JSON hashes), each Astra-reviewed — src: LOG:1404
- [2026-10-25] Packet r-a-astra-f9c08082-01/command-packet.json sha 443ad83f…dd8c, schema bl-residual-r-a-command-packet/v3, source pin f9c08082b4e298a3d6666e7820e49c7973a16b99; 34 source hashes match — src: AR/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:9-11
- [2026-10-25] Packet acceptance required B5 (D4096 drained, Mac exclusive) and a hard cutoff Sunday 2026-10-25 18:00 PT — src: AR/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:21-23
- [2026-10-25] R-A runner birth 10:23:09 PT; claim epoch 1792949988 = 10:39:48 PT; output 10:39:57 PT; JOB_ENDED 10:40:53 PT — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:23
- [2026-10-25] funding-look.json sha 4b889719…c8fe: mean_delta_life_retention 0.06621223984499343, parent-cluster bootstrap lower90 0.048454798628559156 (10,000 resamples, seed 3403), harm_screen_d1024_pass false, verdict STOP_FUNDING — src: AR/exp/f4/r-a-astra-f9c08082-01/funding-look.json:1 | AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:15
- [2026-10-25] Exact ruling wording: "UPHOLD the registered STOP_FUNDING result under F4_REGISTRATION R6.3 and R9. The published F2 strikes volume." Retention 0.0662 passes ≥0.05; LB90 0.0485 passes >0; "D1024 harm screen fails: H >= 1, against H=0." — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:7
- [2026-10-25] Judge: Astra, independent judge ("Fable stopped"); ruling sha be1516a9…eb05 — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:3 | BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:18
- [2026-10-25] Exact H "NOT RECOVERABLE FROM THE RETAINED RECEIPT": funding.py exports only harm_screen_d1024_pass=false; recorded as a reporting defect; no retry/refit/relook to repair it — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:9
- [2026-10-25] funding-look f2_disposition string "UNPUBLISHED_APPLY_STRIKES_BRANCH" is the code's common label; actual branch STRIKES with published F2 (f2_timing_eligible true) — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:20
- [2026-10-25] One canonical R-A claim F4_R_A_845eaa5b….lock.json (sha 3d5a9ab6…) and one funding-look.json exist — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:22
- [2026-10-25] Consequences: no further F4 labels, heads, seeds or refits; D4096 fit/probe/R-B/gameplay NOT_RUN_BY_RULE; R6.4 cannot bypass R6.3; B8 serving unchanged — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:39
- [2026-10-25] Receipt, packet, ruling and seal published under research/bl_residual/ledger/F4_R_A_20261025/ — src: LOG:1404
- [2026-10-25] Post-F4 contingency named by the ruling: round-30 R30-0/R30-1, then conditionally R30-2 → see r30-r32-f5-successors — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:43

## Post-mortem (harm diagnostic export)
- [2026-10-25 16:50 PT] Owner: "gave up too easy"; Astra advisory: F4 stopped correctly but the retention signal (0.066 / LB90 0.048) merits a distinct successor — src: LOG:1427
- [2026-10-25] Exception scope: one aggregate export per dose over natural rows: C (harmful_exact), U (harmful_unknown) split into three disjoint parts, H = C + U, four margin bins [0,.01),[.01,.1),[.1,.5),[.5,∞), all 29 parents — src: BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:62-67,76
- [2026-10-25] Exception forbids invoking funding.main/evaluate, new bootstrap or verdict, new inferential statistics; Mac CPU nice 15, one thread, 30/60 min envelope, once-only, publish-before-read — src: BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:60,119-123
- [2026-10-25] All 29 census parents (16 d256-mac-gate-frozen-*, 13 d256-mac-watch6-*) and descendants are DEVELOPMENT_FOREVER, excluded from successor validation — src: BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:85-115
- [2026-10-25] FIX-FIRST 7f3c0857: P1 publish-before-inspection incomplete (f4_postmortem_result.py:232-289); P1 hard 60-min ceiling relied on Python SIGALRM (probe saw 0.139 s callback delay for a 0.020 s alarm) — src: AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_7f3c0857_20261026T000941Z.md:11-13
- [2026-10-25] FIX-FIRST 1fe60c99: both prior blockers repaired; new P2 stale-PID race in f4_postmortem_watchdog.py:130-149 (SIGKILL to an already-reaped PID) — src: AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:5-11,17-18
- [2026-10-25] ACCEPT 69b14c95: signals blocked around waitpid (watchdog.py:133-141); six mocked signal-at-reap cases FAIL on 1fe60c99 and PASS on 69b14c95; 16 watchdog + 65 focused tests reported — src: AR/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:9-13
- [2026-10-25] Approved operator command: dispatch.sh --mac f4-postmortem-harm-20261025-01 at 69b14c95, only after F5 registration ADOPTED and published — src: AR/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:17-23
- [as of 2026-10-25 17:40 PT] Exporter not run: gated on F5 adoption, which was HOLD on R10a annex gaps — src: LOG:1429
- [2026-10-25] Commits: 03bf2f40 (17:20:11 PT) repaired execution pin; 69b14c95 (17:27:04 PT) signal-safe watchdog; d4290939 (17:27:32 PT) signal-safe execution pin; c13d8b28 (17:34:26 PT) F5 R10a HOLD annex (684 provisional parents); 6cc4580c (17:38:28 PT) log row for exporter ACCEPT and F5 annex gap closure — src: git 03bf2f40 | git 69b14c95 | git d4290939 | git c13d8b28 | git 6cc4580c
- [2026-10-25] F5 annex/gap closure details → see r30-r32-f5-successors — src: LOG:1429

## Superseded / conflicts
- [2026-10-23] D1024 at speed pin bf8ac97d with W12 via 3c5fc403 SUPERSEDED BY [2026-10-24] D1024 OLD W6 rebound to science pin 111c4781 (operational 587ff73e) after T-EQ FAIL — src: LOG:865-867; LOG:993-1021
- [2026-10-24] Sealed D4096 pack-01 at NEW pin bf8ac97d SUPERSEDED BY [2026-10-24] d4096-old-equiv-02-pack-01 at OLD collector / science pin 111c4781 — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:22; AR/exp/f4/JUDGMENT_D4096_START_20261024.md:58-61
- [2026-10-24] D1024 fit series liability 7.2 u + 6 u = 13.2 u (6.8 u left) SUPERSEDED BY [2026-10-24] 9.6 u (incl. two failed D1024 canaries at 1.2 u each) + 6 u = 15.6 u (4.4 u left) — src: AR/exp/f4/HANDOFF_D1024_FIT_20261024.md:35-37; AR/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:25
- [2026-10-24] Handoff upload projection (236 s, bytes ÷ average rate) SUPERSEDED BY [2026-10-24] per-file model ≈110 s — src: LOG:285; AR/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:24-27
- [2026-10-24] D1024 transfer bar 150 s SUPERSEDED BY [2026-10-24] D1024-only prospective exception ≤300 s (staging ≤300 s unchanged) — src: AR/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:9; LOG:1395
- [2026-10-24] R-A waits for F2 until Sun 18:00 PT SUPERSEDED BY [2026-10-24] R-A timing amendment: verified F2 ends waiting, R-A starts immediately once prerequisites complete, by Sun 18:00 PT — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:137-139; LOG:1394
- [2026-10-24] D4096 design "nested mixed-platform (Mac + L4)" SUPERSEDED BY observed placement Mac-only (L4 lane closed for October) — src: LOG:1338; LOG:1366
- [2026-10-25] FIX_FIRST 7f3c0857 SUPERSEDED BY FIX_FIRST 1fe60c99 SUPERSEDED BY ACCEPT 69b14c95 — src: AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:5; AR/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:5
- [2026-10-24] Conflict: D1024 package states 2,160 materialized parents, while preempt/resume counts give 62 done + 2,092 pending and the 11:45 row uses a 2,093 denominator — src: LOG:1003; LOG:252; LOG:1336

## UNVERIFIED
- Exact D1024 harm count H (only H ≥ 1 is established; not exported by funding.py) — JUDGMENT_RA_STOP_FUNDING:9.
- D256 harm-screen pass/fail at R-A — not reported in funding-look.json (only D1024's boolean exported).
- Post-mortem diagnostic outcomes (C, U, margin bins) — exporter not executed as of the sources read.
- Time when JUDGMENT_F2_IMMUTABLE_SUCCESSOR was written (dated 2026-10-24, no clock time found).
- D256 metadata-only pack-01 (e655caac) disposition: superseded by pack-02 (6313fe8f) per file times 00:20 vs 00:23 PT 10-24; no explicit supersession line located.
- The "D256 fit" receipt-based wall time per stage beyond the 676 s total — not opened (output dirs not read).

## Incidents and lessons (facts only)
- [2026-10-24 00:04–00:15 PT] Five UPLOAD_CAPs in one night incl. F4 D256 canary-02 → diagnosis: 2–4 failed ~120 s control calls exhausted staging clock → bundle inputs into one verified archive per job — src: LOG:236 | LOG:238-239
- [2026-10-24 04:12 PT] D1024 preempted by nice-0 Colab supervisor after renicer died (~20 min Mac idle) → ops/renicer-simple.sh and a D1024 resume tool — src: LOG:252-253
- [2026-10-24 17:00–22:18 PT] D1024 fit blocked by L4 transport: two canary UPLOAD_CAPs, requal FAIL at 207.3 s and 221.7 s → D1024-only ≤300 s transfer exception then requal PASS at 48.0 s — src: LOG:1366 | LOG:1392 | LOG:1395-1396
- [2026-10-24 ~22:28 PT → 2026-10-25 08:32 PT] R-A prepare failed on attempt-path bug; failure not acted on until 08:32 PT; D4096 drained ≈11 h Mac idle ("operator gap") → sol-raprepfix derives output from sealed binding — src: LOG:1403
- [2026-10-25] funding.py did not export the integer H, leaving STOP_FUNDING's harm magnitude unknown → post-mortem exception to recover it once — src: AR/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:9 | BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:5
- [2026-10-25] Post-mortem exporter needed three review rounds (publish-before-read gap, Python-signal deadline, stale-PID race) before ACCEPT — src: AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_7f3c0857_20261026T000941Z.md:11-13 | AR/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:9 | AR/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:9

## Terms (for the glossary)
- F4: nested rescue-life dose study for a frozen-B8 residual (D256/D1024/D4096) — src: AR/exp/f4/F4_REGISTRATION.md:1
- Rescue-bearing life: one natural hero death with ≥1 exact H300 survivor action — src: AR/exp/f4/F4_REGISTRATION.md:14
- Reserve: fixed 64 fold-0 rescue lives (≤4 per parent) used for life-macro retention — src: AR/exp/f4/F4_REGISTRATION.md:17-18
- Life-macro rescue retention: mean over reserve lives of the fraction of decisive rows where the served action is an exact survivor — src: AR/exp/f4/F4_REGISTRATION.md:53-54
- Harm screen H: zero-event count of archive rows with a changed action that is a confirmed or unknown-outcome harm; pass iff H = 0 — src: AR/exp/f4/F4_REGISTRATION.md:57-58
- Look R-A / R-B: once-only create-only reserve looks (D256+D1024 jointly / D4096) for funding/admission — src: AR/exp/f4/F4_REGISTRATION.md:51-52
- F2 / volume_struck: 25→100% selected-action survival test; volume struck when one-sided LB90 ≤ 0 — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:128-129
- STOP_FUNDING: R9 label 5, R6.3 strikes-branch failure; terminal, no gameplay, recipe closed — src: AR/exp/f4/F4_REGISTRATION.md:69-70,111
- NOT_RUN_BY_RULE: units skipped because a rule stopped the path; never INCOMPLETE — src: AR/exp/f4/F4_REGISTRATION.md:75
- B6 discard: sealing D4096 as d4096-not-run-by-rule.json on a terminal R-A/fit stop — src: AR/exp/f4/JUDGMENT_D4096_START_20261024.md:109-112
- T-EQ: round-27 speed byte-identity equivalence test (OLD vs NEW collector) — src: LOG:242
- DEVELOPMENT_FOREVER: designation for the 29 exposed census parents and descendants — src: BR/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:81-85
