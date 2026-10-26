# F4 frozen-B8 residual dose study

**Summary:** F4's retention gain passed its funding bars, but D1024 failed the harm screen, producing terminal STOP_FUNDING and sealed D4096 discard; the post-mortem exporter was accepted conditionally. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:7-9; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/f4-study.md`; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [Label factory](label-factory.md), [Successors](r30-r32-f5-successors.md), [Compute infrastructure](colab-and-compute-infra.md)

## Verdict sequence

1. **2026-10-23 ≈00:45 PT — RATIFY-WITH-CHANGES:** the Round-26 Opus stand-in ruled on F4 (the nested rescue-life dose study for a frozen-B8 residual) before data. (src: big-lever/round-26/JUDGMENT.md:1-12; big-lever/exp/f4/F4_REGISTRATION.md:1)
2. **2026-10-23 01:25 PT — APPROVED:** the approval bound the registration and FitRecipe after eight implementation changes and Codex-V verification. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:157; big-lever/exp/f4/judgment.json:1-7)
3. **2026-10-24 PT, clock time unrecorded — ACCEPT:** the initial D1024 fit tooling review accepted `efa01291`. (src: big-lever/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW.json:1-3; big-lever/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW.json:63-65)
4. **2026-10-24 09:15 PT — KEEP_QUALIFIED_UPLOAD_FOR_LABELS_MOVE_B8_TO_EXISTING_PUBLIC_URL:** the transport ruling kept labels private and moved only B8 to its existing public URL. (src: big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:1-3)
5. **2026-10-24 PT, clock time unrecorded — ACCEPT:** the follow-up D1024 review accepted `fa07bd75` after the transport-row and hardening changes. (src: big-lever/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW_02.json:1-12; big-lever/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW_02.json:47-52)
6. **2026-10-24 ≈11:30 PT — ADMIT_D4096_OLD_W12 / ADMIT_SEALED_EARLY_COLLECTION / BUILD_NOW:** the judge admitted sealed Mac collection and commissioned F2 (the selected-action survival volume test). (src: big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:1-3; big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:120-133)
7. **2026-10-24 PT, clock time unrecorded — FIX-FIRST:** I3's D4096 admission repair required exact-zero projection for zero-charge Mac collection, with Mac/B2 checks. (src: research/bl_residual/I3_FIXFIRST_HANDOFF_20261024.md:1-16)
8. **2026-10-24 12:35 PT — conditional ACCEPT:** the D4096 I1 build re-review accepted sealing and starting collection subject to B1/B2; I2 still had a publication-time blocker. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1339)
9. **2026-10-24 13:36 PT — ACCEPT:** Astra accepted I2/F2 tooling at `92f26cc4` after four fix rounds. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1356)
10. **2026-10-24 13:36 PT — ACCEPT:** the I3 L4-accounting repair `da4dbf25` was accepted. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1356)
11. **2026-10-24 17:00 PT — HOLD:** the D1024 fit waited for transport repair after two excluded canaries returned INCOMPLETE / UPLOAD_CAP. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1366)
12. **2026-10-24 18:28 PT — ACCEPT:** Astra accepted transport repair `0bd6f058`, merged as `f628c1af`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1391)
13. **2026-10-24 18:50 PT — REFUSE / HOLD:** requalification measured 207.3 s transfer against the 150 s bar, a real transport-performance failure. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1392)
14. **2026-10-24 PT, clock time unrecorded — ADMIT ONE IMMUTABLE SUCCESSOR, FIX FIRST:** Astra authorized F2's execution-identity repair before its accepted successor. (src: big-lever/exp/astra-20261024/JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md:1-11)
15. **2026-10-24 19:09 PT (from UTC) — ACCEPT:** Astra accepted the immutable F2 successor. (src: big-lever/exp/astra-20261024/ACCEPT_F2_IMMUTABLE_SUCCESSOR_20261024.md:1-6)
16. **2026-10-24 19:36 PT (from UTC, 19:36:53) — ACCEPT:** the F2 post-run review accepted COMPLETE with `volume_struck=true`. (src: big-lever/exp/astra-20261024/JUDGMENT_F2_POSTRUN_adef78ec_20261024.md:1-18)
17. **2026-10-24 20:05 PT — ACCEPT:** the judge accepted throughput repair `55439bff` and amended R-A timing. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1394)
18. **2026-10-24 20:12 PT (from UTC) — ACCEPT:** the kernel replacement acceptance covered the saved plan delta only. (src: big-lever/exp/astra-20261024/ACCEPT_F4_KERNEL_REPLACEMENT_PLAN_1792897885.md:3-5)
19. **2026-10-24 20:45 PT — REFUSE / ADMIT:** after a 221.7 s transfer failure, the judge admitted a prospective D1024-only transfer bar of ≤ 300 s and specified no B6 discard then. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1395)
20. **2026-10-24, recorded by 22:18 PT — ACCEPT:** Astra accepted transfer-exception implementation `de5bde2d`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396; big-lever/exp/astra-20261024/ACCEPT_F4_D1024_TRANSFER_IMPLEMENTATION_de5bde2d.md:5)
21. **2026-10-24, recorded by 22:18 PT — ACCEPT:** Astra accepted live requalification `acceptance-1792902941` after PASS. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396; big-lever/exp/astra-20261024/ACCEPT_F4_D1024_LIVE_REQUALIFICATION_1792902941.md:5)
22. **2026-10-24, recorded by 22:18 PT — ACCEPT:** the attempt-04 pre-canary review accepted the excluded full-inventory canary. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396; big-lever/exp/astra-20261024/ACCEPT_F4_D1024_PRECANARY_ATTEMPT04_de5bde2d.md:5)
23. **2026-10-24, recorded by 22:18 PT — ACCEPT:** following canary PASS, the pre-fit review accepted one registered fit and forbade retry or refit. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396; big-lever/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:5)
24. **2026-10-25 10:22 PT (from UTC, 10:22:40) — ACCEPT:** Astra accepted the R-A command packet, explicitly without deciding its funding outcome. (src: big-lever/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:3-5)
25. **2026-10-25 10:23–10:40 PT — STOP_FUNDING:** the single R-A execution stopped the F4 funding path. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1404)
26. **2026-10-25, recorded by 10:50 PT — UPHOLD STOP_FUNDING:** Astra upheld the result and the operator executed B6 discard/sealing. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:1-9; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1404)
27. **2026-10-25 16:50 PT — ALLOW ONE DIAGNOSTIC EXPORT; STOP_FUNDING REMAINS FINAL:** the judge authorized the bounded post-mortem exception. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1427; research/bl_residual/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:1-5)
28. **2026-10-25 17:09 PT (from UTC, 17:09:41) — FIX-FIRST:** the first exporter review rejected incomplete publication gating and the hard-timeout implementation. (src: big-lever/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_7f3c0857_20261026T000941Z.md:1-13)
29. **2026-10-25 17:24 PT (from UTC, 17:24:06) — FIX-FIRST:** re-review of `1fe60c99` found a stale-PID race after the earlier blockers were repaired. (src: big-lever/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:1-13)
30. **2026-10-25 17:29 PT (from UTC, 17:29:51) — ACCEPT:** exporter `69b14c95` was accepted subject to F5 adoption and publication before diagnostic exposure. (src: big-lever/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:1-5; big-lever/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:17-23)
31. **2026-10-25 17:40 PT — ACCEPTED / HOLD:** the log recorded the exporter as gated and F5 adoption on HOLD for R10a annex gaps. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)

## Registration

- The hypothesis asked whether increasing selected rescue-bearing lives from 1,024 to 4,096 strengthened the same residual in fresh paired H5000 gameplay, and whether the top dose beat bare `b8-s3` (SHA `3e649783…eb4f8`). (src: big-lever/exp/f4/F4_REGISTRATION.md:8-11)
- The primary endpoint was full-horizon terminal-hero mass integral over 5,000 frames, with dead frames contributing zero; inference concerned fixed checkpoints under one fit seed. (src: big-lever/exp/f4/F4_REGISTRATION.md:9-11)
- A rescue-bearing life meant one natural hero death with at least one exact H300 survivor action, with inference clustered by parent and fold 0 held out. (src: big-lever/exp/f4/F4_REGISTRATION.md:14-16)
- D256, D1024 and D4096 contained 192, 960 and 4,032 nested fit lives respectively, plus the same 64-life reserve fixed at D256 with at most four reserve lives per parent. (src: big-lever/exp/f4/F4_REGISTRATION.md:17-20)
- The frozen recipe used a zero-initialized `Linear(256,6)` residual on B8 features, Adam fp32 at learning rate 0.001, 2,000 updates, seed 3403 and batch 256 split equally between doom and ordinary rows. (src: big-lever/exp/f4/F4_REGISTRATION.md:25-31)
- Losses combined exact survival pairs, growth pairs only among exact survivors weighted by `min(g,4)/4`, retention KL coefficient 1 and residual L2 coefficient `1e-4`, without prevalence weights. (src: big-lever/exp/f4/F4_REGISTRATION.md:32-38)
- Non-finite values, illegal actions or more than 90% one-action share triggered KILL_FIT at any dose; the D4096-only drift bars were more than 5% disagreement per length decile, boost share above 1.5× B8 or reserve retention below 0.70. (src: big-lever/exp/f4/F4_REGISTRATION.md:41-48)
- R-A was the once-only joint D256/D1024 reserve look; R-B was the once-only D4096 reserve look, both for funding/admission rather than efficacy. (src: big-lever/exp/f4/F4_REGISTRATION.md:50-54)
- When F2 struck volume or remained unpublished before R-A, D4096 funding required mean retention gain ≥ 0.05, parent-cluster LB90 > 0 and D1024 harm count H = 0; failure meant terminal STOP_FUNDING with no gameplay. (src: big-lever/exp/f4/F4_REGISTRATION.md:67-70)
- The harm screen counted archive rows whose action changed from B8 and either replaced exact survival with exact death or involved a non-EXACT selected/B8 outcome. (src: big-lever/exp/f4/F4_REGISTRATION.md:57-58)
- Registered base/effective Colab caps were D256 40/80 u, D1024 increment 120/240 u, D4096 increment 480/960 u and all fits together 10/20 u; fits used L4 unless a cached A100 fit demonstrated ≥ 5× speed-up. (src: big-lever/exp/f4/F4_REGISTRATION.md:77-79)
- The unexecuted gameplay screen required 128 H5000 worlds per mix across three mixes, both Holm tests at family alpha 0.10, top-vs-B8 gain ≥ +40 mass and dose gain ≥ +20 mass. (src: big-lever/exp/f4/F4_REGISTRATION.md:82-93; big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:39)
- Its four guards covered per-mix top-dose non-negativity, survival band v2, scripted no-death improvement ≥ 0.03 and delivered boost share in [0.5%, 10%]; the subsequent Watch6 product screen required 128 sessions of 14,400 frames each. (src: big-lever/exp/f4/F4_REGISTRATION.md:94-102)
- The approval bound registration SHA `66388c79…2fd65` and recipe SHA `49d12f48…3a5a`; changing the registered block required a new pre-data registration and judge review. (src: big-lever/exp/f4/judgment.json:1-7; big-lever/round-26/JUDGMENT.md:59-62)
- After T-EQ (the OLD/NEW collector speed-equivalence test) failed, D1024's science pin changed from `bf8ac97d` to repaired `111c4781`, with OLD collector `61900af2` at W6 and operational pin `587ff73e`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:242; docs/research/final_run_20261019/CAMPAIGN_LOG.md:993-1021)
- The rebinding preserved the namespace, 2,160 materialized parents, seeds, calibration and 144,000 s cap; only `source_commit` and `mac_gate.source_commit` changed. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:999-1007)

## D256 tranche

- Collection completed on 2026-10-23 at 23:18 PT, with eight attempts, 1,566 sealed units, 864 parents, 16,333.70 s cumulative factory time and zero thermal trips. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:232; docs/research/final_run_20261019/CAMPAIGN_LOG.md:794-810)
- Preparation first failed for a missing explicit device, then passed at `c2c82ff2`; the 2026-10-24 00:08 PT canary returned INCOMPLETE / UPLOAD_CAP before any fit ran. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:234; docs/research/final_run_20261019/CAMPAIGN_LOG.md:236)
- Bundled-input canary PASS at 00:53 PT preceded the L4 fit's COMPLETE at 01:02 PT with process exit code 0, followed by verified publication at 01:21 PT without a D256-only look. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:240-243)
- The metadata post-fit handoff bound `d256-metadata-pack-02` to fit pin `6313fe8f`. (src: research/bl_residual/D256_METADATA_POSTFIT_HANDOFF_20261024.md:3-8)
- The allocation contained 192 training lives from 43 parents, all S-Watch6, and a reserve of 64 lives from 28 parents comprising 49 S-Watch6 and 15 S-gate lives. (src: big-lever/exp/f4/d256-metadata-pack-02/inputs/life-allocation.json:1)
- Of 590 available fit rescue lives, 398 were unused at dose 256. (src: big-lever/exp/f4/d256-metadata-pack-02/inputs/life-allocation.json:1)
- The exported D256 checkpoint `fit-0000.pth` was 6,588,730 bytes with SHA `103d02e9…0360`. (src: big-lever/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:12)
- Total main-fit wall time was 676 s; the transport ruling reported 72.6 s canary transfer and 260.0 s main transfer. (src: big-lever/exp/f4/HANDOFF_D1024_FIT_20261024.md:39; big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:17-23)

## D1024 tranche

- Collection began under OLD W6 on 2026-10-24 at 01:47 PT, was preempted at 03:50 PT and resumed at 04:24 PT with 136,415 s remaining in its cumulative cap. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:245; docs/research/final_run_20261019/CAMPAIGN_LOG.md:252-253)
- The completion record reported 2,092 parents and zero pending at 16:13 PT, with final union `f5ec94cc…` published at `c55d3c66`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365)
- The allocation contained 960 training lives from 372 parents, comprising 753 S-Watch6 and 207 S-gate lives, with the unchanged 64-life reserve. (src: big-lever/exp/f4/d1024-pack-04/inputs/life-allocation.json:1)
- Of 2,260 available fit rescue lives, 1,300 were unused at dose 1,024. (src: big-lever/exp/f4/d1024-pack-04/inputs/life-allocation.json:1)
- The transport ruling retained qualified private label uploads and used the existing public B8 asset, projecting about 110 s clean transfer. (src: big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:1; big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:25-28)
- Preparation at `c55d3c66` failed a clean-environment import check, repaired by `3e6a1f0d` and runbook pin `6deda139` before preflight-02 PASS. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365)
- Excluded canary-01 and the authorized mechanical retry canary-02 both failed UPLOAD_CAP on control-call stalls of about 240 s. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1366)
- Requalification failures of 207.3 s and 221.7 s preceded the exact-inventory D1024 exception implemented at `de5bde2d`, with transfer and staging independently bounded by 300 s. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1392; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1395-1396)
- Live requalification observed all 12 required gates, CPU transfer/staging of 44.6/55.1 s and L4 transfer/staging of 48.0/57.5 s. (src: big-lever/exp/astra-20261024/ACCEPT_F4_D1024_LIVE_REQUALIFICATION_1792902941.md:13-16)
- Canary-04 passed in 600.791 s with transfer 136.405 s and staging 183.582 s, using three synthetic updates separate from the registered training partition. (src: big-lever/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:12-17)
- The accepted main fit retained base/effective limits of 2/4 u, reserve 0.4/0.8 u and timeout 60/120 min; the final reserved series liability was 15.6 u against 20 u. (src: big-lever/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:24-25)
- The single registered fit passed at 22:23 PT in 365 s with stop verified; the phase and binding were published before census or R-A. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1399; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1403)

- The exported D1024 checkpoint `fit-0001.pth` was 6,588,730 bytes with SHA `da1d2656…5093`, bound to fit pin `de5bde2d`. (src: big-lever/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:13)

## D4096 (sealed, discarded by rule B6 — never opened)

- The pre-data ruling distinguished outcome-blind Mac collection from purchasing D4096 for fit, probe, R-B, gameplay or Colab-unit expenditure. (src: big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:83-85)
- B1 required completed D1024 collection and terminal/HOLD fit jobs; B2 required a sealed projection ≤ 130 Mac-h; B3 required PENDING_SEALED; B4 barred Colab expenditure before funding PASS; B5 gave census and R-A priority. (src: big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:97-108)
- The 2026-10-24 17:00 PT log recorded Mac-only OLD W12 collection after B1 satisfaction and a sealed 66.6 h projection against the 130 h cap. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1366)
- The terminal ruling recorded sealed PREEMPTED collection and 18,267.14 s of sunk operational time. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:27)
- B6 discard was executed by the 2026-10-25 10:50 PT log entry through `d4096-not-run-by-rule.json`, preserving collection while excluding its F4 union, fit, probe, R-B and gameplay. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1404)
- The tranche was never fitted, published or opened under F4; skipped stages were NOT_RUN_BY_RULE, and any non-F4 reuse required a new pre-data ruling naming the corpus before access. (src: big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:109-112; big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:39; big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:43)

## F2 immutable successor

- F2 compared 25% and 100% A2 c0 frozen-linear D300 models on 1,481 decisive rows from 269 worlds, with a world-cluster one-sided LB90 from 10,000 resamples at seed 3403. (src: big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:120-129)
- The first sealed run refused before claiming the look because its input ledger bound pre-hardening `f2.py`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1393)
- The successor ruling preserved the original ledger and statistical block `23c86e2b…`, admitting only a create-only `f2-inputs-20261024-r2.json` and the canonical `INPUT_LEDGER` filename change. (src: big-lever/exp/astra-20261024/JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md:6-11; big-lever/exp/astra-20261024/JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md:27-37)
- The result was published at `adef78ec` on 2026-10-24 at 19:24 PT (from UTC, 19:24:24), before R-A and the 2026-10-25 18:00 PT deadline. (src: big-lever/exp/astra-20261024/JUDGMENT_F2_POSTRUN_adef78ec_20261024.md:57-61)
- Paired survival change (100% − 25%) was −0.02903443619176232 and LB90 was −0.04213373082537578, so `volume_struck=true`. (src: big-lever/exp/astra-20261024/JUDGMENT_F2_POSTRUN_adef78ec_20261024.md:13-18)

## Census

- The R5 natural-archive census was frozen and published before R-A, binding receipt `44577f8a…30f0` and table `34ad5395…7b4d` without reporting outcomes. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1402; big-lever/exp/f4/r-a-archive-census-01/census.json:1)
- It recorded 5,936 rows, 560 natural rows, 71 eligible lives including three non-rescue lives, 29 parents, 64 reserve lives and maximum reserve allocation index 321,000,000. (src: big-lever/exp/f4/r-a-archive-census-01/census.json:1)
- The packet reviewer independently recomputed the shared reserve, ancestry and 71 eligible lives. (src: big-lever/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:14)

## R-A packet and STOP_FUNDING

- R-A preparation first failed on an attempt-01/attempt-04 path mismatch, repaired by `e238b337`, followed by `f9c08082` for spaced-versus-compact JSON receipt hashes. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1403-1404)
- The accepted version-3 packet bound SHA `443ad83f…dd8c`, execution pin `f9c08082` and 34 matching source identities. (src: big-lever/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:9-11)
- Acceptance required drained D4096 collection, exclusive Mac use under B5 and the 2026-10-25 18:00 PT cutoff. (src: big-lever/exp/astra-20261024/ACCEPT_RA_PACKET_f9c08082_20261025.md:21-23)
- On 2026-10-25 PT the runner began at 10:23:09, claimed the look at 10:39:48, wrote the result at 10:39:57 and ended at 10:40:53, with one retained canonical claim and funding result. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:22-23)
- **Mean life-macro retention gain D1024 − D256 was 0.06621223984499343**, exceeding the required 0.05. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:7)
- **Parent-cluster LB90 was 0.048454798628559156**, above zero with 10,000 resamples and seed 3403. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:7)
- **D1024 harm failed: H ≥ 1 against required H = 0**, with only `harm_screen_d1024_pass=false` retained. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:7-9)
- The ruling identified the missing integer H as a reporting defect, upheld STOP_FUNDING and prohibited retry, refit or relook to repair reporting. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:9)
- No gameplay, probe or R-B ran; further F4 labels, heads, seeds and refits were barred, and B8 serving remained unchanged. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:39)

## Post-mortem

- The exception authorized one aggregate export per dose of confirmed harm C, unknown-outcome harm U and H = C + U, with three disjoint U components and four fixed margin bins. (src: research/bl_residual/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:62-67)
- It prohibited a new funding evaluation, bootstrap, verdict or inferential statistic, and bound the diagnostic to a once-only Mac CPU run with a 30/60 min base/effective wall envelope and publication before inspection. (src: research/bl_residual/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:60; research/bl_residual/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:119-123)
- All 29 census parents and descendants became DEVELOPMENT_FOREVER, excluded from successor validation evidence. (src: research/bl_residual/f4_postmortem_20261025/JUDGMENT_F4_POSTMORTEM_EXCEPTION_20261025.md:81-85)
- The first FIX-FIRST identified incomplete publish-before-inspection enforcement and a Python SIGALRM implementation that did not enforce the hard ceiling. (src: big-lever/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_7f3c0857_20261026T000941Z.md:11-13)
- The second FIX-FIRST confirmed those repairs but found a stale-PID race that could signal an already-reaped PID. (src: big-lever/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:5-11; big-lever/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:17-18)
- ACCEPT at `69b14c95` followed signal blocking around `waitpid`, with six signal-at-reap regressions changing from FAIL to PASS and reported evidence of 16 watchdog plus 65 focused tests. (src: big-lever/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:9-13)
- As of 2026-10-25 17:40 PT the accepted exporter had not run because F5 adoption/publication remained gated by R10a annex gaps. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)

## Superseded and conflicting records

- **Science pin:** repaired `111c4781` with OLD W6 and operational `587ff73e` superseded the original D1024 `bf8ac97d`/W12 plan after T-EQ FAIL. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:993-1021; docs/research/final_run_20261019/CAMPAIGN_LOG.md:865-867)
- **D4096 collector:** the OLD `61900af2` closure at science pin `111c4781` superseded the original NEW `bf8ac97d` pack, while preserving the original allocation and cap. (src: big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:42-44; big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:58-61; big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:22)
- **Parent counts:** the final log reported 2,092 completed parents; earlier records gave 2,160 materialized parents, 62 done plus 2,092 pending, and a progress denominator of 2,093, without a reconciled common count. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1003; docs/research/final_run_20261019/CAMPAIGN_LOG.md:252; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1336)
- **Fit liability:** the final reserved 9.6 u prior liability plus 6 u for D1024 totaled 15.6 u, leaving 4.4 u; this superseded 7.2 + 6 = 13.2 u and 6.8 u remaining after both failed D1024 canaries were charged at 1.2 u each. (src: big-lever/exp/astra-20261024/ACCEPT_F4_D1024_PREFIT_ATTEMPT04_9c2b88db.md:25; big-lever/exp/astra-20261024/ACCEPT_F4_D1024_PRECANARY_ATTEMPT04_de5bde2d.md:21; big-lever/exp/f4/HANDOFF_D1024_FIT_20261024.md:35-37)
- **Transfer estimate:** the per-file estimate of about 110 s superseded the 236 s bytes/average-rate estimate because the latter multiplied fixed per-file overhead. (src: big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:1; big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:24-28; docs/research/final_run_20261019/CAMPAIGN_LOG.md:285)
- **Transfer bars:** the later exact-inventory D1024/L4 exception allowed ≤ 300 s transfer, superseding the original 150 s bar only in that scope; staging remained ≤ 300 s and CPU/other lanes retained 150 s. (src: big-lever/exp/astra-20261024/ACCEPT_F4_D1024_LIVE_REQUALIFICATION_1792902941.md:21; big-lever/exp/f4/JUDGMENT_D1024_FIT_TRANSPORT_20261024.md:9; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1395)
- **R-A timing:** the 20:05 PT amendment ended waiting once F2 was verified and required R-A as soon as prerequisites completed, by 2026-10-25 18:00 PT, superseding the earlier wait formulation. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1394; big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md:137-144)
- **Placement:** observed Mac-only D4096 collection superseded the nested mixed-platform Mac/L4 design after the October L4 lane closed. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1366; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1338)
- **F2 disposition string:** `UNPUBLISHED_APPLY_STRIKES_BRANCH` was a shared code label, not evidence of absent publication; F2 was published, `f2_timing_eligible=true`, and the applied branch was STRIKES. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:20)
- **Exporter reviews:** ACCEPT `69b14c95` superseded FIX-FIRST `1fe60c99`, which had closed the two blockers in FIX-FIRST `7f3c0857` before exposing the stale-PID race. (src: big-lever/exp/astra-20261024/ACCEPT_F4_POSTMORTEM_69b14c95_20261026T002951Z.md:5-13; big-lever/exp/astra-20261024/FIX_FIRST_F4_POSTMORTEM_1fe60c99_20261026T002406Z.md:5-11)

## Unverified

- **Exact D1024 H:** the retained receipt established H ≥ 1 but omitted the integer count, so no exact count was verified. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:9)
- **D256 harm-screen result at R-A:** neither complete held-out record was exported, so a D256 pass/fail claim was not established by the retained funding receipt. (src: big-lever/exp/astra-20261024/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md:9)
- **Post-mortem C, U and margin-bin outcomes:** the exporter remained gated at the source cutoff, so no diagnostic results were verified. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)
- **F2 immutable-successor ruling clock time:** its header supplied only 2026-10-24, so its exact time could not be assigned. (src: big-lever/exp/astra-20261024/JUDGMENT_F2_IMMUTABLE_SUCCESSOR_20261024.md:1-4)
- **Other date-only verdict times:** the initial/follow-up D1024 reviews and I3 FIX-FIRST lacked clock times; their placement in the ordered sequence follows documented dependencies rather than an invented timestamp. (src: big-lever/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW.json:1-65; big-lever/exp/f4/CLAUDE_V_STANDIN_D1024_FIT_REVIEW_02.json:1-52; research/bl_residual/I3_FIXFIRST_HANDOFF_20261024.md:1-16)
- **Metadata pack-01 disposition:** the continuation handoff explicitly selected pack-02, but did not establish the fact sheet's proposed explicit supersession of pack-01. (src: research/bl_residual/D256_METADATA_POSTFIT_HANDOFF_20261024.md:3-8)
- **D256 stage-level wall time beyond the 676 s total:** full receipt-stage accounting was not verified because output directories were excluded from verification. (src: big-lever/exp/f4/HANDOFF_D1024_FIT_20261024.md:39)
- **Reconciliation of D1024 parent totals:** the cited package, preemption, progress and completion records did not establish why their counts differed. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1003; docs/research/final_run_20261019/CAMPAIGN_LOG.md:252; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1336; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365)
