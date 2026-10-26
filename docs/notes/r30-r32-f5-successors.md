# Round 30, Round 32 and F5 successor work

**Summary:** Round 30’s calibration route closed operationally with its main on HOLD, Round 32 completed synthetic rehearsal, and F5 adoption remained on HOLD. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:257; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1428-1429)
**As of:** 2026-10-25 17:40 PT · **Sources:** fact sheet `_facts/r30-r32-f5-successors.md`; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [F4 study](f4-study.md), [Big-lever program](big-lever-program.md), [GPU engine and speed](gpu-engine-and-speed.md), [Colab and compute infrastructure](colab-and-compute-infra.md)

## Round verdicts 30–37

| Round | Topic | Recorded verdict or judgment boundary |
|---|---|---|
| 30 | Converting search evidence into a servable policy | **ADOPT_DIAGNOSIS_FIRST**: one existing-label optimization/conversion audit and one conditional rescue-only mass-ranking successor. (src: big-lever/round-30/JUDGMENT.md:1-7) |
| 31 | Future-randomness leakage | **ADOPT_BOUNDED_FUTURE_NOISE_AUDIT**: retain common random numbers and qualify a bounded future-only intervention. (src: big-lever/round-31/JUDGMENT.md:1-5) |
| 32 | Consumer rehearsal and Mac evaluation allocation | **ADOPT_BOUNDED_CONSUMER_REHEARSAL_AND_DELIVERY_PREPARATION**. (src: big-lever/round-32/JUDGMENT.md:1-5) |
| 33 | Operations throughput | **ADOPT_BOUNDED_OPS_REPAIR**. (src: big-lever/round-33/JUDGMENT.md:1-5) |
| 34 | Unattributed A100 kernels | No JUDGMENT in the round directory; question only cited, with any external disposition unverified below. (src: big-lever/round-34/QUESTION.md:1-3) |
| 35 | Full-scale stage profile toward 2,000 steps/s | No JUDGMENT in the round directory; question only cited, with any external disposition unverified below. (src: big-lever/round-35/QUESTION.md:1) |
| 36 | Further A100 acceleration | **INTEGRATE_FIRST_AND_FUND_BOUNDED_ACCELERATION**, covering all eleven Round 36/37 memos. (src: big-lever/round-36/JUDGMENT.md:1-3) |
| 37 | Parallel panel on the Round 36 question | No separate JUDGMENT; the Round 36 judgment covered both panels. (src: big-lever/round-37/QUESTION.md:1; big-lever/round-36/JUDGMENT.md:3) |

## Round 30 original ruling

The 2026-10-23 PT ruling ordered R30-0 (corpus admission and consumer rehearsal), R30-1 (optimizer certificate and conversion audit), then conditional R30-2 (rescue-survivor mass-ranking fit versus matched control). (src: big-lever/round-30/JUDGMENT.md:13-19)
R30-3, the continuation diagnostic, was deferred and unfunded. (src: big-lever/round-30/JUDGMENT.md:20)

| Scope | Original base → effective envelope |
|---|---|
| R30-0 | ≤ 0.25 aggregate Mac core-hour; 0 Colab units. (src: big-lever/round-30/JUDGMENT.md:17) |
| R30-1 | 1.2 h / 2.4 u → 2.4 h / 4.8 u. (src: big-lever/round-30/JUDGMENT.md:18) |
| R30-2, conditional | 2.4 h / 4.8 u → 4.8 h / 9.6 u. (src: big-lever/round-30/JUDGMENT.md:19) |
| Combined | 3.6 h / 7.2 u → 7.2 h / 14.4 u, including effective reserve 1.2 h / 2.4 u. (src: big-lever/round-30/JUDGMENT.md:24) |

These were planning allowances at an L4 rate of 1.54–2 u/h; the later amendment superseded their unit ceilings. (src: big-lever/round-30/JUDGMENT.md:22-24; docs/research/round30_postf4_20261025/RULING.md:103-105)

- The split used salt `t34-r30-parent-split-20261023-v1`, with SHA256(salt + NUL + ultimate parent) mod 10 < 2 assigning validation and no redraw. (src: research/bl_round30/README.md:21-26)
- Required validation support was ≥ 32 parents, including ≥ 16 rescue-bearing parents. (src: big-lever/round-30/JUDGMENT.md:32)
- R30-1 preserved seed 3403, the original 2,000 Adam batches of 128 rescue plus 128 ordinary samples, and separate seed-3404 training-audit batches. (src: research/bl_round30/README.md:27-31)
- The residual had 1,542 trainable parameters, with strong-convexity lower bound μ = 2×10⁻⁴/1,536. (src: big-lever/round-30/JUDGMENT.md:7; big-lever/round-30/JUDGMENT.md:42)
- The certificate target was objective suboptimality ≤ 10⁻⁵×max(1,F(0)); implementation used the conservative target 10⁻⁵. (src: big-lever/round-30/JUDGMENT.md:44; research/bl_round30/README.md:32-36)
- The original bounds were 500 L-BFGS closure evaluations / 3,600 s and 600 s for the directed Decimal50 gradient enclosure. (src: research/bl_round30/README.md:31-34)
- Certified objective gain < 1%, retention gain < 1 pp, and action disagreement < 0.5% together closed additional optimizer investment. (src: big-lever/round-30/JUDGMENT.md:46)
- The ranking trigger required aggregate choice loss / available advantage ≥ 0.10 across ≥ 32 ultimate training parents with at least two surviving alternatives. (src: big-lever/round-30/JUDGMENT.md:54)
- The single successor look required ≥ 10% survivor-reference regret reduction, rescue retention ≥ 0.70 and ≥ control, and successor H=0 on the fixed natural ordinary archive. (src: big-lever/round-30/JUDGMENT.md:62)
- Precedence was **INVALID → INCOMPLETE → INSUFFICIENT → HOLD → KILL_SUCCESSOR / GO_OFFLINE**, with GO_OFFLINE funding only preparation of a separately priced Mac gameplay plan. (src: research/bl_round30/README.md:55-59)
- The original F4 closure fence kept R30-2 fit/look on HOLD after R9 labels 4–8 until a new explicit ruling. (src: big-lever/round-30/JUDGMENT.md:66)

The ruling identified `heldout_metrics` in `research/bl_residual/fit.py` as computing best mass over all exact legal selectable actions, including dying actions, rather than the required survivor-reference regret. (src: big-lever/round-30/JUDGMENT.md:56)

## Round 30 on 2026-10-24

Qualification recorded **MAC_PREFLIGHT_PASS / CONSUMER_CANARY_PREPARED**, while scientific R30-1 remained HOLD pending corpus and native-canary admission. (src: docs/research/round30_qualification_20261024.md:3-6)
The final qualification at `72b6fcf0` recorded 66 tests passed in 2.74 s. (src: docs/research/round30_qualification_20261024.md:65-70)
Earlier 63-test canaries were marked **SUPERSEDED_DO_NOT_DISPATCH** after review found the missing CUDA action-selection call. (src: docs/research/round30_qualification_20261024.md:59-62)

| codex_v_light receipt | Scope and recorded disposition |
|---|---|
| `round30_codex_v_light_20261024.json` | **ACCEPT_MAC_PREFLIGHT_AND_SYNTHETIC_CANARY_PREPARATION** at `72b6fcf0`; 257 source hashes verified, 66 tests reviewed, no dispatch or scientific result. (src: docs/research/round30_codex_v_light_20261024.json:1) |
| `round30_round32_mechanical_codex_v_light_20261024.json` | **ACCEPT**, no actionable findings; R30 pin `c1bf44c8`, R32 pin `c1ab2750`, and zero allocations, dispatches or stops by the review. (src: docs/research/round30_round32_mechanical_codex_v_light_20261024.json:1) |
| `round30_upload_codex_v_light_20261024.json` | **CODEX_V_LIGHT_PASS_PREPARED_NOT_DISPATCHED** at `98cc98a7`; 260 source files verified and scientific main HOLD. (src: docs/research/round30_upload_codex_v_light_20261024.json:1) |

The lifecycle repair admitted absent or pre-created empty output directories while refusing symlinks, files and nonempty directories. (src: docs/research/round30_round32_mechanical_relaunch_20261024.md:9-15)
Its frozen runtime `c1bf44c8` recorded 78 tests passed in 3.18 s. (src: docs/research/round30_round32_mechanical_relaunch_20261024.md:17-24)
The 04:24 PT relaunch failed by the 04:27 PT entry on **UPLOAD_CAP** for `b8.pth`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:253-254)
The upload repair used the existing public B8 checkpoint URL and bundled six metadata files into one 10,240-byte archive, reducing local input rows from seven to one. (src: docs/research/round30_upload_relaunch_20261024.md:12-27)
Its packaging proof passed in 9.557 s; **PASS_NOT_DISPATCHED** upload qualification covered 27,995 upload bytes with modelled forecasts of 65.629 s and 90.871 s. (src: docs/research/round30_upload_relaunch_20261024.md:48-50; docs/research/round30_upload_relaunch_20261024.md:80-84)
Two earlier R30 attempts remained **INCOMPLETE**, with verified stops and zero exports. (src: docs/research/round30_upload_relaunch_20261024.md:92-96)

The dated corpus census recorded **HOLD: no eligible corpus** because the live F4 D256/D1024 stream was excluded, F0 canaries supplied about seven validation parents against 32 required, and the other corpora were not in F0 format. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:256)

## Round 30 post-F4 on 2026-10-25

### Scope amendment and data split

At 13:50 PT, the scope amendment recorded **ADMIT_CLOSED_TRAINING_CORPUS_DIAGNOSIS; DISPATCH_HOLD** after F4’s terminal **STOP_FUNDING** disposition. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1405; docs/research/round30_postf4_20261025/RULING.md:5-11)
It admitted exactly 960 selected D1024 fold-1–4 fit lives, including the nested 192-life D256 prefix once. (src: docs/research/round30_postf4_20261025/RULING.md:25)
It excluded fold 0, the 64-life reserve, the R-A natural archive, unused tails and all D4096 material. (src: docs/research/round30_postf4_20261025/RULING.md:38-42)

| Split | Lives / parents | Rows |
|---|---|---|
| Training | 730 / 291 | 65,825: 5,613 ordinary and 60,212 rescue. (src: docs/research/round30_postf4_20261025/HANDOFF.md:50-55) |
| Validation | 230 / 81, all 81 rescue-bearing | 20,698: 1,754 ordinary and 18,944 rescue. (src: docs/research/round30_postf4_20261025/HANDOFF.md:50-55) |

The prospective control correction resolved the optimizer before computing the conversion trigger, using Adam for **OPTIMIZER_ADEQUATE** and certified L-BFGS for **OPTIMIZER_MATERIAL**. (src: docs/research/round30_postf4_20261025/RULING.md:75-81)
**OBJECTIVE_MISMATCH** held the successor; no decisive audit lives yielded **INSUFFICIENT**. (src: docs/research/round30_postf4_20261025/RULING.md:78-79)

### Astra acceptance, native canary and G5 HOLD

The review gates were G0 publication, G1 metadata, G2 preflight/code review, G3 pin/dry-run, G4 native canary and G5 final main-packet acceptance. (src: docs/research/round30_postf4_20261025/RULING.md:118-128)
Astra’s build judgment at `94635bd6`, runtime `a7893b97`, recorded **ACCEPT G1 + G2 + pre-canary**, with main HOLD and scientific decision **NOT_RUN**. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_POST_F4_BUILD_94635bd6_20261025T211841Z.md:3-9)
The judgment linked separate G1, G2 and PRE_CANARY acceptance receipts and verified all 281 preflight source hashes against evidence recording 106 passing tests. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_POST_F4_BUILD_94635bd6_20261025T211841Z.md:9-15)

At 14:50 PT, the campaign record reported native canary **PASS**, CUDA consumption and verified stop. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1406)
The canary took 1,100.253 s supervisor time, with 102.131 s transfer and 173.497 s staging. (src: docs/research/r30_postf4_g5_preparation_hold_20261025.md:10-11)
Charging 18 min history and 20 min canary liability against 144 min left a 106-min main ceiling and a required estimate ≤ 53 min. (src: docs/research/r30_postf4_g5_preparation_hold_20261025.md:12-13)
G5 remained **HOLD** because aggregate worker time did not identify a measured full-main forecast; the 7,962-min scaling figure was an unseparated-cost screening surrogate. (src: docs/research/r30_postf4_g5_preparation_hold_20261025.md:15-25)
No main was packaged or dry-run by that G5 preparation. (src: docs/research/r30_postf4_g5_preparation_hold_20261025.md:27-28)

### Timing probes 01–04 and closure

The first timing authorization added a separately accounted base/effective 20/40-min calibration allowance, taking combined effective liability to 472 min while retaining the 53/106-min main estimate/timeout limits. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1406)

| Probe | Review and observed disposition |
|---|---|
| 01 | **FIX-FIRST** at `185d8395` was followed by **ACCEPT** at `8260c8fe` for one timing attempt. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_TIMING_185D8395_FIX_FIRST_20261025.md:1-3; big-lever/exp/astra-20261024/JUDGMENT_R30_TIMING_8260C8FE_ACCEPT_20261025.md:5) |
| 01 refusal | **UPLOAD_CAP** occurred before allocation because the 512-KiB export limit also capped uploads; Astra classified it **NON-ATTEMPT**, preserving the allocated-attempt allowance. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1408-1409; big-lever/exp/astra-20261024/JUDGMENT_R30_TIMING_UPLOAD_CAP_NONATTEMPT_20261025.md:5-9) |
| 02 | Runtime `773ef710` separated the 64-MiB transport bound from the 524,288-byte timing-output cap and received **ACCEPT** for one probe dispatch. (src: docs/research/r30_timing_upload_cap_probe02_20261025.md:8-17; big-lever/exp/astra-20261024/JUDGMENT_R30_TIMING_PROBE02_0F90A3C2_ACCEPT_20261025.md:5) |
| 02 result | **INCOMPLETE / CHILD_NONZERO_EXIT** after 988.648 s, verified stop and no exports; a blanket handler erased phase/cause information, leaving the workload cause **UNKNOWN**. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md:5-19) |
| 03 | One observability-corrected probe received **ACCEPT**, with durable redacted phase telemetry replacing RAM-only diagnostics and a separate 30-min effective reservation taking the total to 502 min. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE03_3A830E71_ACCEPT_20261025.md:5-9; big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md:25-31) |
| 03 result | **INCOMPLETE / CHILD_NONZERO_EXIT** after 1,121.888 s; 36 telemetry records identified **CERTIFICATE_DEADLINE** at 600.016 s in the certificate phase. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE03_CERTIFICATE_AMENDMENT_20261025.md:15-25) |
| 04 | The bounded amendment prepared a 1,200-s certificate cap, 2,100-s worker cutoff and 2,400-s session at runtime `0644e3a9`, taking combined liability to 542 min. (src: docs/research/r30_timing_certificate_probe04_20261025.md:8-24) |
| 04 refusal | The dry-run was **REFUSED** with zero allocations because qualified probe sessions were capped at 30 min, while the packet required 40 min. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE04_ADMISSION_ROUTE_CLOSED_20261025.md:7-9) |

At 17:05 PT, the recorded ruling closed the calibration route as operationally unresolved / not admitted, rather than a scientific **KILL**, and retained R30-1 main and R30-2 **HOLD**. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1428)
The closure allowed independent F5 preparation with unknown optimizer adequacy disclosed. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE04_ADMISSION_ROUTE_CLOSED_20261025.md:15)

## Round 32

The rehearsal package recorded consumer readiness **BLOCKED** and statistics **RETAIN_FIXED_DESIGN**. (src: research/r32_rehearsal/README.md:16-22; research/r32_rehearsal/README.md:99-103)
The CPU canary dispatched at 03:48 PT on 2026-10-24 and initially failed on the pre-created output-directory lifecycle. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:249-250)
After the directory repair, the 04:24 PT record reported canary **PASS** and main dispatch at `c1ab2750`; the 04:27 PT record reported the main’s missing-`df` error in `standard_t`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:253-254)
The statistics repair supplied `heavy_tail_df=5` to both draws and recorded 23 Mac tests passed plus 28/28 synthetic units. (src: research/r32_rehearsal/HANDOFF_STATISTICS_FIX_20261024.md:17-23; research/r32_rehearsal/HANDOFF_STATISTICS_FIX_20261024.md:38-41)

At 04:48 PT, main `r32-cpu-main-4f11670a-01` recorded **PASS** with 30 exports and 28/28 units, followed by **COMPLETE** Mac reduction. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:257)
Decision cases passed 16/16 and lifecycle cases passed 3/3 positive plus 12/12 faults; all eleven statistics scenarios returned **RETAIN_FIXED_DESIGN**. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:257)
The synthetic result unblocked no scientific work. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:257)
Series accounting recorded 0.0351 u / 26.3 min against 0.384 u / 288 min, with **ALERT UNEXPLAINED_ACCOUNT_DEBIT** of 0.0288 u versus 0.0167 u tolerance. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:257)

## Round 33

The 2026-10-23 PT closeout bound source `d53eaef7`, recorded 64 tests passed in 1.06 s, and recorded a 15-check, three-shard Mac preflight in 3.583 s. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:675-680)
Codex-V receipt `codex-v-02` recorded **ACCEPT**, and hosted CI run `40057728587` at `9bfd99ea` passed 15 Linux nodes. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:682-687)
The CPU pilot closed as **CANCEL_CPU_PILOT_NO_REMAINING_DEMAND**, with actual cloud charges of 0 u, 0 min and 0 assignments. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:689-700)
The prepared pilot manifest had been refused for lacking a target-platform family-canary receipt. (src: research/r33_ops/HANDOFF.md:98-104)

## F5

### Drafts v1/v2 and REVISE

F5’s first draft described one harm-targeted change to F4 versus a matched control and was explicitly **DRAFT_PREDATA — NOT ADOPTED**. (src: big-lever/exp/astra-20261024/F5_DRAFT_PREDATA_REGISTRATION_20261025.md:1-4)
Its menu named P (ordinary KL coefficient 1 → 4), W (survival-pair weight ×4 on specified ordinary harm pairs), and COV (exact H300 coverage completion). (src: big-lever/exp/astra-20261024/F5_DRAFT_PREDATA_REGISTRATION_20261025.md:88-90)
The 17:05 PT adoption review recorded **REVISE**, citing ordering, frozen sampling/allocation, compute totals, fixed-N versus sequential design, drift kill and D4096 exclusion. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1428)
Draft v2 preserved v1’s identity and menu while recording **DRAFT_PREDATA v2 — NOT ADOPTED; EXECUTION HOLD**. (src: big-lever/exp/astra-20261024/F5_DRAFT_PREDATA_REGISTRATION_20261025_v2.md:3-11)
R10a, the metadata-only adoption annex, required adoption/publication before diagnostic exposure, with adoption < exposure ≤ selection < first F5 work. (src: big-lever/exp/astra-20261024/F5_DRAFT_PREDATA_REGISTRATION_20261025_v2.md:531-552)
The independent v2 review at 2026-10-25 17:20 PT (from UTC) again returned **REVISE — adoption prerequisites incomplete**. (src: big-lever/exp/astra-20261024/JUDGMENT_F5_V2_REVISE_20261026T002045162238Z.md:3-6)

### R10a annex HOLD and gap closure

The metadata annex recorded **HOLD_BEFORE_EXPOSURE**, with 960 selected lives across 372 parents and nine ordinary-only siblings making 969 life metadata records. (src: big-lever/exp/astra-20261024/F5_R10A_METADATA_ANNEX_20261025.complete.json:1; big-lever/exp/astra-20261024/F5_R10A_METADATA_ANNEX_20261025.md:10-12)
It excluded 29 census parents and descendants and the original 167-parent fold-0 set, while listing 684 provisional parents split into 672 science and 12 timing. (src: big-lever/exp/astra-20261024/F5_R10A_METADATA_ANNEX_20261025.md:13-16)
Exact COV count U was unknown rather than zero, original runtime identities were unresolved, and the original collision audit covered only 3,024 seeds. (src: big-lever/exp/astra-20261024/F5_R10A_METADATA_ANNEX_20261025.md:15-24)
At 2026-10-25 17:37 PT (from UTC), the gap-closure ruling retained adoption **HOLD** and authorized bounded metadata work: a status-only extractor after independent acceptance, contemporaneous runtime recovery and a complete collision catalog. (src: big-lever/exp/astra-20261024/JUDGMENT_F5_R10A_GAP_CLOSURE_20261026T003745Z.md:3-13)

The sheet’s gap-closure v2 record still stated **HOLD before exposure**, with formal extractor acceptance pending and U unrun/unknown. (src: big-lever/exp/astra-20261024/F5_R10A_GAP_CLOSURE_ANNEX_20261026_v2.md:3-17)
It recorded recovered D256/D1024 source commits `61900af2` / `111c4781`, without closing historical runtime-identity requirements. (src: big-lever/exp/astra-20261024/F5_R10A_GAP_CLOSURE_ANNEX_20261026_v2.md:18)
Its collision union contained 1,160,637 exact raw seeds, including 66 negative seeds, while the 684-bank first-clear round 0 remained provisional and accepted round/hash remained null. (src: big-lever/exp/astra-20261024/F5_R10A_GAP_CLOSURE_ANNEX_20261026_v2.md:19-20)
The publication record bound commit `02d29811`, 108 passing tests, `adopted=false`, `real_status_scan=false` and **HOLD_BEFORE_EXPOSURE**. (src: big-lever/exp/astra-20261024/F5_R10A_CLOSURE_PUBLICATION_20261026.json:2-14)
The extractor packet was **SUBMITTED FOR INDEPENDENT ACCEPTANCE — REAL INPUT UNRUN**. (src: big-lever/exp/astra-20261024/F5_R10A_EXTRACTOR_REVIEW_PACKET_20261026.md:1-8)

### Draft folders by title

| Folder under `big-lever/exp/astra-20261024/` | Document title |
|---|---|
| `F5_COLLISION_RECOVERY_DRAFTS_20261026/` | “F5 R10a historical collision catalog — HOLD”. (src: big-lever/exp/astra-20261024/F5_COLLISION_RECOVERY_DRAFTS_20261026/collision-report.md:1) |
| `F5_EXTERNAL_COLLISION_RECOVERY_DRAFTS_20261026/external-collision-evidence/` | “External collision metadata supplement”. (src: big-lever/exp/astra-20261024/F5_EXTERNAL_COLLISION_RECOVERY_DRAFTS_20261026/external-collision-evidence/README.md:1) |
| `F5_RUNTIME_RECOVERY_DRAFTS_20261026/` | “F5 R10a runtime identity recovery”; the v6 record retained **HOLD_RUNTIME_IDENTITIES_INCOMPLETE**. (src: big-lever/exp/astra-20261024/F5_RUNTIME_RECOVERY_DRAFTS_20261026/runtime-evidence-v6/runtime-identities.md:1-3) |

## Post-mortem exporter ACCEPT pointer

The 17:40 PT record traced the F4 exporter from `a31a1d54` / `7f3c0857` through **FIX-FIRST**, then `1fe60c99` through another **FIX-FIRST** for the stale-PID race, to `69b14c95` **ACCEPT**. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)
That acceptance allowed execution only after F5 was adopted and published; detailed exporter history belongs in [F4 study](f4-study.md). (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)

## Superseded and conflicting records

- The 2026-10-25 PT amendment admitted the named 960-life closed training corpus, superseding the 2026-10-24 PT no-eligible-corpus HOLD for that corpus only. (src: docs/research/round30_postf4_20261025/RULING.md:5; docs/research/final_run_20261019/CAMPAIGN_LOG.md:256)
- The amendment’s narrow label-5 exception superseded the original R9-label-4–8 fence only for the distinct rescue-only ranking mechanism; fit/look remained HOLD. (src: docs/research/round30_postf4_20261025/RULING.md:13-19; big-lever/round-30/JUDGMENT.md:66)
- Qualified CLI tree `c1484312` at registry commit `de5bde2d` superseded the earlier `d0d92bdb` transport binding. (src: docs/research/round30_postf4_20261025/RULING.md:95; docs/research/round30_qualification_20261024.md:22-23)
- The no-unit-ceilings amendment superseded 4.8/9.6/14.4-u ceilings while retaining effective wall allowances of 144/288/432 min. (src: docs/research/round30_postf4_20261025/RULING.md:103-105; big-lever/round-30/JUDGMENT.md:24)
- The verified 18-min historical charge superseded the earlier 60-min summed reservation through `HISTORY_CLARIFICATION.json`. (src: docs/research/round30_postf4_20261025/HANDOFF.md:100-105)
- Calibration reservations increased combined effective liability from 432 to 472 to 502 to 542 min without enlarging the 106-min main ceiling. (src: docs/research/round30_postf4_20261025/RULING.md:105; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1406; big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md:31; docs/research/r30_timing_certificate_probe04_20261025.md:18-24)
- Probe-04’s prospective 1,200-s certificate cap replaced 600 s in that packet, but the packet was never admitted and the calibration route then closed. (src: docs/research/r30_timing_certificate_probe04_20261025.md:18-29; big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE04_ADMISSION_ROUTE_CLOSED_20261025.md:5-7)
- F5 draft v2 superseded v1 after REVISE without establishing adoption. (src: big-lever/exp/astra-20261024/F5_DRAFT_PREDATA_REGISTRATION_20261025_v2.md:3-11)
- The expanded 1,160,637-seed collision union superseded the original 3,024-seed inventory as documented coverage, while both annexes retained HOLD. (src: big-lever/exp/astra-20261024/F5_R10A_GAP_CLOSURE_ANNEX_20261026_v2.md:3-19; big-lever/exp/astra-20261024/F5_R10A_METADATA_ANNEX_20261025.md:3-24)
- Probe-03’s identified certificate deadline did not resolve probe-02’s erased failure cause, which remained UNKNOWN in its ruling. (src: big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE03_CERTIFICATE_AMENDMENT_20261025.md:15-25; big-lever/exp/astra-20261024/JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md:5-19)

## Unverified

- Any Round 34 or Round 35 disposition outside its round directory was not verified; the cited QUESTION files do not establish a verdict. (src: big-lever/round-34/QUESTION.md:1-3; big-lever/round-35/QUESTION.md:1)
- Direct receipt-level verification of probe-03’s reported Adam timing of about 0.05 s per ten updates was not performed; the sheet’s cited campaign summary was documentary evidence only. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1428)
- R30 main/R30-2 status, F5 adoption and post-mortem execution after 2026-10-25 17:40 PT were not verified; the last timestamped campaign entry retained F5 adoption HOLD and gated exporter ACCEPT. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1429)
