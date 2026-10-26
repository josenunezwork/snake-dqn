# Colab, RunPod and Mac compute

**Summary:** This note records compute operations and qualification history; new Colab work uses the Mac supervisor, while strict gates and serving qualification remain Mac-only. (src: AGENTS.md:43-45; AGENTS.md:53-59)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/colab-and-compute-infra.md`; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [Governance and owner rules](governance-and-owner-rules.md), [T34 campaign](t34-campaign.md), [F4 study](f4-study.md), [GPU engine and speed](gpu-engine-and-speed.md), [R30–R32 and F5 successors](r30-r32-f5-successors.md)

## Colab CLI operating model

- The 2026-10-18 PT access record identified the official tool as `google-colab-cli` 0.7.4, from `googlecolab/google-colab-cli`, under Apache-2.0. (src: docs/research/colab_agent_access_2026-10-18.md:3-9)
- The 2026-10-18 PT operating advice placed the CLI behind one owner-controlled Mac supervisor, with Claude planning/reviewing and Sol (Codex implementer) building adapters, neither independently allocating workers. (src: docs/research/advice/2026-10-18_colab_cli_operating_model.md:5)
- All new Colab jobs use `research/colab_cli/` as the sole allocator, with explicit `--auth oauth2`, `COLAB_CLI_ASSIGN_RETRIES=0`, unique session names and per-lane state. (src: AGENTS.md:53-59)
- The supervisor counts notebook workers among the maximum three account assignments and refuses unaccounted endpoints. (src: AGENTS.md:53-56; docs/colab-cli.md:88-95)
- The owner setup specified Python 3.12 and pinned installation of `google-colab-cli==0.7.4`, with the executable under `$HOME/.local/bin/colab` and account configuration under `$HOME/.config/colab-cli/snake`. (src: docs/research/advice/2026-10-18_colab_cli_operating_model.md:24-32)
- The 2026-10-18 PT authentication record distinguished the approximately 59-minute access-token expiry from the stored refresh token that renews access. (src: docs/research/colab_agent_access_2026-10-18.md:31-34)
- Runtime liveness follows kernel execution, so a detached background process alone does not keep a runtime alive. (src: docs/research/colab_agent_access_2026-10-18.md:38-40)
- The runtime contract specifies a Python 3.12 uv environment, cu126 for L4/A100/H100, cu128 for Blackwell, fresh output directories and progress about every two minutes. (src: AGENTS.md:61-64)
- A job requires an approved study, a measured local run of its actual entry point, a pinned 40-character commit, known prices and an unused absolute output directory. (src: docs/colab-cli.md:65-68)
- CLI exit status alone is not job success; telemetry and a terminal receipt are required. (src: AGENTS.md:64-65)
- Agents cannot invoke human-interactive `colab auth` or `drivemount`, and CLI runtimes do not mount Drive. (src: docs/colab-cli.md:428-433)
- `launch.sh` checks AC power and an open lid and runs under `caffeinate -is`. (src: docs/colab-cli.md:325-327)
- Release proof requires disappearance of the exact endpoint from account sessions; the documented stop procedure allows up to three attempts with a 75 s allowance, otherwise recording `STOP_UNVERIFIED`. (src: docs/colab-cli.md:120-129)
- Runtime-side release uses a Mac-minted short-lived access token scoped exactly to `colaboratory` and `userinfo.email`, a private mode-0600 credential file, and the GET/XSRF-POST unassignment sequence with a 130 s full release bound. (src: docs/colab-cli.md:293-321)
- The `final-only` acceptance profile qualifies terminal artifacts through an exact 64 MiB verified download, with busy snapshot, exact resume and checkpoint overhead recorded as `NOT_REQUIRED`. (src: docs/colab-cli.md:393-407)
- The `full-state` profile adds a 600 MiB busy snapshot, exact continuation and checkpoint overhead ≤ 2%. (src: docs/colab-cli.md:409-415)
- Resume-by-default requires deterministic exported units and a Mac-verified ledger that skips exactly verified units and refuses foreign namespaces or allocations. (src: AGENTS.md:74-84)
- Dispatch manifests require `resume.skip_verified: true` and `max_outcome_blind_retries: 2`, or an explicit `short-canary` exemption ≤ 40 min or `acceptance` exemption ≤ 60 min. (src: AGENTS.md:86-104)
- Completion attempts require prospectively registered mechanical triggers, verified failure and stop, missing units, and fresh manifests, output directories and receipts; the declaration does not authorize automatic completion retries. (src: AGENTS.md:106-116)
- The 2026-10-20 PT owner rule authorized Colab whenever it speeds work, using the qualified supervisor and L4 by default, while Path B timing/paced tests, strict gates and serving qualification stay on the Mac. (src: docs/STATUS.md:24)
- The later 2026-10-24 PT owner rule removed Colab unit budgets and ceilings, retaining per-job timeouts and pre-registered science caps unless amended before data. (src: docs/STATUS.md:24)

## Qualification history

### General transport qualification

- On 2026-10-18 19:41 PT, CPU attempt `acceptance-1792377693` calibrated 0.08 u/h and verified release, but stalled before readiness while uploading an approximately 166 MiB whole-commit archive. (src: docs/colab-cli.md:29-43)
- Acceptance run 6 on `fe0fd684` ended `INCOMPLETE` after 1,157 s during the 600 MiB snapshot/client-loss phase because the kernel completion file was not observed. (src: docs/colab-cli.md:52-61)
- The 2026-10-19 PT historical `final-only` PASS on `cd766c63` bound an 18-file source inventory to receipt SHA-256 `54e9f2f8…0e101`, with the L4 64 MiB export taking 43.161 s. (src: research/colab_cli/QUALIFICATION.md:47-74)
- On 2026-10-19 PT, `dab00823` passed after two independent pre-execution reviews, with all 10 required gates `OBSERVED` and receipt `acceptance-1792473566`, SHA-256 `dc419e2a…6089`. (src: research/colab_cli/QUALIFICATION.md:180-190)
- That receipt recorded `PASS`, `qualified: true` and return code 0. (src: docs/research/colab_cli_receipts/acceptance-final-only-dab00823.json:442-465)
- Its L4 export transferred 67,108,864 bytes in 75.386 s including Mac verification, a lower bound of 890,203 B/s. (src: docs/research/colab_cli_receipts/acceptance-final-only-dab00823.json:2-12)
- Executable changes to `research/colab_cli` require a new source-bound live final-only acceptance; offline tests do not renew that qualification. (src: research/colab_cli/QUALIFICATION.md:3-5; research/colab_cli/QUALIFICATION.md:34-43)
- On 2026-10-22 approximately 02:30 PT, `e9ce99f2` passed CPU and L4 final-only acceptance with a 67,108,864-byte artifact at ≥ 1.58 MB/s. (src: research/colab_cli/REQUALIFICATION.md:354-357)
- On 2026-10-22 approximately 08:10 PT, `colab-cli-main-acceptance-20261022` passed on main after accounting change `94fc88eb`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:116)
- On 2026-10-22 PT, warmup acceptance passed at `82fb9376`, merged as `96e9b776`, but a subsequent production dispatch failed with `tracked session lacks kernel probe metadata` after a 30.006 s readiness exec, before uploads, with cleanup verified. (src: research/colab_cli/REQUALIFICATION.md:5-18)
- The diagnosis found that CLI 0.7.4 persisted `kernel_id` only after kernel startup/websocket setup; the repair added idle-kernel identity recovery through `/api/kernels` and a `cold-kernel-metadata-recovery` acceptance gate. (src: research/colab_cli/REQUALIFICATION.md:20-51)
- The 2026-10-23 PT registry bound CLI tree `d0d92bdb` to commit `a36a821d` and `acceptance-1792740334`, recorded as `PASS` with `qualified: true`. (src: docs/research/colab_upload_cap_diagnosis_2026-10-24.md:11)
- On 2026-10-24 18:28 PT, transport repair `0bd6f058`, tree `8387bd16`, was accepted and merged as `f628c1af`; it bounded staging control calls to 30 s and recovery to 60 s and removed redundant mkdir/verification work. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1391)
- At 2026-10-24 18:50 PT, `acceptance-1792890832` on `0bd6f058` received `REFUSE`: verified transfer was 207.3 s against a 150 s bar, despite staging taking 220 s against 300 s. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1392)
- On 2026-10-24 20:05–20:45 PT, the `55439bff` throughput repair used two parallel chunk workers and heartbeat renewal, but requalification attempt 2 failed L4 readiness and attempt 3 received `REFUSE` at 221.7 s transfer and 235.5 s staging. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1394-1395)
- On 2026-10-24 22:18 PT, `acceptance-1792902941` passed on `de5bde2d`, with 48.0 s L4 transfer, 57.5 s staging and registry tree `c1484312`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396)
- The D1024 transport exception and canary are covered in [F4 study](f4-study.md); the 2026-10-24 PT record admitted a D1024-only transfer bar ≤ 300 s and subsequently recorded its canary PASS. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1395-1396)

### L4 v2 judgment chain and October closure

- On 2026-10-23 07:55 PT, requalification of `a18bf6b4` produced CPU PASS and L4 `INCOMPLETE` after two bounded 30 s control-RPC timeouts, one unrecovered; this counted as L4's first failed attempt. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:177)
- On 2026-10-24 approximately 09:30 PT, attempt 2 was on `HOLD`, not dispatched, after a census found 225 failed execs out of 5,938 (3.8%) and estimated a 1–8% probability of a fault-free PASS. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:286)
- `JUDGMENT_L4_V2_REQUAL_ATTEMPT2`, dated 2026-10-24 approximately 12:20 PT, ruled `FIX-FIRST` on pack `rpc-qualification-03` at `b5dd7e4b`, identifying six classifier/gate defects against transport tree `4eefe6b1`. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REQUAL_ATTEMPT2.md:1-3; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REQUAL_ATTEMPT2.md:15)
- `JUDGMENT_L4_V2_HOLD_RESOLUTION`, dated 2026-10-24 approximately 13:10 PT, upheld `HOLD` for pack -04 at `9dcc7352` while allowing evidence-only changes supported by a fake-clock differential trace against tree `4eefe6b1`. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_HOLD_RESOLUTION.md:1-3; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_HOLD_RESOLUTION.md:15-16; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_HOLD_RESOLUTION.md:23-30)
- That ruling introduced `EVIDENCE_MISSING`, which refuses admission for absent evidence without closing the lane. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_HOLD_RESOLUTION.md:27-30)
- `JUDGMENT_L4_V2_REVIEW03`, dated 2026-10-24 approximately 14:40 PT, ruled `FIX-FIRST` on pack -05 at `af40d093` because corrupt-export rejection was counted as C6 and client marker files lacked a remote digest. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW03.md:1-3)
- `JUDGMENT_L4_V2_REVIEW04`, dated 2026-10-24 approximately 16:05 PT in its header, ruled checker-only `FIX-FIRST` on pack -06 at `c851ea6b`, tree `364c5d4b`, for false PASS from manifest presence and false FAIL from child-only injection exits. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW04.md:1-3)
- `JUDGMENT_L4_V2_REVIEW05`, dated 2026-10-24 approximately 17:05 PT in its header, ruled `FIX-FIRST` on pack -07 at `1238da63` because `requalify.py` still required superseded proof `V2_06`. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW05.md:1-3)
- On 2026-10-24 PT, `JUDGMENT_L4_V2_ACCEPT` accepted package `86d58c05`, transport pin `411b5890`, tree `364c5d4b`, and pack `rpc-qualification-08`, with 263/263 tests and a lane-local registry while main retained tree `d0d92bdb`. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_ACCEPT.md:1; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_ACCEPT.md:15-20)
- The acceptance record was `big-lever/round-24/CLAUDE_V_L4_REQUAL_02_20261024.json`; the ruling's approximately 18:00 PT header conflicts with the campaign's 2026-10-24 15:05 PT ACCEPT entry. (src: big-lever/exp/fable-20261024/JUDGMENT_L4_V2_ACCEPT.md:3; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_ACCEPT.md:19-20; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1359)
- The terminal campaign entry at 2026-10-24 16:33 PT recorded attempt 2 FAIL with genuine C6: 12 sidecar-download timeouts caused 13 transfer attempts against the frozen limit of 3. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365)
- That closure ruling closed the L4 label lane for October, allowed no third attempt, and assigned D4096 (the 4,096-parent label tranche) to the Mac alone. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365)

## Uploads, transport and telemetry

- Normal source delivery uses a runtime shallow fetch of the pinned public GitHub commit and verifies both commit SHA and Mac-recorded tree ID; only unreachable GitHub permits a sparse fallback bounded to 32 MiB compressed and unpacked. (src: docs/colab-cli.md:96-104; docs/colab-cli.md:200-230)
- The documented input path uses 2 MiB chunks and a shared five-minute initial staging deadline. (src: docs/colab-cli.md:249-255)
- The documented export path uses 8 MiB chunks, a 120 s bound per Contents call and a configurable per-export maximum up to 2 GiB. (src: docs/colab-cli.md:109-119)
- Later transfers receive a fresh deadline of `max(300, 120 + size_bytes / 524288)` s, subject to session limits. (src: docs/colab-cli.md:259-266)
- The exact 64 MiB acceptance measurement does not qualify arbitrary artifact sizes; T34 (the ExIt-B8 campaign) retained its own 64 MiB export bounds. (src: research/colab_cli/QUALIFICATION.md:71-74)
- On 2026-10-21 PT, E3 (an information-value experiment) hit the upload-time cap with 57.8 MB, and E4's subsequent upload-cap failure triggered Z-9 upload-path qualification before checkpoint-carrying manifests. (src: big-lever/LIVE_FACTS.md:17; big-lever/LIVE_FACTS.md:20)
- At 2026-10-21 approximately 07:10 PT, the Z-9 fix recorded a 300 s shared upload deadline, approximately 44 kB/s measured transfer, and runtime checkpoint download/hash verification. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:104)
- On 2026-10-22 PT, a fresh-runtime bootstrap of 3,244 bytes took 134.095 s to verify, including a 120.007 s failed exec; the 177.075 s staging total failed the then-150 s bar. (src: research/colab_cli/REQUALIFICATION.md:74-80)
- The readiness contract used `/content` metadata plus a nonce exec, bounded to 300 s overall, 30 s per call and 5 s between attempts, with `warmup_seconds` recorded. (src: research/colab_cli/REQUALIFICATION.md:86-97)
- The staging correction separated `transfer_seconds`, `source_delivery_seconds` and `elapsed_seconds`, with canary bars of ≤ 150 s verified transfer and ≤ 300 s whole staging. (src: research/colab_cli/REQUALIFICATION.md:100-124)
- At 2026-10-22 09:16 PT, the L4 native/exact-stop ruler main-02 was `INCOMPLETE` after reaching its deadline during telemetry recovery. (src: big-lever/LIVE_FACTS.md:34; research/colab_cli/TELEMETRY_STALL_20261022.md:1-15)
- At 2026-10-22 18:40 PT, the B2 readiness record was `NOT_READY`, citing L4 kernel-RPC stalls in 11 of 12 sessions that day. (src: big-lever/LIVE_FACTS.md:49)
- Five `UPLOAD_CAP` failures between 2026-10-23 20:22 PT and 2026-10-24 00:08 PT led to a pause in new Colab dispatches pending diagnosis. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:215; docs/research/final_run_20261019/CAMPAIGN_LOG.md:233-236)
- The 2026-10-24 PT diagnosis attributed those failures to slow kernel/control operations and recovery exhausting the shared staging clock, with two to four failed approximately 120 s execs per failing job and no Contents upload record for the four terminal files. (src: docs/research/colab_upload_cap_diagnosis_2026-10-24.md:3; docs/research/colab_upload_cap_diagnosis_2026-10-24.md:109)
- In the diagnosis, P1-04 (the PPO repair canary) transferred 28 files/74,138 B in 113.760 s, whereas the GPU move2 canary transferred 268,884 B in 20.049 s. (src: docs/research/colab_upload_cap_diagnosis_2026-10-24.md:60-62)
- P1's 22 sidecars represented 1.6% of bytes but 80.6% of verified-transfer time. (src: docs/research/colab_upload_cap_diagnosis_2026-10-24.md:60-62)
- The diagnosis modeled F small files as F PUTs plus F parent-mkdir execs plus F verification execs. (src: docs/research/colab_upload_cap_diagnosis_2026-10-24.md:129)
- The 300 s staging cap was not a hard bound on every operation in staging because renewals and exec attempts were not clipped to its residual clock. (src: docs/research/colab_upload_cap_diagnosis_2026-10-24.md:121-131)
- R30 timing probe-02 at `773ef710` classified probe-01's `UPLOAD_CAP` as `NON-ATTEMPT`, set `max_export_bytes` to 64 MiB, and recorded a 12-file offline preflight PASS with G5/main on `HOLD`. (src: docs/research/r30_timing_upload_cap_probe02_20261025.md:3-37)
- The 2026-10-25 PT R30 probe-03 handoff was offline with G5/main on `HOLD`, and probe-04 had not achieved its required `dispatch --dry-run` PASS; related records are in [R30–R32 and F5 successors](r30-r32-f5-successors.md). (src: docs/research/r30_timing_observability_probe03_20261025.md:3-4; docs/research/r30_timing_certificate_probe04_20261025.md:3-6)

## Measured limits, rates and balances

- On 2026-10-21 approximately 01:10 PT, the account accepted three L4 assignments and refused a fourth with `precondition failed`, at approximately 0.1 u test cost. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:100)
- The second concurrency test recorded three T4 assignments at 2 vCPU each, but left the CPU-runtime cap `UNMEASURED` after a ReadTimeout and bash-3.2 empty-array error. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:106)
- The 2026-10-19 PT `dab00823` acceptance measured CPU at 0.08 u/h against a priced ceiling of 0.5 u/h and L4 at 1.54 u/h against 2.0 u/h. (src: docs/research/colab_cli_receipts/acceptance-final-only-dab00823.json:24-26; docs/research/colab_cli_receipts/acceptance-final-only-dab00823.json:75-76)
- The ops dispatch record registered L4 at 1.54 u/h and CPU at 0.08 u/h, with A100 using its broker-record rate and observed calibration required instead of substituting a priced ceiling. (src: tools/ops/README.md:30-36)
- The 2026-10-18 PT overnight A100-SXM4-40GB High-RAM runtime had 12 vCPU and an observed allocation rate of approximately 5.3 u/h; the debit was described as idle/allocation cost with only a 40 s check as work. (src: docs/research/overnight_2026-10-18.md:3; docs/research/overnight_2026-10-18.md:23)
- The compute-economics table separately recorded owner planning inputs of 2 u/h for L4 and 6.8 u/h for A100-80GB, and a public H100 figure of approximately 13 u/h. (src: docs/codex-skills/snake-dqn-compute-economics/SKILL.md:78-81)
- A third-party table recorded on 2026-10-18 PT listed T4 1.19, L4 1.71, A100-40GB 5.40, A100-80GB 7.52 and G4 8.71 u/h, with H100 unmeasured. (src: docs/research/colab_agent_access_2026-10-18.md:36-37)
- The 2026-10-20 PT A100 synthesis reported 12 vCPUs/6 physical cores for both A100 and L4 hosts in the records it reviewed and estimated A100 at approximately 3.2× L4's unit cost. (src: docs/research/advice/2026-10-20_a100/SYNTHESIS.md:8-10)
- The units tracker recorded 1,174.37 u on 2026-10-18 PT, then 1,159.1 u at 2026-10-20 13:45 PT, 1,157.6 u at 15:33 PT and 1,153.84 u at approximately 19:20 PT. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:327)
- Later snapshots recorded 1,098.49 u at 2026-10-23 20:14 PT and 1,092.60 → 1,092.43 u across the canary window reported at 23:50 PT. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:212; docs/research/final_run_20261019/CAMPAIGN_LOG.md:233)
- The compute-economics note recorded historical expiry of units 90 days after purchase. (src: docs/codex-skills/snake-dqn-compute-economics/SKILL.md:159)
- The 2026-10-21 PT placement rule permitted A100/H100 when a job's own L4 profile showed approximately ≥ 70% GPU busy time and expected wall-time gain ≥ 2×. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:105)
- The A100 program was funded at 2026-10-22 19:55 PT, followed by the owner's 2026-10-23 18:45 PT standing direction for an always-on A100 GPU-engine lane. (src: big-lever/LIVE_FACTS.md:51; docs/research/final_run_20261019/CAMPAIGN_LOG.md:196)
- The runtime-core probe received `DEFER` on 2026-10-24 approximately 12:30 PT because its endpoint, workload censoring and ledger completion path were defective; it was not funded or dispatch-ready. (src: big-lever/exp/fable-20261024/JUDGMENT_RUNTIME_CORE_PROBE.md:1-3; big-lever/exp/fable-20261024/JUDGMENT_RUNTIME_CORE_PROBE.md:21-25)

## Legacy Drive queue and Q1

- `research/colab_queue` provided pinned jobs through one Drive-backed queue, with one Mac as control plane and owner-owned runtimes as workers. (src: research/colab_queue/README.md:1-12)
- Its documented idle release followed 10 min without eligible work, using `drive.flush_and_unmount` and `runtime.unassign()` with three retries 15 s apart; `IDLE_MINUTES=100000` overrode idle release. (src: research/colab_queue/README.md:178-180; research/colab_queue/README.md:200-208)
- At 2026-10-18 00:36 PT, the overnight A100 worker used `IDLE_MINUTES=100000`, while a leftover CPU High-RAM worker retained the 10 min default. (src: docs/research/overnight_2026-10-18.md:3-5)
- The overnight host-contract check passed 75/75 tests in 38 s, qualifying host contracts only; Q0 audit jobs were refused while the registration remained `DRAFT_PENDING_OWNER_REVIEW`. (src: docs/research/overnight_2026-10-18.md:8-16)
- The legacy exception applies only to the two existing notebook workers and Q1 (the legacy training/evaluation study), with no new non-Q1 jobs, migrations or additional legacy workers. (src: AGENTS.md:68-72)
- On 2026-10-18 PT, Q1's A3 admission was `REFUSED` on forecasts of 29.47 GPU-h against 19.2 usable/24 total GPU-h and 67.11 CPU-h against 32 usable/40 total CPU-h. (src: research/q0q1_20261017/README.md:99-117)
- A4 raised the GPU-hour cap to 37 GPU-h and superseded A3 for dispatch preparation, but enqueued no Q1 jobs. (src: research/q0q1_20261017/README.md:197-202)
- Q1 A5's final 2026-10-18 PT admission was `REFUSED`, with zero manifests, `NOT ENQUEUED` and the L4 released, because forecasts of 105.63 GPU-h, 51.30 CPU-h and 362.55 u exceeded usable limits of 36 GPU-h, 32 CPU-h and 168.63 u. (src: research/q0q1_20261017/Q1_A5_ADMISSION_20261018.md:3-30)

## R33 ops

- R33 (the operations and verification package) closed on 2026-10-23 PT with implementation verified and the CPU pilot canceled under `CANCEL_CPU_PILOT_NO_REMAINING_DEMAND`, without dispatch. (src: research/r33_ops/HANDOFF.md:1-6; research/r33_ops/HANDOFF.md:92)
- R33 delivered `tools/ops/task_journal.py`, a transactional SQLite journal over `watchd.sh`/`wait-event.sh` with explicit notice acknowledgment, five-minute reminders and fifteen-minute escalations. (src: research/r33_ops/HANDOFF.md:14-19)
- Its frozen source was `d53eaef7`, with CLI subtree `d0d92bdb` matching qualified commit `a36a821d`. (src: research/r33_ops/HANDOFF.md:43-51)
- Recorded verification comprised 64 tests in 1.06 s at nice 15, 15 Mac preflight checks across three shards in 3.58 s, and hosted CI run `40057728587` passing all 15 Linux nodes. (src: research/r33_ops/HANDOFF.md:61-74)
- The canceled CPU pilot had a 120 min/0.16 u forecast at 0.08 u/h and an effective 288 min/0.384 u envelope, but used zero assignments and zero units. (src: research/r33_ops/HANDOFF.md:112-118)
- The three resumable R33 verification shards use qualified CLI transport without allocation, dispatch, authentication, Drive mounting or science-admission authority; only the Mac union consumer returns `PASS_LINUX_CONTRACTS`. (src: research/r33_verification/README.md:3-20)
- As of the 2026-10-23 PT handoff, row 4 for A1-PROF (the A100 profiling owner) was `OWNER_HANDOFF_UNACKNOWLEDGED / NATIVE_PENDING`, with a 0.5 A100 h/3.85–5 u forecast and effective 1.2 h/12 u ceiling including 0.2 h/2 u reserve. (src: research/r33_ops/A1_OWNER_HANDOFF.md:26-34)

## RunPod

- The owner lowered the per-step RunPod speed requirement from 5× to ≥ 4× faster than the Mac on 2026-10-05 PT, counting setup and transfers. (src: docs/STATUS.md:78-80; AGENTS.md:38-39)
- Recorded below-threshold waivers were step-specific and did not carry over. (src: docs/STATUS.md:81-85)
- The 2026-10-06 PT rule preferred cheap CPU pods, with serverless only as fallback because observed cost was approximately 6–10× catalog rates. (src: docs/STATUS.md:86-89)
- The historical RunPod skill recorded approximately $0.59 per serverless worker-hour and approximately $10 per-step caps under the 2026-10-12 PT standing permission. (src: docs/codex-skills/snake-dqn-runpod/SKILL.md:30-32; docs/codex-skills/snake-dqn-runpod/SKILL.md:55-56)
- The 2026-10-08 PT status prohibited new pod work until the runner was stable, and the 2026-10-19 PT T34 authority specified no RunPod for T34. (src: docs/STATUS.md:70; docs/STATUS.md:19)
- The RunPod record stated that spot was discontinued in 2026-09 and REST v1 would return 410 on 2026-11-15, with GraphQL end-of-life in early 2027. (src: docs/research/redesign_scope_2026-10-07.md:454; docs/codex-skills/snake-dqn-runpod/SKILL.md:112-113)
- That API notice identified `rp.py`, `pod_agent.self_delete()` and pod bodies as requiring migration to `https://api.runpod.io/v2`. (src: docs/codex-skills/snake-dqn-runpod/SKILL.md:112-113)
- The 2026-10-17 PT self-delete design used an owner-created pods-read/write-only key in Keychain service `runpod-selfdelete`, account `runpod`, delivered in memory through authenticated `POST /self-delete-key`, separate from `runpod-api`. (src: docs/runpod_selfdelete_key.md:3-19)
- A pod-injected `RUNPOD_API_KEY` had returned HTTP 403 for a pod lookup on 2026-10-08 PT, motivating the restricted credential or explicit waiver. (src: docs/codex-skills/snake-dqn-runpod/SKILL.md:102-105)
- The 2026-10-03 PT x86 test matched 28/28 H5000 episodes, with Q-value differences around 5e-5 but zero argmax flips in 256 probes. (src: docs/research/runpod_x86_consistency_2026-10-03.md:8-11; docs/research/runpod_x86_consistency_2026-10-03.md:68-71)
- That cpu3c pod ran live episodes 3.66× slower than the Mac median, charged $0.03/vCPU-h and cost approximately $0.047 over 43.4 min, while nine 32/16/8/4-vCPU create attempts were refused. (src: docs/research/runpod_x86_consistency_2026-10-03.md:85-96)
- The superseding 2026-10-08 PT diagnosis found random Mac/x86 divergence on exact float32 Q ties, approximately one in 50 H10000 episodes, including S2's one failure in 54 episodes. (src: docs/research/runpod_x86_consistency_2026-10-03.md:3; docs/STATUS.md:122-123)
- The strict-on-RunPod amendment was `NOT RATIFIED` and superseded on 2026-10-08 PT; strict gates and serving qualification remain Mac-only. (src: docs/STATUS.md:90-96; AGENTS.md:43-45)

## Mac

- The 2026-10-02 PT compute policy recorded an Apple M5 Pro with 18 CPU cores and 64 GB memory. (src: docs/research/compute_policy_2026-10-02.md:12)
- On 2026-10-21 00:30 PT, the owner fixed serving to that Mac, using CPU or Apple GPU through MPS/Metal, with Colab GPUs reserved for training, labels and evaluation. (src: big-lever/LIVE_FACTS.md:13)
- The status policy requires AC power, an open lid and thermal guarding, with Tier-1 screens using three two-thread slots and strict gates using two slots. (src: docs/STATUS.md:125)
- The earlier thermal-guard design used `pmset -g therm` with failure on errors/unknown readings and a slowdown trigger when the latest-eight median exceeded the first-12 baseline by more than 60%, with 60 s backoff and at most five consecutive backoffs. (src: docs/research/compute_policy_2026-10-02.md:48-78)
- `research/mac_lease.py` defines a create-only exclusive lease for timing/paced work, using `O_EXCL` under an adjacent advisory lock and holder-only removal, with default path `exp/mac-lease.json`. (src: research/mac_lease.py:1-7; research/mac_lease.py:48-50)
- The 2026-10-23 18:55 PT `ALLOW_BOUNDED_CONCURRENCY` ruling allowed Astra (Codex advisor/judge) and Sol at nice ≥ 15 on a shared 0.25-core budget plus one nice-19 single-thread test lane bounded to 120 s wall and 90 CPU-s per test. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:197)
- The 2026-10-23 19:25–19:35 PT `ALLOW_NICE15_INTERIM` record directed the factory to ignore nice ≥ 15 processes and agents/tests to launch through `agent_nice.sh`. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:202-203)
- `agent_nice.sh` sets numeric thread counts to one, forces `SNAKE_DQN_DEVICE=cpu` and re-execs under nice to reach priority 15 when lower. (src: tools/ops/agent_nice.sh:2; tools/ops/agent_nice.sh:10-22)
- At 2026-10-23 20:14 PT, a nice-0 Colab supervisor's children preempted the Mac factory, prompting an interim nice-15 renicer loop every 1 s. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:212)
- `renicer.py` never raises priority and excludes the verified shared tmux server from reprioritization. (src: tools/ops/renicer.py:2-6)
- `dispatch.sh` delegates `--mac` to `mac_run.sh` and otherwise executes `dispatch.py`. (src: tools/ops/dispatch.sh:5-9)
- Colab dispatch refuses BLOCK/HOLD instructions, reused names, failed validation and qualification-tree drift, and serializes allocation with POSIX `flock` under a maximum ten-minute hold. (src: tools/ops/README.md:16-35)
- `mac_run.sh` launches a pinned job in an isolated detached worktree with an in-use lock, AC/open-lid guards and `caffeinate`. (src: tools/ops/mac_run.sh:2-9; tools/ops/README.md:280-291)
- `watchd.sh` records `JOB_ENDED`, `SOL_DONE`, `COLAB_CLI_UNQUALIFIED`, `COLAB_UNDERUSED` after 10 min below three assignments and `MAC_IDLE` after 15 min without a Python worker above 50% CPU. (src: tools/ops/README.md:217-227; tools/ops/watchd.sh:74-86)
- `wait-event.sh` waits up to 3 h using a persisted per-consumer cursor, with task-journal notices delivered at least once until explicitly acknowledged. (src: tools/ops/wait-event.sh:2-17)
- The 2026-10-18 PT OpenMP repair restarted pytest once with torch's `lib` first in `DYLD_LIBRARY_PATH` so torch and scikit-learn loaded one `libomp.dylib`, without setting `KMP_DUPLICATE_LIB_OK`. (src: docs/testing_openmp_macos.md:9-15)
- The same incident disabled `--showlocals` because printing locals after mmap closure caused SIGSEGVs; the final recorded run had 7,904 passed, 26 skipped, eight ordinary failures and zero native crashes in 440.69 s. (src: docs/testing_openmp_macos.md:53-57; docs/testing_openmp_macos.md:87-89)

## Superseded and conflicting records

- The 2026-10-24 22:18 PT qualification of `de5bde2d`/tree `c1484312` followed `a36a821d`/tree `d0d92bdb`, the 2026-10-22 PT main acceptance after `94fc88eb`, `e9ce99f2`, and the 2026-10-19 PT `dab00823` and `cd766c63` passes; each record qualified its own source. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396; docs/research/colab_upload_cap_diagnosis_2026-10-24.md:11; docs/research/final_run_20261019/CAMPAIGN_LOG.md:116; research/colab_cli/REQUALIFICATION.md:354-357; research/colab_cli/QUALIFICATION.md:180-190)
- The L4 v2 terminal outcome was the 2026-10-24 16:33 PT attempt-2 FAIL/October closure; the campaign recorded prerequisite ACCEPT at 15:05 PT, while the ruling headers placed REVIEW04 at approximately 16:05 PT, REVIEW05 at approximately 17:05 PT and ACCEPT at approximately 18:00 PT. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:1365; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1359; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW04.md:3; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW05.md:3; big-lever/exp/fable-20261024/JUDGMENT_L4_V2_ACCEPT.md:3)
- The 2026-10-24 PT no-budget owner rule superseded the earlier approximately 1,160 u/39 u-per-day spending target and unit-budget restrictions, while preserving timeouts and pre-data science-cap amendments; the older 2× rule had doubled caps, reserves and timeouts without doubling rates, N or retry count. (src: docs/STATUS.md:24; AGENTS.md:118-121)
- The 2026-10-19 PT T34 authority superseded the Q0/Q1 budget envelope without changing the historical Q1 A5 refusal. (src: docs/STATUS.md:15-19; research/q0q1_20261017/Q1_A5_ADMISSION_20261018.md:3-30)
- Later ops registration used observed CPU/L4 rates of 0.08/1.54 u/h, whereas the 2026-10-18 PT admission guidance retained priced ceilings of 0.5/2.0 u/h; records carrying a ceiling still distinguish it from observed calibration. (src: tools/ops/README.md:30-36; docs/colab-cli-pricing-notes-2026-10-18.md:28-42; docs/colab-cli.md:375-378)
- The 2026-10-21 PT measured three-L4 limit replaced the 2026-10-20 PT description of three assignments as an unsupported local assumption. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:100; docs/research/final_run_20261019/CAMPAIGN_LOG.md:97)
- The corrected 2026-10-18 PT kernel-execution liveness rule superseded the access note's earlier suggestion to keep a background process running. (src: docs/research/colab_agent_access_2026-10-18.md:38-40; docs/research/colab_agent_access_2026-10-18.md:9-12)
- The 2026-10-23 PT always-on A100 owner direction followed the 2026-10-22 PT funded program and 2026-10-21 PT GPU-bound ≥ 2× placement rule, superseding the 2026-10-20 PT synthesis's no-A100-that-week conclusion. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:196; big-lever/LIVE_FACTS.md:51; docs/research/final_run_20261019/CAMPAIGN_LOG.md:105; docs/research/advice/2026-10-20_a100/SYNTHESIS.md:6-10)
- The 2026-10-23 23:50 PT correction rejected CPU starvation as the explanation for P1's upload failure because a later canary also failed at nice 0 on an idle Mac, superseding the earlier priority hypothesis. (src: docs/research/final_run_20261019/CAMPAIGN_LOG.md:233; docs/research/final_run_20261019/CAMPAIGN_LOG.md:215)
- The 2026-10-08 PT float-tie failures superseded generalization from the 2026-10-03 PT 28/28 Mac/x86 identity sample. (src: docs/research/runpod_x86_consistency_2026-10-03.md:3; docs/STATUS.md:122-123; docs/research/runpod_x86_consistency_2026-10-03.md:8-11)
- T34's 2026-10-19 PT prohibition of RunPod superseded historical pod spending permission for that campaign, and the 2026-10-08 PT status policy superseded the 2026-10-02 PT compute-policy document as current authority. (src: docs/STATUS.md:19; docs/codex-skills/snake-dqn-runpod/SKILL.md:32; docs/research/compute_policy_2026-10-02.md:3; docs/STATUS.md:125)

## Unverified

- Claimed RunPod budgets of $50 or $77.47 and a top-up to $76.95 were not located in the fact sheet's sources; the cited records support a historical balance near $49.94 and approximately $10 per-step caps only. (src: docs/research/runpod_x86_consistency_2026-10-03.md:85-96; docs/codex-skills/snake-dqn-runpod/SKILL.md:32)
- “H100 scarce” was a public/planning statement, not a measured Colab allocation result established by the cited records. (src: docs/codex-skills/snake-dqn-compute-economics/SKILL.md:81; docs/codex-skills/snake-dqn-compute-economics/SKILL.md:154-155)
- Claimed hook enforcement of `tools/ops/dispatch.sh` was not established by the listed sources; the cited ops guide describes wrapper checks rather than the external hook's enforcement rule. (src: tools/ops/README.md:16-35)
- A separate live PASS for the canary-staging correction could not be tied to a receipt: `REQUALIFICATION.md` retained `PENDING — no Colab operations performed`, despite later registry qualifications. (src: research/colab_cli/REQUALIFICATION.md:359-362; docs/research/colab_upload_cap_diagnosis_2026-10-24.md:11; docs/research/final_run_20261019/CAMPAIGN_LOG.md:1396)
- Live-pod verification of the restricted RunPod self-delete key remained unestablished in its 2026-10-17 PT note. (src: docs/runpod_selfdelete_key.md:40-41)
