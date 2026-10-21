# snake-dqn-artifacts inventory (2026-10-20)

Read-only inventory of `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`, taken
2026-10-20 ~19:20 PDT at main `5a5ab525`. Nothing in that tree was moved, deleted or
modified. For `t34-round0-study-01` only the top-level names were listed; nothing
under `chunks/` was opened.

## Disk

| Item | Value |
|---|---|
| Artifacts tree total (`du -sh`) | **249 GiB** (250 top-level folders + 176 loose files) |
| Loose top-level files (codex-t*.err/.out logs, old HANDOFF/prompt .md) | 0.11 GiB |
| Data volume `/System/Volumes/Data` (`df -h`) | 926 GiB size, 838 GiB used, **43 GiB free (96% used)** |

## Totals by class

| Class | Folders | Size |
|---|---:|---:|
| LIVE | 12 | 1.65 GiB |
| RECEIPT | 185 | 148.88 GiB |
| UNKNOWN | 23 | 5.09 GiB |
| SCRATCH | 30 | 93.65 GiB |
| **Total (folders)** | **250** | **249.27 GiB** |

## How classes were assigned

- **LIVE**: one of the named live families (`t34-round0-study-01`, `phase0-search-20261019`,
  `t34-retention-v2-operator-01/02` (02 does not exist yet), `t35-*`, `t34-claim-a-*`,
  `t34-round0d-prep-*`, `fable-panel-*`), or a folder that a running process uses
  (checked with `ps`): `git-sync` (tmux pull-only sync) and `t34-sol` (Codex test-hygiene handoff).
  Running jobs seen: Round 0 chunk 000 supervisor, Phase 0 selection runner, T35
  two-deviation extend (3 procs), Codex test-hygiene, git-sync.
- **RECEIPT**: cited in `docs/research/final_run_20261019/CAMPAIGN_LOG.md` or `HANDOFF_2026-10-20.md`,
  or named by at least one tracked repo file (docs, registrations, results JSON, code), or by
  Claude memory notes. Seed-audit JSONs (`*seed_audit*`, `SEED_AUDIT*`) list every artifact
  folder, so they were left out of the count. Keep all of these.
- **SCRATCH**: not cited anywhere, and the name marks it as a plan/dry-run, retry, refused,
  no-stock, small/dup/preflight/rc attempt, smoke or pilot. Can be rebuilt.
- **UNKNOWN**: not cited, but the name does not show that it can be rebuilt (completed screens,
  calibrations, archives, or today's in-flight temp dirs).

## Main finding: 34 copies of the same 4.06 GB demo bundle

Pod-staging folders each hold an `inputs/` with `demo.tar` (2.0 GB) + 129 `demo-part-*` +
`repo.tar.gz`. In all 33 `*/inputs` with 131 entries, the hashes of `demo-part-000`, `-064` and
`-128` match. Only `repo.tar.gz` differs (45 MB each). One more copy is at
`m6-dev-20261014/m6-gpu-plan/inputs`. That makes about **34 x 4.06 GB = ~138 GB**, more than
half the tree:

- 23 copies are in SCRATCH folders (~93 GB, nearly all of SCRATCH).
- 1 copy is in UNKNOWN (`redesign-m6-gpu-20261014-a4`; its `attempt00/` holds ~50 MB of
  uncited output).
- 10 copies are in RECEIPT folders (`redesign-m3b-gpu-20261008{,-r4}`, `redesign-m3c-gpu-20261008{,-a1,-recovery}`,
  `redesign-m4a-gpu-20261012`, `redesign-m4a2-gpu-20261012{,-g2fail1,-g2fail4}`,
  `redesign-m6-gpu-20261014-oom3`). This is listed for information only. No action is proposed
  for RECEIPT folders.

The other large trees are `ongoing-research-20260913` (54 GiB, the Sept research umbrella,
cited by about 200 files) and `redesign-autonomous-20261010` (19.5 GiB). Both are RECEIPT.

## SCRATCH candidates for owner review (no action taken)

Every SCRATCH row below is a candidate for the owner to review. Before acting on any
`inputs/` bundle, run a full checksum. The three-part hash match above is a strong sign the
copies are identical, but it does not prove it. Freeing the 23 duplicate bundles in SCRATCH
would recover about 93 GB, and the volume has only 43 GiB free. The
UNKNOWN rows should be triaged by the owner first. Do not touch the
`t34-retention-oc-bare-unpaced-*` and `test-suite-hygiene-*` dirs today: they were
created in the last hour and may belong to running work.

## Folders

### LIVE (12 folders, 1.7 GiB)

| Folder | Size | Last modified | What it is | Basis | Example citation |
|---|---:|---|---|---|---|
| `t34-claim-a-20261020` | 0.82G | 2026-10-20 18:58 | T34 Claim A pipeline build/probe outputs | named LIVE family (running job or ratified study) | `research/t34_claim_a/README.md` |
| `phase0-search-20261019` | 0.37G | 2026-10-19 12:58 | Phase 0 search-teacher premise test; selection-v1 runner live (tmux phase0-selection) | named LIVE family (running job or ratified study) | `docs/research/advice/2026-10-20_a100/REVIEW_A_evidence.md` |
| `t35-outcome-head-20261020` | 0.20G | 2026-10-20 18:25 | T35 outcome-head probe (result KILL, 182d85e2) | named LIVE family (running job or ratified study) | `research/t35_outcome_head/PREREGISTRATION.md` |
| `t34-sol` | 0.19G | 2026-10-20 17:24 | Sol/Codex handoff logs, receipts, last-messages for T34 work | running Codex handoff (test-hygiene) writes its log/receipt here | `docs/research/final_run_20261019/CAMPAIGN_LOG.md` |
| `t34-round0-study-01` | 0.06G | 2026-10-20 16:46 | T34 Round 0 ratified study root (manifests, supervisor, chunks/ -- contents not opened) | named LIVE family (running job or ratified study) | `docs/research/advice/2026-10-20_a100/REVIEW_A_evidence.md` |
| `t35-two-deviation-20261020` | 1512K | 2026-10-20 18:59 | T35 two-deviation probe; extend run live (3 procs) | named LIVE family (running job or ratified study) | `docs/research/final_run_20261019/CAMPAIGN_LOG.md` |
| `t34-round0d-prep-20261021` | 1068K | 2026-10-20 18:49 | T34 Round 0-D dense-teacher prep (A6 preflight) | named LIVE family (running job or ratified study) | `research/t34_round0d/README.md` |
| `t34-retention-v2-operator-01` | 64K | 2026-10-20 14:55 | T34 retention v2 operator trace/timing (trace-receipt.json) | named LIVE family (running job or ratified study) | `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` |
| `fable-panel-claima` | 40K | 2026-10-20 18:48 | Fable review panel outputs | named LIVE family (running job or ratified study) |  |
| `fable-panel-2dev` | 36K | 2026-10-20 18:43 | Fable review panel outputs | named LIVE family (running job or ratified study) | `research/t35_two_deviation/PREREGISTRATION.md` |
| `fable-panel-bare-unpaced` | 24K | 2026-10-20 18:31 | Fable review panel outputs | named LIVE family (running job or ratified study) |  |
| `git-sync` | 12K | 2026-10-15 13:39 | tmux pull-only git-sync script + logs | tmux git-sync runs sync.sh from here (PID 2852, up ~3 d) | `docs/codex-skills/snake-dqn-git-workflow/SKILL.md` |

### RECEIPT (185 folders, 148.9 GiB)

| Folder | Size | Last modified | What it is | Basis | Example citation |
|---|---:|---|---|---|---|
| `ongoing-research-20260913` | 54.31G | 2026-10-08 16:38 | Sept 2026 umbrella research tree (apex canonical retention, coverage, controlled-encounter learning, scaling, horizon); cited by docs/experiments/* READMEs | cited by 207 tracked file(s) | `docs/README.md` |
| `redesign-autonomous-20261010` | 19.51G | 2026-10-12 15:32 | Oct 10-12 autonomous redesign campaign (M3-C/M3-D diagnostics; MC policy / return-target resources ~8.7 GB) | cited by 5 tracked file(s) | `docs/research/m3c_m3d_learnings_2026-10-12.md` |
| `s1-mature-20261018` | 5.02G | 2026-10-19 10:03 | S1 Colab-CLI mature run: admission, barrier fixes, dispatch, settlement repair (Oct 18-19) | cited by 60 tracked file(s) | `docs/research/specs/2026-10-19_phase0_search_teacher.md` |
| `redesign-m3c-gpu-20261008-recovery` | 4.88G | 2026-10-09 05:19 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 2 tracked file(s) | `docs/research/m3c_m3d_evidence/retry-recovery-closeout.json` |
| `frp-v5h-20261008` | 4.63G | 2026-10-07 16:26 | FRP fine-tune round (train / pods / fidelity / Phase R) | cited by 2 tracked file(s) | `research/q0q1_20261017/results/q0_mass_audit_receipt.json` |
| `redesign-m3b-gpu-20261008-r4` | 4.30G | 2026-10-08 06:58 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 19 tracked file(s) | `docs/research/m3c_design_2026-10-08.md` |
| `redesign-m4a-gpu-20261012` | 4.27G | 2026-10-13 00:22 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 3 tracked file(s) | `research/perf_ram_20261013/results/podtel_20261013.json` |
| `redesign-m6-gpu-20261014-oom3` | 4.23G | 2026-10-14 05:41 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 1 tracked file(s) | `docs/research/t4_pod_plan_2026-10-15.md` |
| `redesign-m3c-gpu-20261008-a1` | 4.19G | 2026-10-08 22:46 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 2 tracked file(s) | `research/perf_ram_20261013/results/podtel_20261013.json` |
| `redesign-m4a2-gpu-20261012-g2fail4` | 4.10G | 2026-10-13 00:25 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 1 tracked file(s) | `research/perf_ram_20261013/results/podtel_20261013.json` |
| `redesign-m3c-gpu-20261008` | 4.08G | 2026-10-08 20:01 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 4 tracked file(s) | `docs/research/m3c_m3d_evidence/retry-recovery-closeout.json` |
| `redesign-m4a2-gpu-20261012` | 4.06G | 2026-10-13 02:13 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 2 tracked file(s) | `research/perf_ram_20261013/results/podtel_20261013.json` |
| `redesign-m4a2-gpu-20261012-g2fail1` | 4.06G | 2026-10-12 22:15 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 1 tracked file(s) | `research/perf_ram_20261013/results/podtel_20261013.json` |
| `redesign-m3b-gpu-20261008` | 4.06G | 2026-10-08 05:31 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | cited by 20 tracked file(s) | `docs/research/m3c_design_2026-10-08.md` |
| `frp-v4-20261007` | 3.84G | 2026-10-07 10:47 | FRP fine-tune round (train / pods / fidelity / Phase R) | cited by 3 tracked file(s) | `research/q0q1_20261017/results/q0_mass_audit_receipt.json` |
| `redesign-m2b-20261008` | 2.17G | 2026-10-08 00:44 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 118 tracked file(s) | `docs/research/panel_2026-10-15/librarian_answers.md` |
| `redesign-m2-20261008` | 1.91G | 2026-10-07 19:40 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 35 tracked file(s) | `docs/research/redesign_scope_2026-10-07.md` |
| `colab-cli-acceptance` | 1.41G | 2026-10-19 01:19 | Colab CLI qualification / acceptance / robustness work | cited by 11 tracked file(s) | `docs/colab-cli-pricing-notes-2026-10-18.md` |
| `frp-v2-20261004` | 1.14G | 2026-10-05 02:19 | FRP fine-tune round (train / pods / fidelity / Phase R) | cited by 6 tracked file(s) | `research/frp3_strict_20261005/preflight.py` |
| `frp-v5s2-20261009` | 1.02G | 2026-10-07 17:34 | FRP fine-tune round (train / pods / fidelity / Phase R) | cited by 3 tracked file(s) | `docs/research/INDEX.md` |
| `four-hour-20260913` | 1.00G | 2026-09-13 00:08 | Four-hour experiment wave (docs/experiments/four_hour_2026-09-13) | cited by 4 tracked file(s) | `docs/experiments/four_hour_2026-09-13/EXPERIMENT_LEDGER.md` |
| `t4b-snapshot-restart-20261016` | 0.69G | 2026-10-15 19:40 | Gap/post-release campaign task output (T1-T23) | cited by 12 tracked file(s) | `docs/research/specs/t5_separability_spec_2026-10-16.md` |
| `t4a-posthoc-20261016` | 0.54G | 2026-10-15 22:27 | Gap/post-release campaign task output (T1-T23) | cited by 8 tracked file(s) | `docs/research/t4a_posthoc_2026-10-16_evidence/boost-summary.json` |
| `t34-engine-fanout-20261020` | 0.51G | 2026-10-20 15:33 | Engine fan-out L0 Colab run (run-l0.sh) | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/advice/2026-10-20_a100/astra_p1_systems.md` |
| `redesign-t3-mac-20261015` | 0.46G | 2026-10-15 18:28 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 22 tracked file(s) | `docs/research/specs/t3_probe_spec_2026-10-15.md` |
| `redesign-m6-mac-20261014` | 0.46G | 2026-10-14 12:26 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 54 tracked file(s) | `docs/research/advice/2026-10-15_t3_launch_review.md` |
| `redesign-t2-mac-20261015` | 0.45G | 2026-10-15 02:37 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 3 tracked file(s) | `docs/research/specs/t2_probe_spec_2026-10-15.md` |
| `t4c-drift-20261016` | 0.45G | 2026-10-15 23:44 | Gap/post-release campaign task output (T1-T23) | cited by 6 tracked file(s) | `docs/research/t4c_drift_diagnostic_2026-10-16.md` |
| `redesign-t4c-probes-20261016` | 0.33G | 2026-10-16 10:20 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 5 tracked file(s) | `research/t3_screen_20261015/T4C_README.md` |
| `redesign-t2b-mac-20261015` | 0.31G | 2026-10-15 06:21 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 2 tracked file(s) | `docs/research/specs/t2b_delta_sweep_spec_2026-10-15.md` |
| `t4e-return-bank-20261016` | 0.28G | 2026-10-16 12:49 | Gap/post-release campaign task output (T1-T23) | cited by 32 tracked file(s) | `docs/research/specs/t5_separability_spec_2026-10-16.md` |
| `redesign-m4b-mac-20261013` | 0.28G | 2026-10-13 08:09 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 87 tracked file(s) | `docs/codex-skills/snake-dqn-serving-release/SKILL.md` |
| `rl-repair-20260908` | 0.27G | 2026-09-09 07:36 | Experiment output | cited by 2 tracked file(s) | `docs/experiments/rl_repair_execution_2026-09-08.md` |
| `frp-v3-20261005` | 0.26G | 2026-10-05 12:49 | FRP fine-tune round (train / pods / fidelity / Phase R) | cited by 25 tracked file(s) | `docs/research/frp3_strict_result_2026-10-06.md` |
| `t34-exit-fit-prep-20261019` | 0.26G | 2026-10-19 18:06 | Gap/post-release campaign task output (T1-T23) | cited by 4 tracked file(s) | `research/t34_exit_b8/VERIFICATION.json` |
| `t15-mixed-roster-20261017` | 0.19G | 2026-10-17 07:08 | Gap/post-release campaign task output (T1-T23) | cited by 12 tracked file(s) | `docs/research/t15_mixed_roster_qualification_2026-10-17.md` |
| `redesign-t4c-mac-20261016` | 0.19G | 2026-10-16 09:47 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 15 tracked file(s) | `docs/research/specs/t4c_prereg_2026-10-16.md` |
| `t34-round0-integrity-20261020` | 0.18G | 2026-10-20 16:38 | Round 0 A6 integrity preflights (a6-fixes1-01) | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/final_run_20261019/CAMPAIGN_LOG.md` |
| `mass-objective-20260913` | 0.18G | 2026-09-13 01:00 | Experiment output | cited by 1 tracked file(s) | `docs/experiments/mass_objective_2026-09-13/README.md` |
| `t34-pathb-v2` | 0.18G | 2026-10-20 02:33 | Gap/post-release campaign task output (T1-T23) | cited by 23 tracked file(s) | `docs/research/specs/2026-10-20_t34_retention_prereg.md` |
| `t34-shadow-batch-20261020` | 0.17G | 2026-10-20 13:15 | A5 #2 teacher benchmark lanes (final-l4-*) | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/advice/2026-10-20_t34_a5_speed_QUESTION.md` |
| `redesign-t4a-mac-20261016` | 0.17G | 2026-10-15 21:19 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 20 tracked file(s) | `docs/research/specs/t4a_posthoc_spec_2026-10-16.md` |
| `pqn-followup-20260909` | 0.17G | 2026-10-05 09:54 | Experiment output | cited by 30 tracked file(s) | `docs/codex-skills/snake-dqn-mac-ops/SKILL.md` |
| `m7-diagnostics-20261014` | 0.17G | 2026-10-14 17:16 | Experiment output | cited by 7 tracked file(s) | `docs/research/specs/t1c_fuse_probe_spec_2026-10-15.md` |
| `redesign-t6-mac-20261016` | 0.16G | 2026-10-16 19:15 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 11 tracked file(s) | `docs/research/specs/t6_prereg_2026-10-16.md` |
| `t34-shadow-20261019` | 0.14G | 2026-10-19 23:54 | Gap/post-release campaign task output (T1-T23) | cited by 7 tracked file(s) | `docs/research/final_run_20261019/speed_advice/m2m3_QUESTION.md` |
| `redesign-t11-mac-20261017` | 0.14G | 2026-10-17 00:47 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 15 tracked file(s) | `docs/research/specs/t11_local30_spec_2026-10-17.md` |
| `redesign-m5-mac-20261013` | 0.14G | 2026-10-13 17:40 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 2 tracked file(s) | `research/perf_eval_20261013/results/noisy_mac_20261013.json` |
| `t3-rollout-labels-20261015` | 0.12G | 2026-10-15 13:22 | Gap/post-release campaign task output (T1-T23) | cited by 51 tracked file(s) | `docs/research/advice/2026-10-19_t34_speed_astra_search.md` |
| `apex-veto-strict-20260927` | 0.11G | 2026-09-30 17:26 | Apex veto strict-gate evaluation | cited by 17 tracked file(s) | `docs/experiments/apex_veto_strict_gate_2026-09-27/README.md` |
| `perf-ab-k2-20261014` | 0.11G | 2026-10-14 00:29 | Performance benchmark / A-B probes | cited by 1 tracked file(s) | `research/perf_quiet_20261013/RESULTS.md` |
| `redesign-m4a-mac-c34-20261013` | 0.11G | 2026-10-13 02:35 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 1 tracked file(s) | `research/redesign_m4_20261013/PREREGISTRATION_M4.md` |
| `t1-fuse-probe-20261015` | 0.11G | 2026-10-14 22:34 | Gap/post-release campaign task output (T1-T23) | cited by 9 tracked file(s) | `docs/research/specs/t1c_fuse_probe_spec_2026-10-15.md` |
| `t4e-calibration-20261016` | 0.09G | 2026-10-16 13:27 | Gap/post-release campaign task output (T1-T23) | cited by 4 tracked file(s) | `docs/research/specs/t4e_prereg_amendment1_2026-10-16.md` |
| `t10-certificate-horizon-20261017` | 0.09G | 2026-10-16 23:16 | Gap/post-release campaign task output (T1-T23) | cited by 6 tracked file(s) | `docs/research/t10_certificate_horizon_2026-10-17.md` |
| `frp-v5s-20261009` | 0.09G | 2026-10-07 08:03 | FRP fine-tune round (train / pods / fidelity / Phase R) | cited by 2 tracked file(s) | `docs/research/INDEX.md` |
| `q1-a4-20261018-r2` | 0.08G | 2026-10-18 18:29 | Q0/Q1 Colab factorial (Drive-queue era) runs | cited by 3 tracked file(s) | `research/q0q1_20261017/Q1_A4_ADMISSION_20261018.md` |
| `redesign-t7-phase-r-b8-20261016` | 0.08G | 2026-10-17 00:01 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 10 tracked file(s) | `docs/research/t7_phase_r_result_2026-10-16.md` |
| `t18-h100pod-20261017` | 0.08G | 2026-10-17 14:36 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `docs/codex-skills/snake-dqn-runpod/SKILL.md` |
| `t34-v2-teacher-20261020` | 0.08G | 2026-10-20 02:01 | Gap/post-release campaign task output (T1-T23) | cited by 4 tracked file(s) | `research/t34_exit_b8/README_V2.md` |
| `lh1-screen-frp3-s12` | 0.08G | 2026-10-05 15:38 | LH-1 calibration / screen | cited by 7 tracked file(s) | `docs/research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md` |
| `t5b-risk-and-fuse-20261016` | 0.07G | 2026-10-16 16:21 | Gap/post-release campaign task output (T1-T23) | cited by 8 tracked file(s) | `docs/research/t10_certificate_horizon_2026-10-17_evidence/horizon.json` |
| `death-census-v8-20261003` | 0.07G | 2026-10-03 10:47 | Death census analysis | cited by 2 tracked file(s) | `docs/research/death_census_v8_2026-10-03.md` |
| `colab-cli-carryover-dab00823` | 0.06G | 2026-10-19 22:19 | Colab CLI qualification / acceptance / robustness work | cited by 4 tracked file(s) | `docs/research/colab_cli_receipts/acceptance-final-only-dab00823.json` |
| `runpod-fanout` | 0.06G | 2026-10-17 14:45 | RunPod fan-out / cross-check / stock monitor | cited by 17 tracked file(s) | `docs/codex-skills/snake-dqn-runpod/SKILL.md` |
| `t16-colab` | 0.06G | 2026-10-17 10:02 | Gap/post-release campaign task output (T1-T23) | cited by 168 tracked file(s) | `docs/claude-skills/colab-jobs/SKILL.md` |
| `q0q1-20261017` | 0.06G | 2026-10-18 14:45 | Q0/Q1 Colab factorial (Drive-queue era) runs | cited by 153 tracked file(s) | `docs/codex-skills/snake-dqn-queue-jobs/SKILL.md` |
| `t5-separability-20261016` | 0.06G | 2026-10-16 16:11 | Gap/post-release campaign task output (T1-T23) | cited by 23 tracked file(s) | `docs/research/t10_certificate_horizon_2026-10-17_evidence/horizon.json` |
| `t4f-return-bank-v3-20261016` | 0.05G | 2026-10-16 15:22 | Gap/post-release campaign task output (T1-T23) | cited by 16 tracked file(s) | `docs/research/specs/t4f_prereg_addendum2_2026-10-16.md` |
| `apex-veto-v3-screen-20261001` | 0.05G | 2026-10-01 04:46 | Apex free-space veto variant screen | cited by 2 tracked file(s) | `docs/experiments/apex_veto_serving_and_v3_2026-10-01/README.md` |
| `apex-veto-v8-screen-20261002` | 0.05G | 2026-10-02 22:13 | Apex free-space veto variant screen | cited by 15 tracked file(s) | `docs/experiments/apex_veto_v8_2026-10-03/README.md` |
| `t4c-data-20261016` | 0.05G | 2026-10-15 23:35 | Gap/post-release campaign task output (T1-T23) | cited by 18 tracked file(s) | `docs/research/t4c_cpu_smoke_2026-10-16.json` |
| `t4f-calibration-addendum2-20261016` | 0.05G | 2026-10-16 15:34 | Gap/post-release campaign task output (T1-T23) | cited by 4 tracked file(s) | `docs/research/specs/t4f_prereg_addendum2_2026-10-16.md` |
| `apex-veto-v8-lambda-sweep-20261002` | 0.05G | 2026-10-02 20:31 | Apex veto parameter sweep | cited by 8 tracked file(s) | `docs/experiments/apex_veto_v8_2026-10-03/README.md` |
| `t4f-return-bank-20261016` | 0.04G | 2026-10-16 14:28 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `docs/research/specs/t4f_prereg_2026-10-16.md` |
| `decision-phase-20260912` | 0.04G | 2026-09-12 19:45 | Experiment output | cited by 4 tracked file(s) | `docs/experiments/rl_research_portfolio_2026-09-12/EXPERIMENT_LEDGER.md` |
| `perf-cuda-bench-20261013-try2` | 0.04G | 2026-10-13 22:45 | Performance benchmark / A-B probes | cited by 2 tracked file(s) | `docs/codex-skills/snake-dqn-runpod/SKILL.md` |
| `perf-cuda-compile-20261014-try1` | 0.04G | 2026-10-13 23:18 | Performance benchmark / A-B probes | cited by 1 tracked file(s) | `research/perf_quiet_20261013/RESULTS.md` |
| `perf-cuda-bench-20261013` | 0.04G | 2026-10-13 22:30 | Performance benchmark / A-B probes | cited by 2 tracked file(s) | `docs/codex-skills/snake-dqn-runpod/SKILL.md` |
| `t4f-calibration-20261016` | 0.04G | 2026-10-16 14:37 | Gap/post-release campaign task output (T1-T23) | cited by 4 tracked file(s) | `docs/research/specs/t4f_prereg_2026-10-16.md` |
| `apex-safety-screen-20260926` | 0.03G | 2026-09-27 07:20 | Apex safety veto screen (first) | cited by 7 tracked file(s) | `docs/experiments/apex_safety_veto_screen_2026-09-27/README.md` |
| `t1-commitment-20261015` | 0.03G | 2026-10-14 22:40 | Gap/post-release campaign task output (T1-T23) | cited by 10 tracked file(s) | `docs/research/specs/t1a_commitment_spec_2026-10-15.md` |
| `apex-veto-v5-strict-20261001` | 0.03G | 2026-10-01 14:03 | Apex veto strict-gate evaluation | cited by 14 tracked file(s) | `docs/experiments/apex_veto_v5_2026-10-01/README.md` |
| `t34-unpaced-colab-20261020` | 0.03G | 2026-10-20 13:53 | Unpaced Path B arm on Colab | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` |
| `apex-veto-v7-strict-20261002` | 0.03G | 2026-10-02 13:59 | Apex veto strict-gate evaluation | cited by 26 tracked file(s) | `docs/experiments/apex_veto_v7_2026-10-02/README.md` |
| `t34-v2-fitcache-20261020` | 0.03G | 2026-10-20 00:08 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `docs/research/advice/2026-10-20_a100/astra_p3_distill.md` |
| `oracle-rescue-20261004` | 0.03G | 2026-10-04 11:15 | Experiment output | cited in Claude memory notes |  |
| `trap-horizon-v5-20261002` | 0.02G | 2026-10-01 21:16 | Experiment output | cited by 2 tracked file(s) | `docs/research/death_census_v5_2026-10-02.md` |
| `trap-horizon-20261001` | 0.02G | 2026-10-01 10:47 | Experiment output | cited by 2 tracked file(s) | `docs/research/trap_horizon_2026-10-01.md` |
| `t34-pathb-20261019` | 0.02G | 2026-10-19 20:50 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `research/t34_pathb/README.md` |
| `apex-veto-v6-screen-20261002` | 0.02G | 2026-10-02 07:50 | Apex free-space veto variant screen | cited by 2 tracked file(s) | `docs/experiments/apex_veto_v6_2026-10-02/README.md` |
| `apex-veto-v7-screen-20261002` | 0.02G | 2026-10-02 11:17 | Apex free-space veto variant screen | cited by 11 tracked file(s) | `docs/experiments/apex_veto_v7_2026-10-02/README.md` |
| `apex-veto-v4-screen-20261001` | 0.02G | 2026-10-01 08:34 | Apex free-space veto variant screen | cited by 2 tracked file(s) | `docs/experiments/apex_veto_serving_and_v3_2026-10-01/README.md` |
| `apex-veto-v5-screen-20261001` | 0.02G | 2026-10-01 12:21 | Apex free-space veto variant screen | cited by 7 tracked file(s) | `docs/experiments/apex_veto_v5_2026-10-01/README.md` |
| `t8-probes-20261017` | 0.02G | 2026-10-16 22:15 | Gap/post-release campaign task output (T1-T23) | cited by 12 tracked file(s) | `docs/research/t10_certificate_horizon_2026-10-17_evidence/horizon.json` |
| `frp3-m3s12-strict-20261005` | 0.02G | 2026-10-05 18:32 | FRP-v3 s12 strict gate / serving evidence | cited by 12 tracked file(s) | `docs/research/frp3_serving_result_and_release_2026-10-06.md` |
| `apex-veto-v7-lambda-sweep-20261002` | 0.02G | 2026-10-02 10:26 | Apex veto parameter sweep | cited by 11 tracked file(s) | `docs/experiments/apex_veto_v7_2026-10-02/README.md` |
| `t12-b8-serving-20261017` | 0.02G | 2026-10-17 01:20 | Gap/post-release campaign task output (T1-T23) | cited by 10 tracked file(s) | `docs/research/t12_b8_serving_qualification_2026-10-17.md` |
| `simd-h5000-parity-20261001` | 0.02G | 2026-10-01 09:08 | SIMD sim parity check | cited by 2 tracked file(s) | `docs/research/simd_vector61_plan_2026-09-26.md` |
| `t4f-heldout-outside-v2-20261016` | 0.01G | 2026-10-16 14:01 | Gap/post-release campaign task output (T1-T23) | cited by 9 tracked file(s) | `docs/research/specs/t4f_prereg_2026-10-16.md` |
| `redesign-m3-20261008` | 0.01G | 2026-10-08 06:52 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 1 tracked file(s) | `research/redesign_m3_20261008/results/m3a/summary.json` |
| `redesign-t4f-mac-20261016` | 0.01G | 2026-10-16 15:40 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 14 tracked file(s) | `docs/research/specs/t4f_prereg_2026-10-16.md` |
| `t16-mac-baseline` | 0.01G | 2026-10-17 10:58 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `docs/research/t16_mac_baseline_2026-10-17.md` |
| `t34-round0-20261020` | 0.01G | 2026-10-20 13:28 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `research/t34_round0/VERIFICATION.md` |
| `redesign-m4-phase-r-nv-20261013` | 0.01G | 2026-10-13 13:54 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 7 tracked file(s) | `docs/research/m4_phase_r_result_2026-10-13.md` |
| `vision-check-20261004` | 0.01G | 2026-10-04 10:14 | Experiment output | cited in Claude memory notes |  |
| `apex-veto-v8-strict-20261003` | 0.01G | 2026-10-02 22:52 | Apex veto strict-gate evaluation | cited by 24 tracked file(s) | `docs/experiments/apex_veto_v8_2026-10-03/README.md` |
| `redesign-m3-phase-r-20261008` | 0.01G | 2026-10-08 16:08 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 7 tracked file(s) | `docs/research/panel_2026-10-15/librarian_answers.md` |
| `t4e-bugcheck-20261016` | 9912K | 2026-10-16 13:49 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `docs/research/t4e_bugcheck_2026-10-16.md` |
| `governance` | 8648K | 2026-10-10 15:51 | Advisor / panel / governance notes | cited by 83 tracked file(s) | `AGENTS.md` |
| `t4-prep-20261015` | 7928K | 2026-10-15 15:18 | Gap/post-release campaign task output (T1-T23) | cited by 32 tracked file(s) | `docs/research/specs/t11_local30_spec_2026-10-17.md` |
| `t4d-build-20261016` | 7552K | 2026-10-16 09:46 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `docs/research/t4_return_regression_tooling_2026-10-16.md` |
| `t12b-check4-20261017` | 7324K | 2026-10-17 01:34 | Gap/post-release campaign task output (T1-T23) | cited by 10 tracked file(s) | `docs/research/t12_b8_serving_qualification_2026-10-17.md` |
| `redesign-t4a-probes-20261016` | 7216K | 2026-10-15 22:40 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 9 tracked file(s) | `docs/research/t4a_probe_launch_2026-10-16.json` |
| `redesign-t3-probes-20261015` | 7040K | 2026-10-15 21:34 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 18 tracked file(s) | `docs/research/specs/t3_diagnostics_spec_2026-10-16.md` |
| `t13b-six-snake-20261017` | 6792K | 2026-10-17 06:00 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `docs/research/t13b_six_snake_roster_2026-10-17_results.md` |
| `t34-a5-20261020` | 6668K | 2026-10-20 12:07 | A5 v2 benchmark on 2 L4 lanes | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/advice/2026-10-20_compute_portfolio_astra.md` |
| `t16-mac-n3-baseline` | 6516K | 2026-10-17 14:53 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `docs/research/t16_colab_smoke02_results_2026-10-17.md` |
| `t13-product-roster-20261017` | 6352K | 2026-10-17 05:17 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `docs/research/t13_product_roster_2026-10-17.md` |
| `t9-divergence-20261017` | 5180K | 2026-10-16 22:10 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `docs/research/t9_divergence_2026-10-17.md` |
| `compute-portfolio-20261020` | 4820K | 2026-10-20 13:45 | Compute portfolio Astra proposals + FABLE_JUDGMENT.md | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` |
| `a100-advice` | 4804K | 2026-10-20 17:32 | A100 advisor outputs (docs/research/advice/2026-10-20_a100) | cited by 5 tracked file(s) | `docs/research/advice/2026-10-20_a100/astra_p1_systems.md` |
| `observation-value-20260912` | 4452K | 2026-09-12 19:01 | Experiment output | cited by 3 tracked file(s) | `docs/experiments/observation_value_2026-09-12/PROTOCOL.md` |
| `research-tracks-20261014` | 4380K | 2026-10-14 12:28 | Advisor / panel / governance notes | cited by 4 tracked file(s) | `docs/research/t5b_risk_and_fuse_2026-10-16_evidence/risk-inputs.json` |
| `t34-round0-fixes1-20261020` | 4148K | 2026-10-20 16:26 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `research/t34_round0/VERIFICATION_FIXES1.json` |
| `q0-init-fill-20261018` | 4076K | 2026-10-18 14:33 | Q0/Q1 Colab factorial (Drive-queue era) runs | cited by 1 tracked file(s) | `research/q0q1_20261017/Q0_INIT_A2_R6_20261018.md` |
| `redesign-t2-probes-20261015` | 3512K | 2026-10-15 07:22 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 12 tracked file(s) | `docs/research/specs/t2_probe_spec_2026-10-15.md` |
| `redesign-t6-probes-20261017` | 3196K | 2026-10-16 21:34 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 4 tracked file(s) | `docs/research/t6_screen_result_2026-10-17.md` |
| `b8-death-census-20261014` | 2748K | 2026-10-14 02:34 | Death census analysis | cited by 15 tracked file(s) | `docs/research/panel_2026-10-15/librarian_answers.md` |
| `runpod-xcheck-20261003` | 2412K | 2026-10-03 10:38 | RunPod fan-out / cross-check / stock monitor | cited by 1 tracked file(s) | `docs/research/runpod_x86_consistency_2026-10-03.md` |
| `t34-exit-engine-prep-20261019-v1` | 2096K | 2026-10-19 19:58 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `research/t34_exit_b8/README.md` |
| `t34-speed-advice` | 1784K | 2026-10-19 19:45 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `docs/research/final_run_20261019/speed_advice/QUESTION_GPU.md` |
| `t33-batched-teacher` | 1752K | 2026-10-19 16:34 | Gap/post-release campaign task output (T1-T23) | cited by 20 tracked file(s) | `docs/research/advice/2026-10-19_t34_speed_astra_perf.md` |
| `t34-retention-resize-20261020` | 1716K | 2026-10-20 12:39 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `research/t34_pathb/VERIFICATION_RETENTION_RESIZE.md` |
| `t4e-validation-20261016` | 1508K | 2026-10-16 13:06 | Gap/post-release campaign task output (T1-T23) | cited by 6 tracked file(s) | `docs/research/t4e_collection_preflight_repair_2026-10-16.json` |
| `redesign-t3-curve-probes-20261015` | 1456K | 2026-10-15 18:58 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 7 tracked file(s) | `docs/research/t4_update_every_and_curve_2026-10-15.md` |
| `t4f-validation-20261016` | 1432K | 2026-10-16 14:37 | Gap/post-release campaign task output (T1-T23) | cited by 5 tracked file(s) | `docs/research/specs/t4f_prereg_2026-10-16.md` |
| `redesign-m3-phase-r-SMOKE2-20261008` | 1328K | 2026-10-08 14:13 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 3 tracked file(s) | `research/redesign_m3_20261008/results/phase_r_smoke2/audit.json` |
| `research-portfolio-20260912` | 1280K | 2026-09-12 19:46 | Advisor / panel / governance notes | cited by 5 tracked file(s) | `docs/experiments/four_hour_2026-09-13/NEXT_WAVES.md` |
| `redesign-m3-phase-r-SMOKE-20261008` | 1212K | 2026-10-08 13:53 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 2 tracked file(s) | `research/redesign_m3_20261008/results/phase_r_smoke/audit.json` |
| `redesign-m3c-phase-r-smoke-nv-20261008` | 1120K | 2026-10-08 19:40 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 1 tracked file(s) | `research/redesign_m3c_20261008/results/phase_r_smoke_nv.json` |
| `openmp-tests-20261018` | 1012K | 2026-10-18 20:40 | OpenMP test outputs | cited by 1 tracked file(s) | `docs/testing_openmp_macos.md` |
| `t34-retention-fixes3-20261020` | 916K | 2026-10-20 14:05 | Gap/post-release campaign task output (T1-T23) | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` |
| `t34-retention-fixes-20261020` | 692K | 2026-10-20 12:27 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `research/t34_pathb/VERIFICATION_RETENTION_FIXES.md` |
| `t34-a5-speed-review` | 640K | 2026-10-20 13:55 | A5 #2 speed review (Astra + Fable fable_systems.md) | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/advice/2026-10-20_a100/astra_p1_systems.md` |
| `frp3-s12-serving-20261006` | 592K | 2026-10-06 00:24 | FRP-v3 s12 strict gate / serving evidence | cited by 1 tracked file(s) | `docs/research/frp3_serving_result_and_release_2026-10-06.md` |
| `apex-veto-v8-serving-20261003` | 588K | 2026-10-03 06:56 | Apex veto serving release check | cited by 8 tracked file(s) | `docs/experiments/apex_veto_v8_2026-10-03/README.md` |
| `t34-retention-fixes2-20261020` | 580K | 2026-10-20 13:00 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `research/t34_pathb/VERIFICATION_RETENTION_FIXES2.md` |
| `t34-retention-fixes4-20261020` | 564K | 2026-10-20 14:18 | Gap/post-release campaign task output (T1-T23) | cited in CAMPAIGN_LOG/HANDOFF | `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` |
| `t34-retention-fixes5-20261020` | 564K | 2026-10-20 14:30 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `research/t34_pathb/VERIFICATION_RETENTION_FIXES5.md` |
| `t34-retention-fixes6-20261020` | 564K | 2026-10-20 14:45 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `research/t34_pathb/README_V2.md` |
| `q1-a5-20261018` | 540K | 2026-10-18 19:04 | Q0/Q1 Colab factorial (Drive-queue era) runs | cited by 2 tracked file(s) | `research/q0q1_20261017/Q1_A4_ALLOCATION_CORRECTION_20261018.md` |
| `t34-m2m3-advice` | 520K | 2026-10-20 01:32 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `docs/research/final_run_20261019/speed_advice/m2m3_ASTRA_DISSENT.md` |
| `t34-retention-prereg-20261020` | 500K | 2026-10-20 10:03 | Gap/post-release campaign task output (T1-T23) | cited by 5 tracked file(s) | `research/t34_pathb/REGISTRATION_RETENTION_V2.json` |
| `redesign-m3c-probes-20261008-recovery` | 496K | 2026-10-09 13:19 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 11 tracked file(s) | `docs/research/m3c_m3d_evidence/retry-recovery-closeout.json` |
| `apex-veto-v7-serving-20261002` | 372K | 2026-10-02 20:29 | Apex veto serving release check | cited by 12 tracked file(s) | `docs/experiments/apex_veto_v7_2026-10-02/README.md` |
| `t34-round0-fixes2-20261020` | 364K | 2026-10-20 16:39 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `research/t34_round0/README.md` |
| `apex-veto-v5-serving-20261002` | 352K | 2026-10-01 20:15 | Apex veto serving release check | cited by 2 tracked file(s) | `docs/experiments/apex_veto_v5_2026-10-01/README.md` |
| `apex-veto-serving-20261001` | 348K | 2026-10-01 04:46 | Experiment output | cited by 8 tracked file(s) | `docs/experiments/apex_veto_serving_and_v3_2026-10-01/README.md` |
| `final-run-panel-20261019` | 296K | 2026-10-19 17:56 | Advisor / panel / governance notes | cited in Claude memory notes |  |
| `redesign-t11-probes-20261017` | 280K | 2026-10-17 01:47 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 3 tracked file(s) | `research/t11_screen_20261017/README.md` |
| `redesign-m3b-probes-20261008` | 260K | 2026-10-08 16:39 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 5 tracked file(s) | `docs/research/t4a_counter_matching_2026-10-16.json` |
| `simd-v7v8-parity-20261003` | 236K | 2026-10-03 04:59 | SIMD sim parity check | cited by 3 tracked file(s) | `docs/research/runpod_x86_consistency_2026-10-03.md` |
| `t4f-addendum2-validation-20261016` | 232K | 2026-10-16 15:35 | Gap/post-release campaign task output (T1-T23) | cited by 4 tracked file(s) | `docs/research/specs/t4f_prereg_addendum2_2026-10-16.md` |
| `redesign-m4-mac-probes-20261013` | 180K | 2026-10-13 11:10 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 11 tracked file(s) | `docs/research/t4a_counter_matching_2026-10-16.json` |
| `colab-cli-robustness-20261018` | 172K | 2026-10-18 21:25 | Colab CLI qualification / acceptance / robustness work | cited by 1 tracked file(s) | `research/colab_cli/ROBUSTNESS.md` |
| `redesign-m4-p2-candidates` | 168K | 2026-10-13 05:44 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 4 tracked file(s) | `research/redesign_m4_20261013/PREREGISTRATION_M4.md` |
| `redesign-m6-mac-probes-20261014` | 128K | 2026-10-14 13:08 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 9 tracked file(s) | `docs/research/m6_result_2026-10-14.md` |
| `perf-quiet-mac-20261013` | 116K | 2026-10-13 22:56 | Performance benchmark / A-B probes | cited by 1 tracked file(s) | `research/perf_quiet_20261013/RESULTS.md` |
| `redesign-m3c-takeover-20261008` | 100K | 2026-10-09 13:21 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 1 tracked file(s) | `research/redesign_m3c_20261008/STATUS_A1_TAKEOVER_2026-10-08.md` |
| `redesign-m4a-probes-20261012` | 80K | 2026-10-13 00:30 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 7 tracked file(s) | `docs/research/t4a_counter_matching_2026-10-16.json` |
| `redesign-m3-p2-candidates-20261008` | 60K | 2026-10-08 13:40 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 2 tracked file(s) | `research/redesign_m3_20261008/phase_r.py` |
| `redesign-m3c-probes-20261008` | 56K | 2026-10-08 22:23 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 15 tracked file(s) | `docs/research/m3c_m3d_evidence/retry-recovery-closeout.json` |
| `t34-v2-teacher-amend2-followup-20261020` | 52K | 2026-10-20 02:12 | Gap/post-release campaign task output (T1-T23) | cited by 1 tracked file(s) | `research/t34_exit_b8/VERIFICATION_V2_AMEND2_FOLLOWUP.json` |
| `redesign-t4e-mac-20261016` | 48K | 2026-10-16 13:37 | Gap-campaign task (T2-T11) Mac runs / probes | cited by 14 tracked file(s) | `docs/research/specs/t4e_bugcheck_spec_2026-10-16.md` |
| `redesign-m5-probes-20261013` | 44K | 2026-10-13 21:59 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 6 tracked file(s) | `docs/research/panel_2026-10-15/round1_rl.md` |
| `perf-ab-probes-20261014` | 36K | 2026-10-14 01:20 | Performance benchmark / A-B probes | cited by 6 tracked file(s) | `docs/research/t4a_counter_matching_2026-10-16.json` |
| `t33-reviews` | 32K | 2026-10-19 15:02 | Gap/post-release campaign task output (T1-T23) | cited by 3 tracked file(s) | `docs/research/specs/2026-10-19_batched_teacher_round2.md` |
| `rules-fixes-20261019` | 16K | 2026-10-19 21:52 | Advisor / panel / governance notes | cited in Claude memory notes |  |
| `simd-v5-parity-20261002` | 16K | 2026-10-02 07:28 | SIMD sim parity check | cited by 2 tracked file(s) | `docs/research/simd_vector61_plan_2026-09-26.md` |
| `t23-analysis` | 16K | 2026-10-17 20:38 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `research/t23_r0_screen_20261017/analyze.py` |
| `t34-v2-specs` | 16K | 2026-10-19 19:42 | Gap/post-release campaign task output (T1-T23) | cited in Claude memory notes |  |
| `t4e-return-bank-20261016-attempt1-preflight-failed` | 4K | 2026-10-16 10:47 | Gap/post-release campaign task output (T1-T23) | cited by 2 tracked file(s) | `docs/research/t4e_collection_preflight_repair_2026-10-16.json` |
| `redesign-m4a2-probes-20261012` | 0K | 2026-10-12 22:15 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 7 tracked file(s) | `docs/research/t4a_counter_matching_2026-10-16.json` |
| `redesign-m6-probes-20261014` | 0K | 2026-10-14 06:34 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | cited by 5 tracked file(s) | `docs/research/t4a_counter_matching_2026-10-16.json` |

### UNKNOWN (23 folders, 5.1 GiB)

| Folder | Size | Last modified | What it is | Basis | Example citation |
|---|---:|---|---|---|---|
| `redesign-m6-gpu-20261014-a4` | 4.11G | 2026-10-14 06:35 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | no citation found |  |
| `worktree-archive-20261012` | 0.53G | 2026-10-12 19:04 | Archived worktree contents (m3d-untracked.tgz, g0_raster31v3_initial.pth, pqn-coverage-runs, rl-assessment-runs) | no citation found |  |
| `death-census-frp3-20261006` | 0.17G | 2026-10-06 04:37 | Death census analysis | no citation found |  |
| `growth-ceiling-20261003` | 0.06G | 2026-10-03 11:55 | Experiment output | no citation found |  |
| `lh1-calibration-20261004` | 0.06G | 2026-10-04 16:20 | LH-1 calibration / screen | no citation found |  |
| `apex-veto-v9-dev-sweep-20261003` | 0.05G | 2026-10-03 13:10 | Apex veto parameter sweep | no citation found |  |
| `train-smarter-20261007` | 0.04G | 2026-10-06 22:43 | Experiment output | no citation found |  |
| `redesign-m3c-design-20261008` | 0.03G | 2026-10-08 16:17 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | no citation found |  |
| `t3-diagnostics-20261016` | 0.01G | 2026-10-15 20:12 | Gap/post-release campaign task output (T1-T23) | no citation found |  |
| `t4f-heldout-outside-20261016` | 0.01G | 2026-10-16 14:00 | Gap/post-release campaign task output (T1-T23) | no citation found |  |
| `apex-veto-v9-screen-20261003` | 8520K | 2026-10-03 21:36 | Apex free-space veto variant screen | no citation found |  |
| `rules-audit-20261019` | 2872K | 2026-10-19 20:28 | Advisor / panel / governance notes | no citation found |  |
| `test-suite-hygiene-20261020-xeyheao_` | 2864K | 2026-10-20 19:03 | Temp dir from today's test-hygiene work (Codex running) | no citation found |  |
| `t34-astra` | 1476K | 2026-10-19 18:38 | Gap/post-release campaign task output (T1-T23) | no citation found |  |
| `t34-retention-oc-bare-unpaced-52dec74f` | 360K | 2026-10-20 18:59 | Gap/post-release campaign task output (T1-T23) | no citation found |  |
| `t34-retention-oc-bare-unpaced-f7e5d688` | 360K | 2026-10-20 18:52 | Gap/post-release campaign task output (T1-T23) | no citation found |  |
| `apex-veto-v9-softk-sweep-20261003` | 308K | 2026-10-03 17:32 | Apex veto parameter sweep | no citation found |  |
| `redesign-m3c-stage0-20261008` | 120K | 2026-10-08 19:44 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | no citation found |  |
| `redesign-t3-diagnostics-20261016` | 60K | 2026-10-15 19:15 | Gap-campaign task (T2-T11) Mac runs / probes | no citation found |  |
| `colab-cli-implementation-20261018` | 52K | 2026-10-18 18:45 | Colab CLI qualification / acceptance / robustness work | no citation found |  |
| `redesign-m4-selection` | 28K | 2026-10-13 11:10 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | no citation found |  |
| `colab-rate-calibration-20261018` | 12K | 2026-10-18 19:27 | Colab rate/limit calibration | no citation found |  |
| `t4f-prep-20261016` | 4K | 2026-10-16 13:53 | Gap/post-release campaign task output (T1-T23) | no citation found |  |

### SCRATCH (30 folders, 93.7 GiB)

| Folder | Size | Last modified | What it is | Basis | Example citation |
|---|---:|---|---|---|---|
| `m6-dev-20261014` | 4.08G | 2026-10-14 03:09 | M6 dev smokes (ctrl/deathw/tight) + `m6-gpu-plan/` holding a copy of the 4.06 GB demo bundle | uncited retry/plan/smoke/pilot staging |  |
| `t4-pod-validation-20261015-plan` | 4.06G | 2026-10-15 15:33 | T4 pod-validation plan staging; holds a copy of the 4.06 GB demo bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-nostock2` | 4.06G | 2026-10-13 00:08 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-try8` | 4.06G | 2026-10-13 04:28 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-try7` | 4.06G | 2026-10-13 03:10 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-try9` | 4.06G | 2026-10-13 05:43 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m6-gpu-20261014-small1` | 4.06G | 2026-10-14 03:40 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m6-gpu-20261014-preflight1` | 4.06G | 2026-10-14 04:23 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m6-gpu-20261014-rc1` | 4.06G | 2026-10-14 03:46 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m6-gpu-20261014-dup4` | 4.06G | 2026-10-14 06:34 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m6-gpu-20261014-plan` | 4.06G | 2026-10-14 03:24 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-small3` | 4.06G | 2026-10-13 00:19 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-small5` | 4.06G | 2026-10-13 01:14 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-ledger5` | 4.06G | 2026-10-13 01:14 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a2-gpu-20261012-refused1` | 4.06G | 2026-10-12 22:14 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a-gpu-20261012-ledger-refused` | 4.06G | 2026-10-12 20:50 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4a-gpu-20261012-plan` | 4.06G | 2026-10-12 20:34 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m3c-gpu-20261008-recovery-plan` | 4.06G | 2026-10-08 22:56 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m3c-gpu-20261008-a1-plan` | 4.06G | 2026-10-08 20:53 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m3c-gpu-20261008-plan` | 4.06G | 2026-10-08 19:50 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m3b-gpu-20261008-r2` | 4.06G | 2026-10-08 05:44 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m3b-gpu-20261008-r3` | 4.06G | 2026-10-08 06:23 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m3b-gpu-20261008-plan` | 4.06G | 2026-10-08 05:35 | RunPod GPU pod staging/attempt dir (plan.json, attemptNN logs) carrying a copy of the 4.06 GB demo-bundle `inputs/` | uncited retry/plan/smoke/pilot staging |  |
| `q1-a4-20261018-r1` | 0.08G | 2026-10-18 18:19 | Q0/Q1 Colab factorial (Drive-queue era) runs | uncited retry/plan/smoke/pilot staging |  |
| `t18-h100pod-secure-20261017-try1` | 0.08G | 2026-10-17 14:44 | Gap/post-release campaign task output (T1-T23) | uncited retry/plan/smoke/pilot staging |  |
| `t18-h100pod-secure2-20261017` | 0.08G | 2026-10-17 14:50 | Gap/post-release campaign task output (T1-T23) | uncited retry/plan/smoke/pilot staging |  |
| `t9-divergence-20261017-pilot-compiled` | 132K | 2026-10-16 22:04 | Gap/post-release campaign task output (T1-T23) | uncited retry/plan/smoke/pilot staging |  |
| `t9-divergence-20261017-pilot` | 132K | 2026-10-16 22:02 | Gap/post-release campaign task output (T1-T23) | uncited retry/plan/smoke/pilot staging |  |
| `runpod-stock-monitor-20261014` | 124K | 2026-10-14 12:13 | RunPod fan-out / cross-check / stock monitor | uncited retry/plan/smoke/pilot staging |  |
| `redesign-m4b-live-dev` | 16K | 2026-10-13 05:39 | Redesign milestone (M2-M6) Mac runs / probes / Phase R | uncited retry/plan/smoke/pilot staging |  |
