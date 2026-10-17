# T7 B8 fresh-bank Phase R — 2026-10-16

**Decision: STRICT_PASS.** All deciding T7 clauses pass. Serving qualification and release remain a separate planner-approved step.

Candidate seed: **3**. Tier 2: **PASS**, scripted episode survival difference vs champion_a5 + v8 **-0.015625** (SE 0.022947; threshold >= -0.05; 960 paired worlds). Tier 2 does not decide Tier 1. A candidate identity on FAIL grants no release authorization.

Selected checkpoint SHA256: `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.

Reason(s): Every deciding clause passed.

Execution: Mac CPU, at most six single-thread evaluation workers, owned tmux session `t7-phase-r`, AC/open-lid/thermal guard. Elapsed start-to-decision wall time **24579.57 seconds (6.828 hours)**. One look after all **5,760 H10000 episodes** and **45 direct-H5000 controls**; H5000 estimates use exact nested prefixes. No interim outcome analysis, extension, RunPod, GPU evaluation, serving-default change, or interaction with unrelated sessions/processes occurred.

## Operational recovery

The first coordinator exited with `PermissionError: [Errno 1] Operation not permitted` when calling `os.killpg(..., SIGCONT)`, then encountered the same error during cleanup. Its remaining owned workers finished, producing 168 records. The exact macOS cause was not established. A reviewed external coordinator adapter then resumed the same plan/start: it signals only registered, unreaped `Popen` children directly and sends CONT only after it paused a worker. Its owned-child signal self-test passed. All frozen evaluation workers, source hashes, checkpoints, banks, record identities and decision rules were unchanged; completed records were validated and reused. Wall time includes the interruption. No extra bank, interim outcome look or scientific retry occurred.

Adapter SHA256: `9216e36c21e76682a9197a292d46dc3ed2d572b15ca005be2bf6e1ea332c2e04`. Recovery receipt SHA256: `b481eb37000a82ff8d27d5e1e9ce8374425784037686b4b9acddeabf2c9e3012`. The first attempt's log and exit status are preserved as `coordinator-attempt0.log` and `coordinator-attempt0.exit`; the resumed session ownership is recorded in `ownership-resume-1.json`.

Only after the full study, isolated audit and independent result review completed, the coordinator's permanent lifecycle repair was applied to the repository. It uses direct signals to owned children, suppresses redundant CONT, and signals all owned workers before waiting during cleanup. Seven added lifecycle/scheduler tests and the original 34 focused tests pass together (41 total); no gameplay was rerun. The scientific producer, bank, rule, record and audit semantics are unchanged. The run and isolated audit above remain bound to the original frozen commit, not this later repair. Applied validation SHA256: `34816c47b7ed3c135a1ed273351be9078dc4781c4dd099a55f4cbb49c0087683`.

## Clause table

All survival differences are fractions. SEs are descriptive estimates at the registered full N; deciding survival floors are point comparisons with the registered tolerance. Mass effects use the existing M4 equal-mix seed effects and HK pool; increasing scripted N does not double its efficacy weight.

| Clause | Estimate | SE | Threshold | Result |
| --- | ---: | ---: | --- | --- |
| Efficacy H5000 mass (LB90 +129.477718) | +138.666306 | +5.993054 | LB90 > 0 | PASS |
| c1_mass_point | +138.666306 | +5.993054 | >= 20.0 | PASS |
| c2_frozen_alive | +0.046215 | +0.013547 | >= -0.03 | PASS |
| c2_mixed_alive | +0.042840 | +0.012882 | >= -0.03 | PASS |
| c2_scripted_alive | -0.006909 | +0.012254 | >= -0.03 | PASS |
| c2_scripted_mass | +150.219146 | +10.857283 | >= 0.0 | PASS |
| c4_positive_seeds | 5/5 | — | >= 4 | PASS |
| c5_mi10 | +93.746685 | +13.104930 | >= 20.0 | PASS |
| c5_prefix_controls | 45/45 identical | — | 45/45 | PASS |
| c6_guard_mass_seed0 | +178.857550 | +12.338792 | > 0.0 | PASS |
| c7_feasibility | 83 | — | <= 600 | PASS |
| Tier 1 absolute frozen episode survival | +0.743750 | +0.019947 | >= 0.50 | PASS |
| Tier 1 absolute mixed episode survival | +0.785417 | +0.018758 | >= 0.50 | PASS |
| Tier 1 absolute scripted episode survival | +0.538542 | +0.016098 | >= 0.50 | PASS |
| Tier 1 scripted seed 0 alive-frame floor | -0.006235 | +0.027506 | >= -0.15 | PASS |
| Tier 1 scripted seed 1 alive-frame floor | -0.023532 | +0.026222 | >= -0.15 | PASS |
| Tier 1 scripted seed 2 alive-frame floor | +0.028696 | +0.028664 | >= -0.15 | PASS |
| Tier 1 scripted seed 3 alive-frame floor | -0.018318 | +0.028571 | >= -0.15 | PASS |
| Tier 1 scripted seed 4 alive-frame floor | -0.015155 | +0.026042 | >= -0.15 | PASS |

Mass futility G=50: HK UB90 **+147.854894**, condition UB90 < 50 = **False**. It is evaluated only at the sole full-N look and is nonbinding when every pass clause holds. Feasibility uses SD* 336.400606, MDE* 129.477718, n_fixed 81, inflation 1.02 and cap 600.

## Candidate selection

The rule was fixed before gameplay: eligible seeds pass their own scripted alive-frame floor; select the largest lower endpoint of the primary paired-mi5 two-sided 80% Welch–Satterthwaite interval, with ties to the lowest seed. Eligible seeds: [0, 1, 2, 3, 4].

| Seed | Equal-mix mass effect | 80% WS lower endpoint |
| --- | ---: | ---: |
| 3 | +149.650051 | +131.939914 |
| 2 | +139.885738 | +123.301986 |
| 1 | +138.209615 | +121.292634 |
| 4 | +134.897030 | +117.230321 |
| 0 | +131.008288 | +113.772612 |

## Survival sensitivity and Tier 2

Fresh banks, pooled over all five finals; both references were measured on exactly the candidate worlds. Episode survival is no death by H5000, including a snake that dies later before H10000.

| Reference | Mix | Alive-frame difference | SE | Episode difference | SE | Pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| champion | frozen | +0.061643 | 0.013849 | +0.139583 | 0.030626 | 480 |
| champion | mixed | +0.010372 | 0.012387 | +0.060417 | 0.027608 | 480 |
| champion | scripted | -0.017383 | 0.012323 | -0.015625 | 0.022947 | 960 |
| incumbent | frozen | +0.046215 | 0.013547 | +0.106250 | 0.029610 | 480 |
| incumbent | mixed | +0.042840 | 0.012882 | +0.110417 | 0.028377 | 480 |
| incumbent | scripted | -0.006909 | 0.012254 | -0.018750 | 0.022586 | 960 |

Required historical sensitivity, disclosed only and never re-scored as T7 gate evidence:

| Historical comparison | Alive-frame difference | Episode difference |
| --- | ---: | ---: |
| B8 seed-0 M4 look 2 vs champion | -0.156 | -0.188 |
| Census vs champion | -0.039 | -0.157 |
| M4 pooled vs served incumbent | -0.003 | -0.050 |

The M4 outcome remains **PARTIAL** and motivated the single survival-guard amendment. B8 was trained with v8 labels and uses v8 features as inputs, although no veto acts at evaluation time. The 192 scripted / 96 frozen / 96 mixed worlds per seed were chosen prospectively for power, citing T5b's 38.8% historical survival-clause failure-risk forecast. These historical estimates are not new gate evidence. The campaign's completion bar remains Tier 1 AND Tier 2; this task does not release serving.

## Freshness and registration history

The original pre-registration was committed at `8fe1ad39`, before gameplay. Freshness rejected round 74 (collision with the T3 label namespace) and round 75 (collision with an older FRP training-exclusion inventory). Both complete rejected allocations and sibling errata were preserved and pushed before gameplay; no outcome was generated on either. The final run uses namespace `redesign-t7-phase-r-b8/v1`, keys `round76-seed<S>`, 960 globally distinct world seeds. Its passing audit compares 1,183,588 registered/historical exclusion values and completes 391 repository/artifact scan partitions with no hits. The audit binds T3/T4 collection manifests, M3/M4 banks, census coverage and all historical FRP exclusion inventories. It cannot establish facts about unrecorded external experiments or opaque binaries absent their manifests.

## Verification and immutable evidence

- Run implementation commit: `7e1734357115d44b11038634a0ce1cb18afddfa3`; preregistration and both allocation errata bound in the plan.
- Rule SHA256: `9bb8944b12d8dc87bbea3afd8259a25a99bd2c0509e3ce30db9851a124a815cc`.
- Plan SHA256: `76731e5f25d6327e6e4e21cd084151333c6889b1e26936e3f00d065982275327`.
- Decision SHA256: `8fd97ecc564fdc31a1d77ed50b8112dcffac42d736db20b2b410b67c6ef139fb`.
- Isolated audit SHA256: `c631930c0af441e72e304a08cea401ed1a494c02c3e73071a8a459fb33dde07e`; exact 5805/5805 records, hashes/identity and recomputed decision pass.
- Independent result review SHA256: `135b1b8b5736619dbf1d57460d606c3a60a4ff4f59019c4ee51392dc5e2aecbe`. Publication copies are under `research/redesign_t7_phase_r_20261016/results/`; raw records remain at the artifact root and are individually hash-bound by the decision inventory.
- Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t7-phase-r-b8-20261016`. `decision.json` retains every record path/hash; `plan.json` binds 920 source/config files, candidate/reference/opponent checkpoints, world rosters, P2 receipts and profiles. `ownership.json`, `run.log`, `run.exit`, `audit.log`, `audit.exit` preserve execution evidence.
- Pre-gameplay validation: 34 focused tests pass, including every rule, candidate ranking/ties/filter, nonblocking Tier 2, one-look completeness, altered records, resume sidecars and unchanged M4 rule pins; Black/isort checks pass. Existing P2 receipts validate for all five finals (9/9 H5000 and 3/3 H10000 each).
- The final audit runs in a separate `python -I` process on the unchanged run commit. This Phase R result changes no serving defaults.

## Secondary length-matched hazards

Descriptive H5000 deaths per 1,000 valid decisions within the same pre-decision length bin. The >=800 rows are the requested long-length diagnostic. These are conditional bin rates, not causal effects or deciding clauses; exposure differences remain visible. Zero exposure is reported unavailable.

| Hero | Mix | Length bin | Decisions | Deaths | Deaths /1,000 decisions |
| --- | --- | --- | ---: | ---: | ---: |
| B8-noveto | frozen | 0-199 | 424996 | 5 | 0.011765 |
| B8-noveto | frozen | 200-399 | 354022 | 9 | 0.025422 |
| B8-noveto | frozen | 400-799 | 703825 | 40 | 0.056832 |
| B8-noveto | frozen | 800+ | 712061 | 69 | 0.096902 |
| B8-noveto | mixed | 0-199 | 441417 | 5 | 0.011327 |
| B8-noveto | mixed | 200-399 | 361709 | 12 | 0.033176 |
| B8-noveto | mixed | 400-799 | 720369 | 37 | 0.051363 |
| B8-noveto | mixed | 800+ | 692730 | 49 | 0.070735 |
| B8-noveto | scripted | 0-199 | 662999 | 20 | 0.030166 |
| B8-noveto | scripted | 200-399 | 501363 | 50 | 0.099728 |
| B8-noveto | scripted | 400-799 | 1042407 | 112 | 0.107444 |
| B8-noveto | scripted | 800+ | 1670701 | 261 | 0.156222 |
| champion | frozen | 0-199 | 550844 | 14 | 0.025416 |
| champion | frozen | 200-399 | 533423 | 38 | 0.071238 |
| champion | frozen | 400-799 | 920513 | 132 | 0.143398 |
| champion | frozen | 800+ | 42248 | 6 | 0.142019 |
| champion | mixed | 0-199 | 557416 | 8 | 0.014352 |
| champion | mixed | 200-399 | 539965 | 13 | 0.024076 |
| champion | mixed | 400-799 | 1026383 | 103 | 0.100352 |
| champion | mixed | 800+ | 67598 | 8 | 0.118347 |
| champion | scripted | 0-199 | 969369 | 44 | 0.045390 |
| champion | scripted | 200-399 | 933467 | 68 | 0.072847 |
| champion | scripted | 400-799 | 1769570 | 263 | 0.148624 |
| champion | scripted | 800+ | 288487 | 53 | 0.183717 |
| incumbent | frozen | 0-199 | 488855 | 10 | 0.020456 |
| incumbent | frozen | 200-399 | 447322 | 28 | 0.062595 |
| incumbent | frozen | 400-799 | 846302 | 98 | 0.115798 |
| incumbent | frozen | 800+ | 301559 | 38 | 0.126012 |
| incumbent | mixed | 0-199 | 499093 | 7 | 0.014025 |
| incumbent | mixed | 200-399 | 456623 | 23 | 0.050370 |
| incumbent | mixed | 400-799 | 857132 | 96 | 0.112001 |
| incumbent | mixed | 800+ | 300613 | 30 | 0.099796 |
| incumbent | scripted | 0-199 | 860116 | 35 | 0.040692 |
| incumbent | scripted | 200-399 | 755475 | 42 | 0.055594 |
| incumbent | scripted | 400-799 | 1448820 | 216 | 0.149087 |
| incumbent | scripted | 800+ | 846204 | 132 | 0.155991 |

## Serving boundary

Serving qualification and release are a separate planner-approved step. The pre-registered procedure reuses the FRP-v3 Mac protocol: 25 Watch × H5000, 25 Play up to H5000, two parity probes × H5000, identity/counter/default/refusal/rollback criteria S1–S6 and independent audit, adapted to the selected bare B8. Existing rollback environment variables remain intact. On T7 FAIL, frp3-s12 + v8 stays served with no second guard amendment.
