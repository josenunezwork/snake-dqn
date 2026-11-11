# Other experiment series — catalog

Index of every `docs/experiments/` subfolder **not** covered by the `state_system_audit*`, `solo_food*`,
`task_aligned*`, `apex*` and `four_hour*` catalogs. One row per folder; the question and outcome are
paraphrased from each folder's own lead paragraph or status line (built 2026-10-20). Read the linked
README for gates, numbers and provenance. None of these studies promoted a model; where a README says
so, Apex remained the incumbent.

## PQN infrastructure and early portfolio (2026-09-06 .. 2026-09-13)

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-06 | Does the full-epoch PQN sampler beat the legacy sampler (5 training seeds x 12 worlds)? | No benefit shown; keep legacy sampler default, PQN stays experimental | [statistical_review.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit) |
| 2026-09-09 | Interface freeze for PQN per-environment autoreset (B packets) | Contract only (B0 interface freeze); no run | [contract.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit) |
| 2026-09-12 | Index of what the RL portfolio tried and what happened | Durable index (ledger + waves) | [README.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit) |
| 2026-09-12 | H1-A: can rotating one enemy's heading change the near-best hero action? | `INCONCLUSIVE_NOT_ADVANCED`; no holdout evidence of action relevance | [REPORT.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit) / [PROTOCOL.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit) |
| 2026-09-13 | Add a canonical Watch scripted opponent anchor for future training | Engineering integrated, opt-in only; no quality claim | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Do 12 existing checkpoints show late collapse on common fresh worlds? | `HETEROGENEOUS_RETENTION / HETEROGENEOUS_NET_CHANGE`; neither collapse nor reliable learning | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Is MPS faster than CPU for the PQN update on the M5 Pro? | MPS 2.36x / 2.53x faster on short workload; `BOUNDED_STABILITY_TRIAL_ONLY` | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Does randomizing initial heading help? | `INCONSISTENT_NOT_ADVANCED` (gain seed 1, loss seed 2) | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Does raising epsilon floor 0.02 -> 0.20 help? | `NOT_ADVANCED`; did not replicate across seeds/mixes | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Does lowering Adam LR 5e-4 -> 1e-4 improve retention? | `NOT_ADVANCED`; passed seed 1, failed 5/6 on seed 2 | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Does a corrected living-mass objective improve PQN? | `INCONSISTENT_STOPPED_NOT_PROMOTED`; gain reversed on fresh seed, futility stop | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Add valid-hero food-contact telemetry to PQN | Engineering metric integrated (`633cf6f`); no quality claim | [README.md](../research/history/rl-runs.md#four-hour-2026-09-13) |
| 2026-09-13 | Does food-seeking scripted exploration advice help? | `NOT_ADVANCED`; lost to control in random-safe mix both seeds | [README.md](../research/history/rl-runs.md#solo-ambient-early) |

## Solo (non-`solo_food`) diagnostics (2026-09-13 .. 2026-09-14)

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-13 | Solo diagnostic infrastructure | Integrated (`e083116`), audit 107 checks / 0 failures; no training | [README.md](../research/history/rl-runs.md#solo-ambient-early) |
| 2026-09-13 | Does rewarding ambient (not corpse) food change short solo training? | Audited descriptive screen (PASS, 75 checks); not a gate | [README.md](../research/history/rl-runs.md#solo-ambient-early) |
| 2026-09-13 | Does the ambient-food signal replicate on new seeds? | `INDEPENDENT_SOLO_AMBIENT_SIGNAL_NOT_ESTABLISHED` (1 of 2 seeds) | [README.md](../research/history/rl-runs.md#solo-ambient-early) |
| 2026-09-13 | Do solo recipes improve with 200k vs 50k transitions? | Audited descriptive screen; no advancement | [README.md](../research/history/rl-runs.md#solo-ambient-early) |
| 2026-09-13 | Does eps_end 0.02 -> 0.10 help solo learning? | Stage A failed (16/24 learning, 6/12 nonregression) | [README.md](../research/history/rl-runs.md#solo-ambient-early) |
| 2026-09-13 | Did the failed seed stop seeing food during training? | No; contacts continued; cause of drop unidentified | [README.md](../research/history/rl-runs.md#solo-ambient-early) |
| 2026-09-13 | Does native Q(lambda) preserve ambient-food credit? | Yes on all 3 trajectories; failed seed had very few contacts (descriptive) | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-13 | Do joint-supervised policies transfer food collection to native solo? | `transfer_not_reliable` | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-13 | Does broader native-food BC give reliable 64-frame food collection? | Joint 0/3; held-out fit failed every seed | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-13 | Does spreading teacher examples over 16 worlds help? | `INCOMPLETE`; held-out fit failed all 3 | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-13 | Do food-geometry-only inputs fix native food behavior? | Gameplay 3/3 pass, held-out fit fail -> joint FAIL | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-13 | Does 64-frame food behavior repeat on fresh worlds? | `FRESH_NATIVE64_BEHAVIOR_CONFIRMED` (behavior only) | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | Does food behavior hold for 128 frames? | `NATIVE128_BEHAVIOR_CONFIRMED` | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | Does food behavior hold for 256 frames? | `NATIVE256_BEHAVIOR_NOT_RELIABLE`; survival bottleneck | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | Were 256-frame deaths caused by advisory-safe actions? | No; all 100 deaths had an empty safe set | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | Is the free-space veto a better scripted teacher? | Yes as teacher: 96/96 survived vs 67; eligibility passed | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | (AG) Can learners imitate the spacious teacher? | `INCOMPLETE / INCONCLUSIVE`; food learned, survival/fit gates missed | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | (AH) Do AG policies miss spacious-teacher turns? | `ADVANCE_TARGETED_AUGMENTATION` | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | (AI) Does one learner-state aggregation round help? | Failed reliability gate (CI includes zero) | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | (AJ) Does a fixed learner-state refresh help? | Failed reliability gate | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | (AK) Are AJ errors on recoverable food decisions? | `ERRORS_ON_RECOVERABLE_FOOD_DECISIONS` | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |
| 2026-09-14 | (AL) Does a training-only food view help native solo256? | Failed frozen joint gate | [solo-native-imitation.md](../research/history/teaching-search.md#solo-native-imitation) |

## Controlled-encounter series (2026-09-21 .. 2026-09-22)

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-21 | Build a finite scripted encounter bank | Calibration complete; 576/576 candidates admitted | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | How do fixed action-priority policies score on the bank? | Static controls measured (no learning) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | How do 6 existing checkpoints handle the bank at two doses? | Inference-only probe complete; paired CIs reported | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Can the raster net learn four-step native-teacher trajectories? | Dataset rejected at admission (input aliases) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | What caused the two input aliases? | Data-admission diagnostic only | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does food-only labeling pass the original gates? | No seed passed at mark 256 | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Do food-label learners stay inside teacher-covered states? | Coverage measured; frequent off-dataset visits | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Can the teacher label train-only prefixes (aggregation)? | Expanded training dataset admitted | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Do aggregated labels give held-world benefit? | Train-world gain; no seed passed held packages | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Is finite target membership sufficient to reproduce teacher food? | Insufficient on observed cases | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Is there finite-horizon regret in H4-optimal witness games? | Yes (79 games); early-food tie is a target candidate | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Admit the earlier-food target | Admitted (teacher/data only) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does the early-food target give held-world benefit? | Train-world gain; every lineage failed held packages | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Collect a broader (48-world) teacher corpus | 7,184 rows admitted | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does broader teacher coverage help held worlds? | Held food up, survival unreliable; joint false | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Are broader-data models competent on added training worlds? | Yes (91-94% teacher food, 99-100% survival); not held evidence | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Expand to 256 worlds | Rejected during collection (exact-input alias) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does future RNG change the target at the alias witnesses? | Yes, causal counterexample in 4 selected states | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does hidden RNG change targets before fatal actions? | No change in 180 eligible cells (finite noncontrast) | [README.md](controlled_encounter_hidden_rng_2026-09-21/README.md) |
| 2026-09-21 | Is a common-path teacher target feasible? | All 6 feasibility gates passed | [README.md](controlled_encounter_common_path_2026-09-21/README.md) |
| 2026-09-21 | Qualify a receding teacher | `COMPLETE_AUDITED`; 9/9 gates passed (teacher only) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does the receding-teacher label intervention change labels? | `COMPLETE_DATA_REJECTED`; zero label change | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Qualify an empirical finite-bank teacher | All 8 gates passed; resolves known alias | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Build a global finite-bank teacher (256 worlds) | All 8 gates passed; 25,616 rows admitted | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does the global-teacher expansion help learners? | Positive paired held food; absolute gates failed | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does training on 256 worlds from update 1 help? | No reliable held benefit at 1,536 updates | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does L2 (wd 1e-4) help? | No; all packages false | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Why do final expanded-policy held games die? | First action outside viable H4 target in all 48 fatal games | [README.md](controlled_encounter_residual_policy_diagnostic_2026-09-21/README.md) |
| 2026-09-21 | Does one TD update immediately damage behavior? | No immediate damage shown (delta exactly zero) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does short native-TD adaptation retain the imitation parent's benefit? | No; H4 food falls in every lineage | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does native TD at LR 1e-4 help? | One stability signal; all packages false | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does a cap-16 collector help native TD? | `COMPLETE_AUDITED_NO_RELIABLE_BENEFIT` | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does continuing native TD to age 256 help? | Rejected; retention packages failed | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Where do short-cap parent and continuation first diverge? | Divergences localized; cause not established | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-21 | Does one divergent action decide H16 survival? | Single-action sufficiency in 44 selected cases | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Where does teacher-set agreement change during native TD? | Changes localized (saved-record census, audit passed) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Does a teacher NLL term pass gradient compatibility? | Rejected (all-parent compatibility gate failed) | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Does crossing-family teacher rehearsal help? | No seed met recovery + retention | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Does a sparse direction anchor help? | Failed 3-lineage recovery rule | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Did the anchor arm retain more parent directions? | Slightly more, but gameplay gates still false | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Does a full native cycle (1,024 updates) recover behavior? | No; unchanged-native-dose line ended | [README.md](../research/history/teaching-search.md#controlled-encounter) |
| 2026-09-22 | Does a kappa=0.1 ambient-food overlay preserve competence? | No; reward line ended | [README.md](../research/history/teaching-search.md#controlled-encounter) |

## Compute

| Date | Question | Outcome | Link |
| --- | --- | --- | --- |
| 2026-09-21 | Best PQN env/thread setting on the Mac? | 128 envs x 4 threads = 1,284 transitions/s (+38%); more threads alone no help | [README.md](../research/history/rl-runs.md#solo-ambient-early) |

## Loose top-level files (not folders; not cataloged here)

[pqn_followup_2026-09-12.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit),
[research_stopping_point_2026-09-21.md](../research/history/rl-runs.md#solo-ambient-early),
[rl_repair_execution_2026-09-08.md](../research/history/rl-runs.md#pqn-redesign-and-sept-audit),
[solo_learning_status_2026-09-20.md](../research/history/rl-runs.md#solo-ambient-early).
