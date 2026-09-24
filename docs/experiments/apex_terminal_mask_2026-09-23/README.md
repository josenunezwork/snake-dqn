# Saved Apex terminal-mask census

This retrospective audit covers all 576 saved games from the six H64/H256 endpoint files on the adaptive H256 bank and classifies all 15 terminal death frames. It did not run a game, load a checkpoint, call a model, or update a learner.

The predeclared result branch was **`ALL_TERMINAL_DEATHS_AFTER_EMPTY_SAFE_MASK`**. All 15 deaths followed an empty raw and legal-safe action set, after which the policy used its normal-only fallback. Every recorded death cause was self. There were no deaths after an action marked safe and no selected-outside-safe discrepancies.

| Seed | Arm | Games | Frames | Survivors | Deaths | Empty safe set | Selected safe | Invalid alternative |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2026101001 | H64 | 96 | 24,406 | 94 | 2 | 2 | 0 | 0 |
| 2026101001 | H256 | 96 | 24,445 | 92 | 4 | 4 | 0 | 0 |
| 2026101002 | H64 | 96 | 24,450 | 95 | 1 | 1 | 0 | 0 |
| 2026101002 | H256 | 96 | 24,413 | 93 | 3 | 3 | 0 | 0 |
| 2026101003 | H64 | 96 | 24,500 | 94 | 2 | 2 | 0 | 0 |
| 2026101003 | H256 | 96 | 24,395 | 93 | 3 | 3 | 0 | 0 |
| **Total** |  | **576** | **146,609** | **561** | **15** | **15** | **0** | **0** |

The paired survival counts are descriptive within each seed's 96 shared worlds, not independent seed replications: seed 2026101001 had 90 both survive, 4 H64 only, 2 H256 only, and 0 both die; seed 2026101002 had 92, 3, 1, and 0; seed 2026101003 had 91, 3, 2, and 0, respectively.

This is a terminal-state taxonomy. It does not establish what caused a snake to enter an empty-safe-mask state, whether an earlier choice could have avoided it, or whether another action would have survived. No learned-survival improvement or promotion follows. The earlier continuation's fresh-bank **`INVALID_STOP`** and retention failures remain unchanged.

No learning curve applies to this zero-update diagnostic. The prior continuation's figures are reused for context: [learning curves](../apex_horizon_continuation_2026-09-23/learning-curves.png), [food accrual](../apex_horizon_continuation_2026-09-23/food-accrual.png), and [representative paths](../apex_horizon_continuation_2026-09-23/representative-paths.png). The copied terminal examples and full census are in [representatives.json](representatives.json) and [table.md](table.md).

The analysis and independent audit receipts both passed naturally. For those two guarded jobs only, combined guarded time was 5.457375209080055 seconds, peak child RSS was 1,364,672,512 bytes, and minimum available memory was 36,504,567,808 bytes. Each prospective stage parsed the six bound raw files once. These figures do not account for prior preparation access.

Before the new freeze, a broad source search also scanned the continuation subtree, including raw evaluation files, and printed raw excerpts. Its output was truncated; the tool reported 412,011,556 omitted bytes. The exact lifetime physical reads and elapsed time are unknown. The fixed design predated this access, and no criteria adaptation is attested. The incident and its independent review are preserved in the frozen record; preparation cannot be described as having had zero prior access.

The independent audit report SHA-256 is `59deb8370e1c6777eab46494382315b0ad16887b5dc595c817abccf63db9d879`; the analysis report SHA-256 is `ed575bc8f6c587a334b2c4c4d09e87d709ee6bc6b7393f204a70d7f5953135ba`. The copied table SHA-256 is `200b4a8cec2a2839b0069530e1e8d656f1fdbf71cb387e2781ea92096ec2ffa1`, and the copied representatives SHA-256 is `5629a1518592443f6f97642a80eccaae0ce1fa34f941e79c6e62431d4a6e309a`. The complete [closeout record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-saved-terminal-mask-diagnostic/closeout.json), including both receipts, has SHA-256 `bbd8c9d8d8f214f13f1c7963830671626d9199d764bb776caccf9acc4c459852`.
