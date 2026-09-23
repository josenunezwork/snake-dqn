# Saved Apex pretrap-choice diagnostic

This retrospective audit examines 15 fixed deaths from six saved endpoint cells (96 games per cell). The predeclared branch was `RECORDED_SPACE_GAP_IN_ALL_THREE_SEEDS`.

| Seed | Arm | Fixed deaths | Comparable normal predecessors | Lower recorded-space choices |
|---:|---|---:|---:|---:|
| 2026101001 | h64 | 2 | 2 | 2 |
| 2026101001 | h256 | 4 | 3 | 3 |
| 2026101002 | h64 | 1 | 1 | 1 |
| 2026101002 | h256 | 3 | 3 | 3 |
| 2026101003 | h64 | 2 | 2 | 2 |
| 2026101003 | h256 | 3 | 3 | 3 |

Fourteen of the 15 predecessors selected a normal action whose recorded free-space feature was `0.03125`, while another safe normal action scored `1.0`. The remaining case had no other safe normal action. In every case, the first empty raw safe-action mask occurred on the terminal frame.

This is a selected, retrospective association in the recorded free-space feature. That feature is not a survival guarantee, and the saved data cannot show whether choosing an alternative would have survived or identify a causal model or optimization defect. There were no new games, model queries, checkpoint loads, or optimizer updates. This result establishes no learned survival gain, model improvement, winner, or promotion. It does not repair the prior fresh-bank continuation `INVALID_STOP`.

The two guarded analysis and audit jobs used 5.6309175 seconds total, with peak child RSS of 1.27 GiB and minimum available RAM of 33.94 GiB. These resource figures cover only those two guarded jobs, not lifetime preparation or research activity.

The earlier terminal-mask study's preparation-access incident remains disclosed: a broad search accessed raw files, and lifetime physical raw/prefix reads are unknown. That earlier study's design predated the access. This new diagnostic was designed after the completed terminal findings; its implementation used named sources and completed metadata before the two admitted raw passes. No analyst-blindness claim is made.

Prior horizon figures are reused; this diagnostic generated no new curve: [learning curves](../apex_horizon_continuation_2026-09-23/learning-curves.png), [food accrual](../apex_horizon_continuation_2026-09-23/food-accrual.png), and [representative paths](../apex_horizon_continuation_2026-09-23/representative-paths.png).

See the [per-target table](table.md) and [representative records](representatives.json). The closeout records the full [audited run](../../../../snake-dqn-artifacts/ongoing-research-20260913/apex-saved-pretrap-choice-diagnostic/closeout.json).
