# Round 0 A5 admission erratum — 2026-10-20, before ratification or data

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. This sibling preserves the
hash-bound original preregistration bytes. It supersedes only the A5 row-step
floor in `PREREGISTRATION_ROUND0.md`, “Chunk sizing and spend cap,” following
binding `DESIGN_V2_SHADOW.md` Amendment 2 addendum B, committed on main as
`41270de6` before A5 run #2.

A5 v2 GO requires **at least 50 completed worlds/hour per L4 lane**, measured
over the registered **1,200-second steady window, excluding cold fill**.
Active row-steps/second remains reported and has no 3,000 threshold. The receipt
records these as `completed_worlds_per_hour_per_lane`, `steady_window_seconds`
and `cold_fill_excluded`, alongside `active_row_steps_per_second`.

Round 0 still separately requires measured **paired** worlds/hour for its exact
entrypoint recipe to size chunks. A teacher-only completion rate is not a paired
rate. Both arms retain native vector-opponent dispatch, including on CUDA; Round 0
explicitly disables the newer optional vector batching in the teacher so it
matches bare B8. Any paired-rate derivation must account for that frozen recipe,
bare execution and metric regeneration in its hash-bound evidence.

The fixed bank, statistical rules, spend calculation, two-lane rehearsal,
ratification requirement and dispatch hold are unchanged. The source inventory
binds this erratum, and the operator ratifies it with the original draft.
