# T35 two-deviation rescue probe — EXTEND + pooled verdict (2026-10-20 PT)

**Pooled verdict (APPROVAL_EXTEND C3, fixed pre-data): KILL — W_pool = 2/7 worlds.**
The registered beam heuristic found ≤ 2 rescued worlds → do not fund GPU search now. This is a statement about the
registered beam, not a proof that no two-deviation rescue exists (one known path exists, see below).

**Reporting conditions (C4):**
- W_novel = 0: the only rescued worlds are b8_s3_19 and b8_s3_9, the two worlds with an A1.7 known T3 multi-deviation path.
- W_uncensored = 1 (b8_s3_9). b8_s3_9@4188's rescue (t1 4198 a0, t2 4204 a3) **is** the A1.7 known path; reach 600.
  b8_s3_19@4693's rescue (t1 4813 a3, t2 4965 a0) is episode-end censored: reach 35 steps (frame 5000 cap).
- EXTEND: 26/26 events NOT_RESCUED, 0 incomplete/unknown (U_pool = 0). Primary: 2 RESCUED_DEPTH2, 5 NOT_RESCUED.

**Run:** EXTEND launched 19:18 PT from the frozen worktree `snake-dqn-t35-2dev-frozen` at 321ba7c7 (tmux t35-extend,
3 procs, nice 10, cap 7200 s), finished 19:37 PT (EXIT=0), 0 Colab units. Provenance (C2): all 26 EXTEND receipts and
all 7 primary receipts commit 321ba7c7, dirty "", plan root = frozen worktree, cap 7200. Thermal log (C5): AC
attached, lid open, no thermal/performance warnings at every 5-min sample. No retries or replacements (C6).

**Approvals:** primary `snake-dqn-artifacts/fable-panel-2dev/APPROVAL.md` (Fable); EXTEND
`.../APPROVAL_EXTEND.md` (Opus stand-in; Fable usage limit). Outputs: `snake-dqn-artifacts/t35-two-deviation-20261020/{primary,extend}/`.

| world | events | rescued | note |
|---|---:|---|---|
| b8_s0_21 | 5 | no | |
| b8_s0_8 | 3 | no | |
| b8_s3_19 | 1 | yes (censored, reach 35) | new path, episode-end |
| b8_s3_26 | 3 | no | |
| b8_s3_6 | 6 | no | |
| b8_s3_8 | 8 | no | |
| b8_s3_9 | 6 | yes (reach 600) | = A1.7 known T3 path |
