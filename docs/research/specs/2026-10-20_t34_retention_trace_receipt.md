# T34 retention — final-source trace receipt for ratification (Sol spec, 2026-10-20 ~15:00 PT)

The registration is APPROVED for ratification (independent reviewer REVIEW: RATIFY on ea1508b9; Fable approval judge
APPROVE). Ratification needs `--trace-receipt`: an actual final-source PASS trace receipt bound to the registration's
`source_commit` 97232d51c94cfa1cbe00839c45601dfa632508fc (the old receipt in
snake-dqn-artifacts/t34-pathb-v2/verification-20261020/ is stale).
1. Create a fresh detached worktree at 97232d51 (outside this worktree), run the registered trace-parity procedure
   (the same one that produced the old `trace-receipt.json`: reference vs optimized search traces incl. exact Q ties,
   deaths, stale cases + the bound focused tests; see research/t34_pathb/VERIFICATION_V2.md and README_V2.md) and write
   the receipt to /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-v2-operator-01/trace-receipt.json
   (create the directory; create-only file), binding exactly the registered import closure at 97232d51.
2. Run `registration_retention_v2 validate` and confirm the receipt is accepted by the ratify path in a DRY check
   (if a dry-run flag exists; otherwise report the exact ratify command). Do NOT ratify and do NOT run timing,
   pre-check or efficacy.
3. Report: receipt path + sha256, PASS/FAIL, test counts. If FAIL, stop and report the failing traces.
Rules: Mac only; <=2 CPU threads; no live services/tmux; do not modify 97232d51 sources; commit nothing except
this spec note if needed.
