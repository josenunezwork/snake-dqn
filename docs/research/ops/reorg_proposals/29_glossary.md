# Reorg proposal 29: glossary extension

Scope 29 extended `docs/research/GLOSSARY.md` on 2026-10-20 with 25 rows. They sit in a new section, "Added 2026-10-20 (B8 release and T34 era)". The 23 original rows are unchanged.

New rows: B8 / B8 s3, FRP, v2/v5/v7 veto, free-space features, Watch/Watch6, scripted/frozen/mixed, T3 dense, T23, T34, T35, Claim A/B, Path B, shadow/bisection teacher, Phase 0, Round 0, Round 0-D, R1/R2/R3, ExIt/DAgger (T34), A5/A6, P7, OC, relative boost floor, FIX-FIRST/RATIFY, Sol/Astra/Fable and u (Colab units). Every row is sourced from docs/STATUS.md, docs/research/final_run_20261019/*, research package heads or the Claude memory files it cites.

Terms the scope listed that already had rows and were left unchanged: H5000, strict gate / Tier 1/2, survival band v2, LB90, vector61, v8 veto, DAgger.

## Proposals (not applied)
1. **Update the stale header.** It still reads "As recorded through 2026-10-08". Change it to 2026-10-20 and note that pre-T34 rows describe the vector61/redesign era.
2. **Expand "FRP".** None of the sources this scope read spells out the acronym. Someone with the FRP-v2 protocol (`frp-v2:research/frp_v2_20261004/protocol.md`) should add it.
3. **Disambiguate "A5".** It means both the old champion checkpoint and FINAL_PLAN check A5. Consider renaming the checks in future docs, for example "P-A5", or always writing "A5 champion" and "A5 benchmark".
4. **Merge the duplicates.** If a root `docs/GLOSSARY.md` lands from another scope (README proposal 24 links one), fold it into this file and leave a one-line redirect stub.
5. **Retire the memory citations.** Several rows (v5/v7, free-space, Sol/Astra) can only cite files in `~/.claude/.../memory/`. Durable repo docs for the veto history and the agent-role conventions would let those citations go.
