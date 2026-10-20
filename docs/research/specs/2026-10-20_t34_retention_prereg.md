# T34 "Path B v2 paced retention" — pre-registration package (Sol spec, 2026-10-20 ~10:00 PT)

Binding: DESIGN_V2_SHADOW.md Amendment 2 §3, SUCCESS_CONTRACT.md, P7_STATISTICS.md,
speed_advice/m2m3_fable_on_astra.md §4 (GO/KILL/INCONCLUSIVE), research/t34_pathb/{PROTOCOL_V2,README_V2}.md
("Registration schema and evidence"). Prepare EVERYTHING the registered run needs, so it can start the moment
Phase 0 finishes and the pre-check passes. Draft only: no efficacy rows may be produced.

1. `research/t34_pathb/PREREGISTRATION_RETENTION_V2.md`, header DRAFT_PENDING_OPERATOR_RATIFICATION (the owner
   delegated this call to Claude; Claude ratifies after review). Fix: question, comparator B8 s3 (sha 3e649783…),
   arms (bare / paced 24 fps / unpaced descriptive), fresh disjoint namespaces per mix (32 paired H5000 worlds each)
   + 16 continuous-respawn mixed-Watch6 sessions, exact product opponents, one look, fixed N, Holm over the two
   active arms, P7 inference (paired world / session cluster), every contract bar with metric/denominator, GO/KILL/
   INCONCLUSIVE verbatim from m2m3_fable_on_astra.md §4, caps 32 h elapsed / 48 CPU-h / $0, quiet-host condition
   (Phase 0 finished; record load), pre-check dependency (>=50/64), what it does not authorize.
2. Seed/bank allocation with a collision audit against research/, docs/, scripts/ and all prior T-study banks
   (frozen allocator; reject the whole allocation on any hit). Commit SEED_AUDIT and the bank JSON.
3. Study-specific fixed-N survival-band operating-characteristic simulation (survival band v2 per its amendment;
   use development variance only — T3 dense deltas, B8 Watch baselines — never retention outcomes). Also report
   power for +40 mass at N=32/mix and the fixed-N vs sequential cost note P7 §5 requires.
4. The registration JSON (`schema: t34-pathb-v2-registration/v1`) with all {path,sha256} bindings except the
   trace receipt for the final source, which the operator binds at ratification; a script that validates it.
Rules: Mac only; no live services/tmux; <=2 CPU threads (OC sim bounded, resumable); do not run the pre-check or
any efficacy arm; commit/push often; merge to main; README section with the exact ratify-and-run commands.
