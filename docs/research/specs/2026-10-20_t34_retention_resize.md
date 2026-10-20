# T34 retention — pre-data resize after the power review (operator ruling, 2026-10-20 ~13:30 PT)

RETENTION_OC_V2.json (main) `t3_like_power_review_required`: joint GO probability at T3-like effects (+92 mass,
+0.25 scripted survival) is 0.27-0.32 at a 20% Watch death reduction and ~0.55 at 40%; product power with 16
sessions is 0.46-0.55 at 20%. No retention or pre-check data exist. Owner delegated the call. Ruling:
1. Terminal worlds 32 -> 48 per mix (144 paired); product sessions 16 -> 32 (continuous-respawn mixed Watch6).
2. Caps 40 h elapsed / 60 CPU-h, $0 (minimum paced exposure ~ (144+32) rows x 2 paced arms x 5000/24 s ~ 20.4 h,
   plus timing gate, pre-check, setup and the unpaced descriptive arm). Stop at the first cap -> INCONCLUSIVE.
3. Allocate the additional seeds with the frozen allocator + collision audit (zero hits, all prior banks incl. the
   existing 113 retention seeds and the Round 0 draft bank if present); keep the existing seeds.
4. Re-run the development-only OC at the new N (protected false-pass <= 0.06; report joint GO power at the same
   scenarios and the product power at 32 sessions). If joint GO at T3-like effects with 20% reduction is still
   < 0.5, REPORT it (do not change N again).
5. Update PREREGISTRATION_RETENTION_V2.md (record this resize and its reason as a pre-data amendment), the
   profile, bank, registration JSON (draft), README; validator passes in draft mode.
Rules: Mac only; no live services/tmux; <=2 CPU threads (Phase 0 running); no efficacy rows; commit/push often;
merge to main; rerun the 103 retention tests; report commit SHA and the new power numbers.
