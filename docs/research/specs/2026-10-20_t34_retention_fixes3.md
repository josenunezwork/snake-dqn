# T34 retention — FIX-FIRST #3 rulings (operator, pre-data, 2026-10-20 ~14:35 PT)

Final re-review on main 1c052a35: all earlier findings fixed; unpaced Colab path cannot leak; one blocker (the 2%
contention budget) plus one wording point. Rulings:
1. Contention budget = 2% of each phase's REGISTERED CAP (timing gate cap, pre-check 8 h, study 40 h), evaluated
   ONLY at phase end (not at every between_rows boundary). Exceeding it at phase end -> that phase INCONCLUSIVE.
2. HostGuard sampling interval 5 s (not 30 s); each contended sample adds 5 s.
3. Load-average excess (and process-pattern CPU) counts only when it persists > 60 s continuously; record the
   threshold and the host's measured idle baseline (it reads ~7.8 on 18 cores while Phase 0 runs).
4. Admission: before each phase starts, require the host quiet for a registered window (e.g. 120 s); if not quiet,
   REFUSE to start (exit non-zero, no decision.json / no terminal record), so the operator can retry later.
5. §4 wording: "any unconditional fault -> KILL" applies to the bare and paced arms; faults in the descriptive
   unpaced Colab arm are reported as missing/failed descriptive rows and never change the verdict. State this in
   PREREGISTRATION_RETENTION_V2.md (pre-data amendment, with this reason).
6. Tests: one contended sample early in a phase does not end it; budget check only at phase end; sustained-load
   rule; refusal path writes nothing terminal; existing 154+ retention tests stay green.
Rules: Mac only; no live services/tmux; <=2 CPU threads; no efficacy rows; commit/push often; merge to main; report SHA.
