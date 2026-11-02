# T34 retention pre-registration — FIX-FIRST rulings (operator, pre-data, 2026-10-20 ~12:45 PT)

Independent pre-ratification review returned FIX-FIRST (findings with file:line in the operator log; summary below).
Owner delegated these calls to Claude. Implement exactly; all rulings are pre-data.

1. Paced efficacy ticks >= 41.7 ms (analyze_v2.py ~L364): DIAGNOSTIC, not KILL. Each such tick already executes
   bare (fallback counted in efficacy). The zero-overrun bar applies to the registered timing probes only.
2. Near-zero INCONCLUSIVE exception (analyze_v2.py ~L404-409): applies ONLY when mass is the sole failing
   category; any other failing bar (survival guards, boost, floor, product, timing) decides normally.
3. Timing (analyze_v2.py ~L293-300): update p95 <= 11.688 ms is required at the 24 fps Watch6 probe (contract §4);
   at 12 fps require zero overruns and REPORT p95. Make the quiet-host timing probe (ON/OFF, 24 and 12 fps, the
   registered ~0.36 h) a GATE that runs FIRST, before the T3-bank pre-check; if it fails, stop with KILL-timing
   before any efficacy work. Update PREREGISTRATION_RETENTION_V2.md accordingly.
4. Quiet host enforced by the runner: re-check AC/lid/thermal every <= 60 s during all phases; compute `contended`
   from (a) 1-min load average above a registered threshold (state it, e.g. > 3.0 beyond the run's own workers),
   (b) any other process using > 50% CPU for > 60 s (list it), (c) known heavy jobs (Phase 0 runner, other
   research runners) present. Contended or power/thermal change -> pause; persistent > registered limit ->
   INCONCLUSIVE (never a silent pass). Remove hard-coded attestation fields.
5. The create-only decision.json written at the end of the single look is the ONLY verdict; re-analysis is
   descriptive. Runner refuses to start unless the ratified registration JSON (not only the markdown) is committed
   in HEAD and pushed.
6. Fix RETENTION_OC_V2 field `max_joint_regression_pass_rate` to the observed value (limit separately); unify load
   key names between profile and validator.
7. Power (development data only, never retention/pre-check outcomes): extend the OC with realistic Path B effect
   scenarios (e.g. scripted survival gain +0.10/+0.25 with mass +40/+60/+92, using T3 dense deltas as the effect
   shape) and report joint GO probability; add 16-session Watch6 product power from the T15/T13b Watch baselines
   (deaths/1k paired CI excludes 0 at a 20% and 40% reduction). Keep N=32/mix unless joint GO power at T3-like
   effects is < 0.5 — then report, do not change N (operator decides).
Rules: Mac only; no live services/tmux; <=2 CPU threads (Phase 0 running; OC sims bounded); no efficacy rows;
commit/push often; merge to main; rerun the 71 retention tests + new tests; README update; report commit SHA.
