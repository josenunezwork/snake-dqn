# T13b fixed measurement protocol

Decision-rule preregistration: f36a20d30f5c53cff3d22978cceafab435666b4b,
verified published to main and t13-20261017 before running.
Authority: docs/research/specs/t13b_six_snake_spec_2026-10-17.md and
binding advice steps 1–3. Serving defaults unchanged; no training or qualification.

Reuse T13 real GameSession.step()/snapshot(), continuous Watch respawns, raw
per-snake counters, actual decision lengths, executed boosting, consumed food,
authoritative kills and cause events, post-step live mass with dead frames zero.
At six snakes, use mechanics_v2.yaml unmodified default: all_b8, all_frp3,
all_champion and mixed. At eight snakes, only all_b8 via constructor-local hook.
Legacy homogeneous FRP3/champion wrap hero only; mixed wraps only FRP3 opponents
and leaves B8 hero bare. Identity and wrapper assertions must pass in each episode.
Offline monitor telemetry is suppressed, as in T13.

25 fresh paired world seeds per cell, H5000, 125 total episodes. Audit explicitly
excludes T7 final/rejected allocations, both T12 banks, T13 worlds/smoke, historical
registered exclusions and visible JSON/JSONL evidence. A separate smoke seed is
excluded from measured worlds. Audit limitations remain those of T13.

Tables: hero and arena decisions, deaths/1k actual decisions, pooled decision
length median/p95/max, H5000 mass (arena total and per-snake), kills made/taken,
head-on/body/wall/self and unexpected deaths, executed boost percentage, food/1k.
Quantiles: nearest order statistic round((N-1)*q), ties-to-even, as in T13/T12.
All six pairwise roster differences at six snakes, hero and arena, for every T13
metric. Paired seed bootstrap: 10,000 draws, NumPy RNG seed 20261017, resample the
same 25 indices across rosters, recompute pooled rates/distributions each draw,
95% percentile intervals, no multiplicity correction. Eight-snake baseline is
supplemental; no pairing of initial worlds across different population counts.

Apply the exact rule in docs/research/t13b_six_snake_roster_2026-10-17.md to pooled
six-snake point estimates. CIs do not alter the pre-registered decision rule.
No roster/default change here, even if mixed becomes the planned default.

Execution: CPU only, four single-thread episode workers plus one coordinator
(five processes), leaving one process slot for focused checks. Exclusive-create
output, no retries, preserved logs/exit codes, source hashes and raw receipts.
Never touch tmux or processes not started by this task. After every commit:
fetch origin, merge origin/main, push HEAD:main, push t13-20261017.

Monitoring erratum preserves the release record. Death rates are independently
assessed per driven Watch roster: six baseline 0.424/trigger 0.848; twelve baseline
0.768/trigger 1.536; eight uses this run's all-B8 hero rate and twice that rate.
Existing latency qualification remains unchanged; this gameplay measurement does
not create an eight-snake latency baseline or authorize a serving rollout.
