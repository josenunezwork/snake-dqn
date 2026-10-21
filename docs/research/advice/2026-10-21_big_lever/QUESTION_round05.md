# Big-lever research rounds — Round 5 question (feasibility round 2: what makes a small executable signal predictive of 100M?)

Judge: Fable (round-4 judge wrote this, 2026-10-21 ~02:45 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` … `../round-04/JUDGMENT.md` (rulings, shortlist, amended experiment list — each round's §(a) and §(c)
supersede the previous where they say so), then `../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over
everything below) and `../QUEUE.md`. Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of
`main`; start with `AGENTS.md`). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Rounds 1–4's
verified facts and negative record hold with the corrections in the four judgments; cite them, do not re-derive.

## 1. Goal (unchanged)

One big-lever push, grounded in research that works in slither.io-like environments, using GPUs/CUDA on Colab at scale (~1,100
units, spend within a week, speed over thrift), producing a **served snake that beats bare B8 s3 by ≥ +40 pooled H5000 mass at
equal-or-better survival on the Mac strict gate**, then serving qualification and owner OK. Rounds 1–3 mapped the space; Round 4
fixed the pre-scale measurement; Rounds 5–7 attack cost, predictive validity, parity/trust and evaluation validity; Rounds 8–10
converge on a staged plan with kill rules. The T34 campaign (defensive search teacher; Round 0 verdict due Wed) continues in parallel
and is not yours.

## 2. What Round 4 established (per-advisor record in `../round-04/JUDGMENT.md`)

- **One ranking measurement, registered as E16.** Mechanisms are ranked only by natural-start, candidate-owns-the-whole-life,
  paired H5000 mass on fresh worlds: `G` (arm − B8) and `J` (arm − common parent) with simultaneous world-clustered bounds, explained
  by the three-term ledger, *productive rescues* (B8 dies, candidate survives H5000, tail ≥ 90 % of B8's last length), newly caused
  deaths, and collateral (negative common-alive + B8-only mass over all natural worlds). E6/E7/E10/E3/E12 are prerequisites and
  diagnostics; none can sit on the leaderboard. E16 arms grow from E2's critic-qualified F endpoint with 2M critic preparation +
  4M actor transitions each: A control, G geometry residual (after E3 GO + E14 pass on the *E3* architecture), R archive
  (only under UB90(U) ≥ 40 + TRANSFER_ADMISSIBLE + τ ≤ 1), S one-deviation current-policy targets (only after E1 CONTINUE),
  P whole-world PLR replay (optional). ≤ 2 treatments per 20-u dispatch at E4 ≥ 2k/s; unavailable arms UNRANKED.
- **100M admission (frozen in Round 4 §c1):** Holm-adjusted LB90(J) > 0, point G ≥ +10, productive rescues ≥ 5/100 point in each
  mix with retention ≥ 90 %, collateral UB90 ≤ 10, survival guard A, E12-qualified critic. UB90(G) < 40 kills the artifact at the
  bar, not the mechanism. "+40 at the pilot" was ruled unfalsifiable (SE ≈ 18 at 96 worlds/mix).
- **Arithmetic:** +40 needs `(40+10)/306 = 16.3` added full-horizon survivors per 100 natural worlds (symmetric planning scenario;
  the gate is pooled mass with per-mix survival guards); at 15 rescues the collateral budget is 5.9, i.e. ≈ .004 endpoint mass per
  PPO batch update over 1,526 updates at 100M. At q = .4 this is 41 % of eligible deaths rescued and executed, or 58 % of eligible
  worlds needing a productive prefix at 70 % transfer. E7's 20 % is an investigation threshold (≈ +14.5 even if every confirmation
  were a full rescue). 1 % exploration finds a specific two-step correction ≈ 3.2 times per 8k eligible lives (100M).
- **Withdrawn from Round 3:** `q × EPD` (the registered EPD is the balanced early/ordinary average; use `q × mean(early)`);
  "a named non-B8-local change escapes FINITE_MENU_INSUFFICIENT" (a name supplies no frontier); E7 confirmation admits delay-only
  prefixes, so E10 counts productive confirmations separately.
- **Not levers:** opponent modelling (all 13 resolved opponent sealers are deterministic scripted greedy_food; perfect opponent-only
  rescue ≤ 10.6/100 at q = .4), curricula (≤ 1.3× eligibility enrichment at q = .4; T6 raised exposure 1.65× with zero survival gain).
- **Build facts:** E8 was verified unimplemented at HEAD (critic has no horizon input) and is now the first queue item; E6/E7 bank
  allocation was rejected six times (seed collisions) and E10 is still pre-data; the E3 student (7 planes + joint GRU(43,64)) and the
  E14 stub (8 planes + five GRUs) were different encoders — E14 now probes the E3 architecture; the GPU spike's rollout continues B8
  and ranks survival first — the search-target operator is a rebuild.
- **Shortlist (Round 4 §d):** 1 representation (35 / 20); 2 long-horizon PPO backbone (30 / 12); 3 engine (30 %); 4 search-in-the-loop
  (20 / 12); 5 restart archive + PLR (8); 6 opponent-conditioned add-on (5); 7 world model (3); 8 league (3); 1b retention sub-arm.

Experiments Claude may have dispatched since Round 4 (results, if any, in `../LIVE_FACTS.md`): E8 commit, D0 rerun, E4, E5, E10
registration, E6 + E7, E1, E3, E2, E12, E14, E16. Treat any reported number as authoritative over your priors.

## 3. The gap Round 4 opened and nobody has closed

E16 will produce, per arm, a 4M-transition incremental signal `J` with SE ≈ 18 and a productive-rescue count with a handful of events,
and the frozen rule will admit an arm to 2 × 100M on LB90(J) > 0 plus ≥ 5 rescues/100. **No learning curve on record justifies that
extrapolation.** Every B8-warm-start fine-tune (T23 −80.8, S1 −12.5/−14.9, T4c −9.9 vs CTRL, T6 −40.1 vs CTRL, M3-C −250 no-veto)
ended negative; nobody has reported what their *first* blocks looked like, whether any early block was positive, or whether the
endpoint was predictable from within-run observables (drift by length decile, common-alive growth slope at L ≥ 800, rescue/re-death
events, critic EV in late-body cells, occupancy of eligible states). The 32M futility-only look exists but its content is "tripwires,
survival band, UB90(G) ≤ 0, negative candidate-owned early dividend" — a stop rule, not a predictor. Replication costs 60–120 u and
two to three calendar days of the week; a wrong admission burns the week's one shot at 100M; a wrong KILL at 4M discards the only
mechanism that needed more samples. Published RL scaling and learning-curve work (sample-efficiency curves, performance-vs-compute
fits, early-stopping and predictive-probability-of-success methods, group-sequential designs) was built for monotone objectives from
scratch — not for a near-deterministic incumbent whose gains come from rare 300-decision route corrections paid for in continuous
growth loss. Mechanisms differ in what their early signal should look like: geometry should show immediate ordinary-state regret
reduction before any rescue; search targets should show rescues on target roots before natural-start transfer; archives should show
restart-stratum success before natural-stratum success; pure PPO should show nothing until a sequence is found.

## 4. The open question for Round 5

**What within-run and endpoint observables of a 4M-transition E16 block (and of the 32M futility look) carry genuine predictive
information about whether a 100M endpoint of that arm will reach +40 at equal survival — mechanism by mechanism — and what is the
cheapest way (≤ 20 u, or 0 u from existing E2/E16 logs and the historical T23/S1/T4c/T6/M3-C runs) to calibrate those predictors
this week, so that the 2 × 100M admission is a measured forecast with a stated probability rather than a threshold crossed by
noise?**

Answer from your discipline, not as a committee. Name the published systems or methods you rely on (paper/system + year), give
numbers in this game's units (worlds, decisions, hero transitions, events, L4 hours, Colab units, Sol days), state what may differ
from the Mac reference sim in training and what may not, give a kill rule for your claim, and say where Round 4's judgment, its E16
admission rule or its shortlist is wrong and why. Separate known results from conjecture. The negative-results-forensics persona
should say what the early blocks of the losing fine-tunes actually looked like (from the repo's logs and result files, cite file:line)
and whether any observable there would have predicted the endpoint. The sequential-design persona should propose the exact form of
the 4M and 32M looks (predictive probability of success, group-sequential spending, Bayesian or frequentist) with operating
characteristics at true J ∈ {0, +10, +20, +40}. The red-team persona should argue that 4M → 100M extrapolation is a lottery for
every mechanism and name the measurement that would prove it this week. Changing the game's rules or the benchmark is not beating
B8. Keep to the output format in your task prompt.
