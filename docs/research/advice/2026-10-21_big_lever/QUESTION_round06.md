# Big-lever research rounds — Round 6 question (feasibility round 3: what binds between offline search and a served gain on this Mac?)

Judge: Fable (round-5 judge wrote this, 2026-10-21 ~03:45 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` … `../round-05/JUDGMENT.md` (rulings, shortlist, amended experiment list — each round's §(a) and §(c)
supersede the previous where they say so), then `../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over
everything below) and `../QUEUE.md`. Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of
`main`; start with `AGENTS.md`). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Rounds 1–5's
verified facts and negative record hold with the corrections in the five judgments; cite them, do not re-derive.

## 1. Goal (unchanged) and one new owner constraint

One big-lever push, grounded in research that works in slither.io-like environments, using Colab GPUs at scale (~1,100 units,
spend within a week, speed over thrift), producing a **served snake that beats bare B8 s3 by ≥ +40 pooled H5000 mass at
equal-or-better survival on the Mac strict gate**, then serving qualification and owner OK. Rounds 1–3 mapped the space; Round 4
fixed the ranking measurement (E16); Round 5 fixed the funding rules (4M → 32M futility look → 100M decision by declared-model
forecast, UNCALIBRATED ⇒ balanced endpoints; E17 continuation falsifier). Rounds 6–7 attack feasibility, parity/trust and
evaluation validity; Rounds 8–10 converge on a staged plan with kill rules.

**Owner decision 2026-10-21 00:30 PT (`../LIVE_FACTS.md`): serving stays on THIS Mac** — Apple M5 Pro, **6 performance + 12
efficiency cores** (`sysctl hw.perflevel0/1.physicalcpu` = 6/12), Apple GPU via MPS/Metal (torch 2.9.1, `mps.is_available()` True),
no CUDA. GPU-host serving is out. Serving-latency levers must run on the Mac (CPU processes and/or the Mac GPU); Colab GPUs are for
training, labels and evaluation only. The owner asked whether cutting per-decision search latency on the Mac is "the real unlock",
since served search is the only lever that has clearly beaten B8. This round answers that head-on, alongside the RL/representation
line (which continues under Round 5's rules and is not re-litigated here).

## 2. What Round 5 established (per-advisor record in `../round-05/JUDGMENT.md`)

- **No early observable carries a calibrated P(+40 at 100M)**; E16 admission is a funding screen that buys a 32M futility-only
  tranche; the 100M decision is taken at 32M by a declared-model forecast (plateau / continued / forgetting; UNCALIBRATED if models
  disagree by > .20 ⇒ balanced endpoints, control + pre-nominated treatment, 2 seeds each). E17 (4 → 12M continuation, forecasts
  frozen before reveal, persistence comparator) is the first prospective test of forecast skill. Critic preparation has a finite
  deadline (E19). SE at 96 worlds/mix is **11.8**, not 18 (`screen.py:88`); Holm's first threshold ≈ 19.4 (two arms) / 27.4 (five).
- **Early damage is visible, early promise never was**: T23 −92.9/−77.3 at 2M/4M with food per alive frame −12..−17 %
  (`Q0_MASS_AUDIT.md`); T4c decisive agreement 10 → 44 % while counter agreement sat at 99 %; T6's restart stratum had 16 L ≥ 800
  decisions in its 4M block; M3-C is an M2b fine-tune, not a B8 one. Round 0 (T34 premise test) printed **KILL / INTEGRITY_FAILURE
  by operator error with no outcome read** (`CAMPAIGN_LOG.md:98`) — unread, not negative.
- **Verified facts about served search (the owner's lever):**
  - The T3-dense ceiling (`t3-rollout-labels-20261015/dense/ceiling-result.json`): B8 + exact-sim search at **every 5th decision at
    L ≥ 600** (`research/t3_rollout_labels_20261015/trigger.py:20-27`), **6 actions × H300**, eps 0, B8 greedy continuation, no veto;
    **scripted-only, on the 32 reused round-52 census worlds for B8 s0 and s3**: 54/64 vs 38/64 survival, mass 700 vs 608 (+92; the
    s3 row +122.8), **75 of 301,589 decisions changed**, 36,996 batches, **0.757 batches per worker-second (1.32 s per batch)**,
    1,126 env-frames/worker-s, 3 workers, 4 h 38 m. **Nothing measures it on frozen or mixed worlds**; if those gains were zero the
    strict gate's equal-mix pooled value would be ≈ +31 (< 40).
  - Continuous serving of that mechanism at 24 fps needs **4.8 batches/s = 6.34× one worker**. The serving contract pins Path B to
    "one extra CPU thread, ≤ 1 outstanding request, never blocks a tick, deadline miss → bare action" (`SUCCESS_CONTRACT.md` §4;
    `web/backend/app.py:37` `torch.set_num_threads(1)`); the direction judge already proposed lifting the one-thread clause to a
    registered N (`docs/research/advice/2026-10-21_direction/FABLE.md:24-26`); it is a product rule, the owner's to change.
  - The Phase 0 batched teacher (`research/phase0_search_20261019/PROFILE.md`, `SPEED_REPORT.md`) already measured, fixture-identical:
    **3.35–4.24× per search on one Mac CPU thread and 4.12–6.02× with the B8 forward on MPS** (N8/H128, 48 rollouts = 6.1k hero
    steps; 1.0–1.7 s CPU, 0.7–1.2 s MPS per search); remaining CPU cost is B8 inference (0.66–0.96 s of 1.0–1.7 s), then sim/obs, then
    Vector61 carry for frozen opponents; six worker **processes** on six searches gave 2.79×. The "20× Mac CPU target" was not met.
  - B8 CPU forward batch 1 one thread = 0.29–0.41 ms, so an H300 verify needs ≥ 92–124 ms of sequential inference — more than one
    41 ms tick (`DESIGN_V2_SHADOW.md:15`). Sequential depth, not batch width, is the latency floor; a GPU shortens the per-step
    forward only if its launch latency is below the CPU's ~0.3 ms.
  - The Path B v2 smoke (`t34-pathb-smoke-20261021/RESULT.md`): doom → LAD search on one worker thread; 68 natural dooms; the first
    fully evaluated improving candidate arrived **inside the lead in 53/68** (median 6.5 s vs median lead 10.1 s), yet **0/34
    rescue-capable intervals and 7/75 delay-only changes were delivered** — 475 improving evaluations completed for frames already
    passed. Six in-process **threads** (N = 6) collapsed the paced tick to 5–16 fps with 21,077 overruns. Per-doom records are
    retained (`lad-*.json`).
  - Long-body worker time on the Mac (S0, contended): v8 opponent-veto hook 56.8 %, inference 13.5 %, digest 11.9 %, sim 3.4 %;
    S2 native v8 is merged (1.9× whole-workload). Watch6 opponents are FRP3 + v8 (dearer per step than the dense ceiling's scripted
    opponents); Watch6 update p95 bar 11.688 ms, zero overruns.
  - Because T3-dense changes 0.025 % of decisions and the world is deterministic from an exact shadow, its searches are
    **speculatively pipelineable** (search decisions t+5, t+10, … in parallel; an executed change invalidates the queue ~75 times
    per 300k decisions). On paper W = 2 worker processes with the batched engine cover 4.8 batches/s; W = 7 at the current engine
    does not fit six P-cores. None of this is measured on the serving path.
- **Shortlist (Round 5 §d):** 1 served exact-sim search on this Mac (40 / 25, conditional on the frozen/mixed ceiling); 2 representation
  (30 / 18); 3 long-horizon PPO backbone (25 / 10); 4 GPU engine (30 %, Colab only); 5 search-in-the-loop (15 / 10); 6 archive (6);
  7 opponent add-on (4); 8 world model / league (2).

Experiments Claude may have dispatched since Round 5 (results, if any, in `../LIVE_FACTS.md`): **L2** (dense ceiling on fresh
frozen/mixed worlds — the lever's kill test), **L0** (smoke replay at k× speed with LAD-first ordering), **L1** (serving-shape latency
probe, W worker processes, CPU vs MPS), E8, D0, E4, E5, E10, E6/E7, E1b, E3, E2, E12, E14, E16/E17. Treat any reported number as
authoritative over your priors.

## 3. The gap Round 5 opened

The owner's lever has a verified offline margin and three unverified bridges: (i) **mechanism** — the +92 is scripted-only on reused
worlds, and the strict gate equal-weights three mixes whose opponents differ (13 of 13 resolved opponent sealers are deterministic
scripted greedy_food; frozen opponents are nets wrapped in v8); (ii) **serving** — the same search must run at 4.8 batches/s from an
exact shadow on a machine with six performance cores while the tick keeps p95 ≤ 11.688 ms and zero overruns, under a contract that
currently allows one extra thread; (iii) **identity and evaluation** — a served planner is a Claim B artifact (visibly labelled,
rollback pins B8 s3), its gain must be measured *paced* on fresh paired banks with misses and fallbacks counted, and nobody has said
what the paced identity test looks like for a speculative pipeline (what is invalidated, what is counted as delivered, how the
Mac-only strict gate treats a controller whose decisions depend on wall-clock). The RL/representation line, meanwhile, can only ever
produce a *bare* net; if lever 1 is servable, the two lines compete for the same Mac core budget (E14's geometry encoder and the
planner) and the same calendar, and the bare net's role may become "frontier supplier for the planner" rather than the push itself.

## 4. The open question for Round 6

**Between T3-dense's offline, unpaced, scripted-only +92 and a served hero that passes the Mac strict gate and the Watch6 product
bars on this 6P + 12E-core Mac, what is the binding constraint — the mechanism's unmeasured frozen/mixed gain, per-batch engine
latency (CPU batching, MPS, sequential depth), worker-process parallelism and speculative pipelining against the tick's zero-overrun
budget, the choice of which decisions to search and when, or the paced identity/evaluation rule for a wall-clock-dependent
controller — and what is the cheapest ordered sequence of Mac and Colab measurements this week that settles it, so that Round 7 can
split the big-lever push between a Mac served-search systems package and the Colab RL/representation line (whose funding rules are
fixed) with a stated kill point for each?**

Answer from your discipline, not as a committee. Name the published systems or methods you rely on (paper/system + year), give
numbers in this game's units (batches/s, ms per step, P-cores, worlds, decisions, Colab units, Sol days), state what may differ
between the served engine and the Mac reference sim and what may not (chosen actions, scores, mechanics, opponents, H5000
accounting), give a kill rule for your claim, and say where Round 5's judgment, its lever-1 placement or its L0/L1/L2 specs are
wrong and why. Separate known results from conjecture. The Mac-systems persona must give measured-or-measurable per-step budgets
(sim, observation, v8, B8 forward on CPU vs MPS, process hand-off) and a process/QoS layout for six P-cores and twelve E-cores. The
search-serving persona must give the search schedule (periodic vs doom-triggered vs anytime, horizon ladders, speculative queue
depth, invalidation) and its delivered-rescue arithmetic against the smoke's 34 rescue-capable intervals. The forensics persona must
say, from the smoke's retained records, why 53/68 inside-the-lead first improvements produced zero rescues. The evaluation persona
must write the paced identity test and the family of claims for a Claim B artifact. The red-team persona must argue that served
search on this Mac is a mirage (scripted-only margin, Watch opponents, paced overruns, calendar) and name the measurement that
proves it this week. Changing the game's rules, the opponents, or the benchmark is not beating B8. Keep to the output format in
your task prompt.
