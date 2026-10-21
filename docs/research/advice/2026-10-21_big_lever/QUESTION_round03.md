# Big-lever research rounds — Round 3 question (last mapping round: from a rare early correction to +40 at the gate)

Judge: Fable (round-2 judge wrote this, 2026-10-21 ~01:00 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` and `../round-02/JUDGMENT.md` (rulings, shortlist, the amended experiment list), then
`../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over everything below). Repo:
`/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`; start with `AGENTS.md`).
Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Round 1's `QUESTION.md` §2 and
Round 2's `QUESTION.md` §2–3 (verified facts, negative record) still hold with the corrections in the two judgments.
Do not re-derive them; cite them.

## 1. Goal (unchanged)

One big-lever push, grounded in research that works in slither.io-like environments, using GPUs/CUDA on Colab at
scale (~1,140 units, spend within a week, speed over thrift), producing a **served snake that beats bare B8 s3 by
≥ +40 pooled H5000 mass at equal-or-better survival on the Mac strict gate**, then serving qualification and owner
OK. Rounds 1–3 map the space; 4–7 attack feasibility/cost/parity/eval validity; 8–10 converge on a staged plan with
kill rules. The T34 campaign (defensive search teacher; Round 0 verdict due Wed) continues in parallel and is not yours.

## 2. What Round 2 established (per-advisor record in `../round-02/JUDGMENT.md`)

- **The mechanism is specified and near-unanimous.** Exact finite-horizon H5000 alive-mass return (γ=1; zero
  bootstrap at death and at frame 5000; bootstrap at rollout cuts), a separately refitted critic that receives the
  remaining horizon (B8's 60 scalars carry no episode clock — `src/simd_env/ego_raster.py:62`, `ego_raster_b.py:14-30`),
  PopArt-style value normalisation, calibrated B8 warm start, cumulative drift from B8 *measured* on ordinary states
  (not anchored by a new loss in the pilot), fixed opponents, natural resets, exact-sim counterfactuals used first as
  measurement. γ=.998 was not blind to the pocket (0.819 weight at 100 decisions) but undervalued the post-3,200 tail
  (23.3 %) and relied on five-step bootstrapping under a v8 anchor. Round 1's "no new return machinery" was wrong.
- **E2 as frozen had two gaps** (no critic horizon input; rollout-cut rule unspecified); they are pre-data amendments
  (Round 2 §f, E8), E2 is NOT_DISPATCHED.
- **The mechanism observable is the early-prefix escape dividend (E6):** roots ~100 decisions before B8 enclosure
  deaths plus matched ordinary roots; candidate steers 64 decisions, B8 resumes to H5000; paired remaining mass vs
  uninterrupted B8, world-clustered. UB90 ≤ 0 on the early stratum with a critic that passed its held-out EV check =
  wrong mechanism; LB90 > 0 with ordinary stratum ≥ −2 = needs scale. Six advisors converged on it independently.
- **Exploration is the under-specified half.** 1 % non-greedy ≈ 50 departures per surviving life; a specific two-step
  alternative at specified decisions has probability ~4e-6. Whether an earlier reachable frontier exists for
  B8-local stochastic exploration is being tested (E7: 128 late-death worlds, restore 100/150/200 decisions before
  death, 8 stochastic 64-decision prefixes, confirm under independent RNG; threshold ≥ 20 % of worlds with a
  confirmed ≥ +20 prefix). T6 refuted fixed fatal-state TD restarts, not successful-prefix backward curricula.
- **Throughput reality.** CPU collection ≈ 1–4k useful hero transitions/s per 12-vCPU (6-core) runtime; 100M per seed
  is a week object, 10^9 is not without the device engine. E1 is rebriefed to three kernel contracts (tiled union-find
  for uncapped region size; per-world sparse-frontier BFS for the count-capped v2/v5/v7 searches; reach-time untouched)
  with a realistic benchmark (1,024–4,096 worlds, L 800–1,500, five real opponents). A100/H100: no, same host.
- **A +10 E2 GO buys replication (2 × 100M, endpoint-only), never 10^9.** At illustrative SE ≈ 18, a null arm clears
  +10 ~29 % of the time; GO now requires LB90 > 0 with Holm across arms, and an operating-characteristics simulation
  (E9) is due before data.
- **Corrections.** T4c −9.9 / T6 −40.1 are vs paired CTRL, not vs B8 s3. Claim A D0 is a mechanical failure
  awaiting rerun, not a registered KILL. E2's boost tripwire (3 × 0.12 % = 0.36 %) sits below the contract's 0.5 %
  floor (`SUCCESS_CONTRACT.md:32`). Search-in-the-loop (06's refreshed current-policy Q̂ targets) is the right form for
  lever 4 but waits for a measured branch-rollout rate.
- **Shortlist (Round 2 §d):** 1 long-horizon on-policy RL from B8 (40 % / 20 %); 2 GPU engine (30 %); 3 representation
  (30 / 20); 4 search-in-the-loop (20 / 15); 5 world model (5); 6 league (5); 7 restart archive add-on (10).

Experiments Claude may have dispatched since Round 2 (results, if any, in `../LIVE_FACTS.md`): E1, E2 (amended), E3,
E4, E5, E6, E7, E9. Treat any reported number as authoritative over your priors.

## 3. The arithmetic nobody has closed (argue with these numbers)

The only large positive effect on record changed **75 of 301,589 decisions (0.025 %)** and produced **+122.8 mass on
s3 over 32 scripted worlds, 54/64 vs 38/64 pooled survival** (`docs/research/t3_ceiling_test_2026-10-15.md:117-129`).
B8 s3's scripted census median death is frame 3,283 at length 972. Round 2's advisor 05 showed that keeping a
length-1,000 snake alive from frame 3,300 to 5,000 adds ~340 endpoint mass to that world, so rescuing ~12 % of
comparable worlds would be worth +40 pooled — **if ordinary growth is preserved**. Every fine-tune so far paid for
survival with growth (T23 −80.8 with survival unchanged or better; T3 LABELS lost 55.33 of 75.15 mass on common-alive
frames, `docs/research/t3_diagnostics_2026-10-16.md:181-185`; T4c changed 3.81 % of consequential decisions vs B8's
0.99 % and agreed on only 39.46 % of decisive rows at gap 100–300). T9 found no clean early geometric separator;
survivors occupy pockets too (`docs/research/t9_divergence_2026-10-17.md:7-20`). The gate is pooled over three mixes
(scripted 96 / frozen 32 / mixed 32 worlds, equal weight), with survival band v2 and a scripted no-death gain ≥ +.03.

## 4. The open question for Round 3

**Suppose the pilots return the likely case — a small positive early-prefix escape dividend (say +5 to +15 remaining
mass per root on the early stratum) and a pooled whole-world gain that is inconclusive at 8M. What is the transfer
function from that number to +40 pooled H5000 mass across the three mixes at the Mac gate — how many rescues per
hundred worlds, at what collateral cost on ordinary roots and on the growth rate of the rescued snake, at which death
frames and lengths, in which mix — and therefore what must the training distribution (world mix, body-length
occupancy, restart sources, exploration rate), the sample budget (hero transitions per seed, seeds) and the
evaluation (bank sizes, strata, what is reported) be so that the number we measure on Colab predicts the number the
gate will print, rather than repeating T3's "99.6 % label agreement, −75 mass" surprise?**

Answer from your discipline, not as a committee. Name the published systems you rely on (paper/system + year), give
numbers in this game's units (worlds, decisions, hero transitions, world-steps/s, L4 hours, Colab units, Sol days),
state what may differ from the Mac reference sim in training and what may not, give a kill rule for your claim, and
say where Round 2's judgment or shortlist is wrong and why. The expert-player persona should describe concretely what
the +40 snake does differently in the 50–300 decisions before a B8 death and whether that behaviour needs memory,
full-resolution geometry, or search to be expressed at 24 fps on the Mac. The red-team persona should argue that a
positive dividend will not transfer — and name the measurement that would prove it before the units are spent.
Changing the game's rules or the benchmark is not beating B8. Separate known results from conjecture. Keep to the
output format in your task prompt.
