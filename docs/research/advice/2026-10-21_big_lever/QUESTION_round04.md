# Big-lever research rounds — Round 4 question (first feasibility round: where do 15 rescues per 100 worlds come from?)

Judge: Fable (round-3 judge wrote this, 2026-10-21 ~01:45 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md`, `../round-02/JUDGMENT.md` and `../round-03/JUDGMENT.md` (rulings, shortlist, the amended experiment
list — Round 3 §(a) and §(c) supersede Round 2 where they say so), then `../LIVE_FACTS.md` (measured results Claude adds between
rounds; authoritative over everything below). Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only
copy of `main`; start with `AGENTS.md`). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`.
Rounds 1–3's verified facts and negative record hold with the corrections in the three judgments; cite them, do not re-derive.

## 1. Goal (unchanged)

One big-lever push, grounded in research that works in slither.io-like environments, using GPUs/CUDA on Colab at scale
(~1,140 units, spend within a week, speed over thrift), producing a **served snake that beats bare B8 s3 by ≥ +40 pooled H5000
mass at equal-or-better survival on the Mac strict gate**, then serving qualification and owner OK. Rounds 1–3 mapped the
space; Rounds 4–7 attack feasibility, cost, parity/trust and evaluation validity; Rounds 8–10 converge on a staged plan with
kill rules. The T34 campaign (defensive search teacher; Round 0 verdict due Wed) continues in parallel and is not yours.

## 2. What Round 3 established (per-advisor record in `../round-03/JUDGMENT.md`)

- **The transfer function is closed and unanimous.** Exact ledger `ΔM = [Σ both-alive (L_cand − L_B8) + Σ cand-only L_cand −
  Σ B8-only L_B8]/5000`, equal-weight mean of three mix means. Planning form `G_m ≈ q_m · ρ · K − C_m`, `K = L(5000−d)/5000`.
  At B8 s3's scripted census median death (frame 3,283, L 972) K ≈ 340; with 90 % retained tail and 8–10 collateral, **+40
  needs ≈ 15–17 additional full-horizon survivors per 100 natural worlds in every mix**; scripted-only needs ~45–50/100
  scripted worlds; deaths after frame ~4,200 are worth ≤ 150–190 each. Collateral on record is larger than any rescue signal
  (T3 LABELS −55 on common-alive frames; T23 −81; T6 shared survivors −24..−28 cells).
- **E6's dividend is already in endpoint-mass units; its population value is `q · EPD`.** At q ≈ .4 and EPD +10 that is ≈ +4
  pooled before collateral. Round 2's "positive E6 ⇒ needs scale" and "cannot be bought by caution" were withdrawn; the frozen
  E6 label NEEDS_SCALE admits planning and the population bridge (E10), never a replication dispatch.
- **Prevalence is free:** E6 collects worlds in deterministic rotating order until 44/42/42 eligible deaths
  (`research/bl_prefix/REGISTRATION_E6.md`), so the collection ledger yields q_m and the death-stratum histogram at 0 units.
  **E10** registers, before E6/E7 gameplay, the ledger, all 24 per-root E7 prefix outcomes, and the red team's oracle envelope
  `U_i = max(0, max_24 ΔM_i)` (ineligible = 0) with the label **FINITE_MENU_INSUFFICIENT iff UB90(U) < 40** — the B8-local
  single-prefix rescue thesis cannot reach the gate even with perfect selection.
- **Opportunity is scarce and short-local.** T1: a D300-surviving alternative exists within 300 decisions in 46/49 late deaths
  but within 100 decisions in only 25/49; sealers are 13 opponent / 13 hero / 7 unresolved / 16 none
  (`docs/research/t1_commitment_study_2026-10-15.md:35-48,75-76`). T9: no clean early geometric separator; survivors occupy
  pockets too. **T35 two-deviation beam: pooled KILL, zero novel rescues** (`CAMPAIGN_LOG.md:73-76`) — two-deviation template
  search is closed at this budget. Path B served lookahead: not deliverable (LIVE_FACTS).
- **Units do not bind replication.** 100M hero transitions at 1–4k/s = 7–28 L4-h; 2 × 100M is a day of allocation. Replication
  = 2 × 100M (120M if E4 ≥ 2k/s) with a pre-registered 32M futility-only look, endpoint-only efficacy, nomination rule.
  The engine (E1, three kernel contracts, unchanged rule) is needed for 10^9 and search branch rates, not for 100M.
- **Exploration:** 1 % non-greedy ≈ 50 departures per surviving life; E7 at 5 % measures discoverability under B8, not E2's
  learner; any exploration or restart change is a separately named treatment (archive form: 80/10/10, half pre-deviation
  starts, back up 32 decisions on ≥ 70 % productive, final natural-only phase).
- **Representation:** geometry ahead of memory (expert player and representation advisor agree); E3 runs as frozen, E3b
  current-token control only if C − B ≥ 10 %; a Mac serving-cost probe (E14) must pass p95 ≤ 11.688 ms before any training
  commits to a full-resolution encoder. Function-preserving insertion `Q_B8/T + W z` with a separate richer critic.
- **Critic:** E6's EV ≥ .1 is preparation; scale qualification (E12) needs EV ≥ .3 per mix × length × horizon cell and GAE
  bias within ±5 on early roots. Retention trigger (ordinary EPD UB90 < −2 or common-alive loss < −10) ⇒ arms A/B/C.
- **Shortlist (Round 3 §d):** 1 long-horizon on-policy RL from B8 (35 % / 15 %); 1b retention sub-arm; 2 GPU engine (30 %,
  re-scoped); 3 representation (30 / 20, geometry first); 4 search-in-the-loop one-deviation only (20 / 15); 5 world model (5);
  6 league (5); 7 restart archive add-on (10).

Experiments Claude may have dispatched since Round 3 (results, if any, in `../LIVE_FACTS.md`): D0 rerun, E4, E5, E6+E7 with
E10 readouts, E1, E3, E2 (E8/E9-amended), E12, E14. Treat any reported number as authoritative over your priors.

## 3. The feasibility gap nobody has closed (argue with these numbers)

Every route to +40 now needs the same thing: a policy that, on ~15–17 of every 100 natural worlds in each mix, departs from
B8's route 50–300 decisions before an enclosure death, survives to frame 5,000 at ≥ 90 % of its length, and costs the other
83–85 worlds ≤ 10 mass each. Nothing on record has produced even one-third of that: T3's +122.8 came from exact-sim search on
32 scripted worlds (75 changed decisions in 301,589) and did not survive distillation (99.6 % label agreement, −75 mass);
every fine-tune paid for survival with growth; two-deviation beams find no new rescues; B8-local stochastic prefixes at 5 % are
being counted (E7/E10) but, if `UB90(U) < 40`, no single-prefix B8-local menu reaches the gate. The only remaining candidates for
*supplying* the frontier are (i) on-policy exploration at 10^8–10^9 transitions with a horizon-aware critic, (ii) a restart
archive of confirmed early prefixes, (iii) exact-sim one-deviation counterfactual targets refreshed on current-policy states
inside the loop, (iv) a representation that makes the exit visible (full-resolution geometry, opponent history), or (v) an
opponent-conditioned mechanism (13/49 late deaths were opponent-sealed), including curricula over opponents/worlds that raise
eligible-death exposure without destroying growth. Each has a different compute shape (CPU collection at 1–4k hero/s vs ≥ 20k
branch world-steps/s on a device engine vs 0-unit Mac probes), a different Sol build time, and a different pre-scale measurement.

## 4. The open question for Round 4

**Given the required rescue rate (≈ 15–17 added full-horizon survivors per 100 natural worlds in every mix, ≤ 10 collateral),
which frontier-supplying mechanism can plausibly discover and make executable that many early route corrections within this
week's compute (three L4 slots, ~1,000 units, CPU collection unless E1 qualifies a device engine, Sol builders in days), and
what is the single pre-scale measurement — computable from E6/E7/E10, E3, E12, E14 or one new ≤ 20-u job — that ranks the
mechanisms against each other *before* any 100M tranche, so the week's units go to the mechanism with a measured route to 15/100
rather than to the one with the best story?**

Answer from your discipline, not as a committee. Name the published systems you rely on (paper/system + year), give numbers in
this game's units (worlds, decisions, hero transitions, branch world-steps/s, L4 hours, Colab units, Sol days), state what may
differ from the Mac reference sim in training and what may not, give a kill rule for your claim, and say where Round 3's
judgment or shortlist is wrong and why. Treat `UB90(U) < 40` (FINITE_MENU_INSUFFICIENT) and `UB90(U) ≥ 40` as two scenarios
and say what your mechanism does in each. The expert-player persona should say what fraction of late deaths have *any*
executable early correction that keeps growth, by sealer type (hero vs opponent), and what the correction looks like when the
sealer is an opponent. The red-team persona should argue that no mechanism reaches 15/100 at equal growth within this compute
and name the measurement that would prove it this week. Changing the game's rules or the benchmark is not beating B8. Separate
known results from conjecture. Keep to the output format in your task prompt.
