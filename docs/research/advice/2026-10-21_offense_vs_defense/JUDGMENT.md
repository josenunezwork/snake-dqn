# Offense vs defense — independent Fable judgment (2026-10-20 ~22:40 PT)

Judge: Fable (wrote none of the five Astra answers). Read-only; verified every load-bearing fact below in the
repository at `439d0ee4` (advisors cite `61b8f751`; the cited lines are unchanged between the two). Sealed Round 0
study directory not opened; reasoning from source and registrations only. No jobs, no Colab, no tmux.

## Bottom line (≤ 10 lines)

1. **The quoted premise is wrong on four of its five facts** and right on one. Mechanics v2 (full corpse drops, boost
   trail pellets, kill credit, size-resolved head-ons) is implemented in all three engines and is the **served default**.
   B8 s3 kills (19 in 25 Watch6 sessions; 332 credited kills in 36 terminal worlds) and boosts 0.12–0.17 %, not 89 %.
   T3-dense's "+92" is the *pooled s0+s3* figure; the served checkpoint's own row is **+122.8** (563.9 → 686.7). It is
   a reused-bank, scripted-only, unpaced result, not a ceiling. The one correct claim: T34's ranker is purely defensive.
2. **Biggest credible gain over B8 s3 is still defense**, because the endpoint is mass *integral*: B8 s3 dies in 15/32
   scripted worlds at median frame 3283 / median length 972; keeping that body alive to H5000 is worth ~300 mass in
   such a world. Offense in *this* economy is capped hard: ≤ 150 corpse pellets board-wide (shared FIFO with trails),
   one pellet per tick, mean opponent death length 42, and deaths are already abundant (22–75 per game). A kill cannot
   "double you in seconds" here; the historical cut-off census bounded untaken forced kills at ≈ +4.7 mass.
3. **Decision: no pivot.** Path B stays the critical path; Round 0 analysis proceeds after a one-file erratum. Fund one
   bounded offense lane (**O1 harvest census, ≤ 8 u L4**) that tests *harvesting of existing corpses*, not hunting; it
   must beat the rescue controller, not just B8, to earn O2 (≤ 24 u, conditional). No mechanics change this campaign.
4. **v1-annotation ruling:** a provenance-label defect, not a validity problem — the constructor cannot produce v1.

---

## 1. Facts I verified (file:line)

### (a) Mechanics v2 is implemented and is the served default

| Fact | Evidence |
|---|---|
| Kill credit on body collision; v2 size-resolved head-on (≥ 1.15×) with credit to the winner | `src/simd_env/grid_sim.py:831-873`; `src/core/mechanics_constants.py:21` (`HEADON_SIZE_RATIO = 1.15`); live `src/game/game_state.py:699-723` |
| Death drops: v1 every other segment (ambient class), v2 every segment (corpse class) | `src/simd_env/grid_sim.py:875-896` (`stride = 1 if self.v2 else 2`); live `src/game/game_state.py:601-630` |
| Boost trail pellets (v2) | `src/simd_env/grid_sim.py:509-517`; `src/core/mechanics_constants.py:93-95` |
| Corpse-class cap = `max(0.5 × max_food, 10)` = **150** at `max_food 300`, oldest evicted first, shared by corpses and trails | `src/core/mechanics_constants.py:24-64`; `src/simd_env/grid_sim.py:435-444` |
| Served default: v2 on unless `SNAKE_MECHANICS_V2=0`; ego2s (B8) checkpoints are routed to the 61-D v2 world | `web/backend/session.py:323-345`, `:755-760`; `configs/mechanics_v2.yaml:32` (`mechanics_version: 2`) |

This is a source/default verification, not an inspection of the running server's environment. One real difference
between benchmark and product: the benchmark profile (`research/apex_safety_20260926/deployment.yaml:36`) uses
`initial_food: 300`; the served config (`configs/mechanics_v2.yaml:36`) uses `initial_food: 250`, both `max_food 300`.
Top-up runs each tick, so this is a transient, not a different economy. The served Watch roster is bare B8 s3 vs
frp3-s12+v8 (`docs/STATUS.md:27-31`).

**"Mechanics v2 was never implemented" is false.** The July blueprint was implemented; the memory note saying "NOT yet
implemented" is stale.

### (b) B8 s3's kills, death-drop share and boost share

Repository measurement (served profile, continuous respawn, 25 paired H5000 mixed-Watch6 sessions):
`research/t13b_six_snake_20261017/results/summary.json:390-405` — hero `kills_made 19`, `kills_taken 16`,
`boost_rate 0.001728` (216 / 125,000 decisions), deaths 37 (21 self, 14 enemy body, 2 head-on), mass 316.76.
Table form at `docs/research/t13b_six_snake_roster_2026-10-17_results.md:28-35`. Path B's bare arm: 0.12 % delivered
boost on 190,779 decisions (`docs/research/final_run_20261019/CAMPAIGN_LOG.md`, 10-20 ~19:55 row).

Astra-measure's local diagnostic (36 terminal-hero H5000 worlds, 12/mix, B8 s3 `3e649783…`):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/run02/` — receipt and script hashes
re-computed by me and matching (`receipt.json` 005f4453…, `measure.py` 77ae1074…, `per-game.csv` 588605c0…).
Pooled: 332 credited kills; gross intake 20,952 ambient / 16,467 death-drop / 4,049 trail (death drops **39.7 %**;
per mix 59.8 / 29.6 / 26.8 % scripted / frozen / mixed); 206 boosted frames of 164,514 (**0.125 %**); 1,563 opponent
deaths at mean length 42; the 9 deaths at length ≥ 800 each retained exactly 150 new pellets.

**Method review of `measure.py` (I read all 250 lines):**
- Uses the *same* world constructor as T34 (`bank_engine.build_world` → pinned `deployment.yaml` profile), asserts
  `sim.cfg.mechanics_version == 2` before play, and records `dataclasses.asdict(sim.cfg)` per world; I confirmed
  `world-00.json` records `mechanics_version 2, max_food 300, initial_food 300, num_snakes 6, frame_rate 1`.
- Instrumentation wraps seven sim methods; the original dynamics execute exactly once; death-drop bookkeeping is replayed
  in side state and asserted equal to the native food/corpse sets after every drop. Parity: three worlds re-run
  uninstrumented with identical action+state hash chains (12,999 frames). Per-world mass-balance identity asserted.
- **Limits that matter for interpretation** (the author states most of them; I add emphasis):
  1. `kills` is the sum of `get_kill_credit()`, which includes *passive* body-collision credit. The 2026-10-03 census
     (`git show 10ca76af:docs/research/kill_opportunity_census_2026-10-03.md`) found ~81 % of champion+v8's kills were
     passive and only 3.7 % matched a flagged cut-off. So "332 kills" is **not** evidence of hunting; it is evidence
     that long bodies get run into. Likewise "near head/body" is proximity, not causation.
  2. Death-food share is a *gross intake* share. It does not say how much of it B8 is leaving on the board, nor what a
     different policy would collect; it is a census, not a counterfactual.
  3. The scripted mix's 60 % death-food share reflects greedy-food anchors dying ~75 times per game (respawn timer 1):
     a conveyor of small corpses. On the mixes that look like the product (frozen/mixed) death food is 27–30 %.
  4. 12 worlds per mix; seeds audited against the repo only (not artifacts) — diagnostic, never a gate bank.
- **Verdict on the method:** sound for what it claims (descriptive source accounting); no disqualifying flaw. Its
  conclusion that "the current source of growth" ≠ "the largest remaining lever" is the right reading.

**The "~0 kills / ~89 % boost abuse" numbers belong to June 2026 Apex agents** (`git show
bf4d147:docs/history/slither_training_log_20260615.md`, lines 134-161, recovered by the skeptic; the file is absent at
HEAD). They do not describe B8 s3. The owner's memory note is stale on this point as well.

### (c) T3 "+92" vs the B8 s3 row

`docs/research/t3_ceiling_test_2026-10-15.md:117-120`:

| Arm | B8 survival | Dense survival | B8 mass | Dense mass | Δ |
|---|---|---|---:|---:|---:|
| b8_s0 | 21/32 | 28/32 | 652.259 | 713.349 | +61.1 |
| **b8_s3 (served)** | 17/32 | 26/32 | 563.850 | 686.662 | **+122.8** |
| pooled | 38/64 | 54/64 | 608.054 | 700.005 | +92.0 |

So "+92" is the pooled two-checkpoint number; the served checkpoint gained +122.8. Caveats (all in the source):
reused round-52 scripted census worlds, scripted mix only, dense trigger every 5th decision at L ≥ 600, D=300, no
deadlines, 75 changed actions in 301,589 decisions; 6 s3 deaths remained (lengths 1218, 1200, 1020, 53, 559, 1067).
It is neither a ceiling nor a three-mix nor a paced result. Note also that it was measured **under mechanics v2**
(same constructor path as above), i.e. the defensive gain already lives in the corpse economy.

### (d) The "v1" annotation on the Round 0 / T34 bank — RULING

What is annotated: `research/t34_round0/BANK_ROUND0.json:82` (`"mechanics_version": 1`), hard-coded at
`research/t34_round0/bank.py:142`; prose at `research/t34_round0/PREREGISTRATION_ROUND0.md:46` ("mechanics v1");
repeated in Round 0-D (`research/t34_round0d/bank.py:261`, `PREREGISTRATION_ROUND0D.md:87`).

What actually runs (traced, every link read):
`research/t34_round0/runner.py:330` `build_world(row, checkpoint)` → `research/t34_exit_b8/trajectory.py:106-112` →
`research/phase0_search_20261019/teacher.py:101-113` → `research/t3_rollout_labels_20261015/bank_engine.py:400-435`
→ `grid_h5000_identity._context(1)` (`research/redesign_scope_20261007/grid_h5000_identity.py:258-281`), which
SHA-checks and loads `research/apex_safety_20260926/deployment.yaml` (`game.mechanics_version: 2` at line 32) and
checks the profile digest `d396d3ed…` (`research/apex_safety_20260926/dev_screen.py:93`) → `run_simd_eval(profile=…)`
→ `src/simd_env/eval_engine.py:469` takes `mechanics_version` from `profile.world`, **not from the bank JSON**.
Nothing consumes the bank's `profile.mechanics_version` field (grep across `research/t34_*`, `phase0_search`,
`bank_engine.py`, `eval_engine.py`: only `bank.py:142` writes it; no reader).

**Decisive:** `src/evaluation/protocol.py:106-107` — the `promotion-v2-watch-rect` profile raises
`"… requires rectangular mechanics v2"` unless `world.mechanics_version == 2`. The profile named in the bank
(`"base": "promotion-v2-watch-rect"`) cannot be instantiated under v1. Empirical confirmation: Astra-measure's `run01`
asserted v1 through this exact constructor and failed before gameplay (`measure/run01.log`), then passed asserting v2.

The registration hash-binds the executed world identity: `research/t34_round0/REGISTRATION_ROUND0.json:42`
(`deployment.yaml` 4146baa3…) and `:721` (`mechanics_constants.py`), alongside the mislabelled files (`:526` bank,
`:531` prereg, `:537` bank.py). Path B's registration binds the same files
(`research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json:299-301,400`).

**Ruling: provenance-label defect, not a benchmark/validity problem.**
- Round 0 and Path B ran (run) **v2**, i.e. the served mechanics — the comparison is to the right game.
- Both arms of every pair share the same world; a wrong label cannot bias a paired contrast.
- The executed configuration is hash-bound by the registration even though the human-readable label is wrong.
- Astra-skeptic's "Round 0's declared v1 benchmark is a different economy" is **incorrect**; Astra-mechanics'
  "provenance defect, resolved by code tracing" is **correct**.

**Required before Round 0 analysis (minutes, no re-run, no relabel, no edit of frozen files):**
1. A sibling erratum file (new file, e.g. `research/t34_round0/ERRATUM_MECHANICS_LABEL.md`, committed to main) stating
   that `profile.mechanics_version: 1` in `BANK_ROUND0.json`/`bank.py`/`PREREGISTRATION_ROUND0.md:46` is a labelling
   error; the executable path selects mechanics v2 (cite the chain above and `protocol.py:106-107`). Do not touch the
   hash-bound files.
2. The already-registered Mac replay (deterministic, 10 % of pairs) should **attest** `sim.cfg.mechanics_version == 2`
   and record `asdict(sim.cfg)` in its receipt. If adding that line to the replay harness would alter a bound source
   file, run the attestation instead as a separate one-world check on an *excluded preflight fixture* row (never a
   study world) and file its receipt next to the erratum. This changes no bank, look, statistic or candidate identity.
3. Round 0-D (stage 2 not ratified; pre-data): fix the label in `research/t34_round0d/bank.py:261` and
   `PREREGISTRATION_ROUND0D.md:87` *before* stage-2 ratification and regenerate its hashes — the stage-1 judge already
   requires regeneration after the contract append, so this costs nothing extra.

Nothing here delays analysis beyond writing one file. Do not reopen or reinterpret any Round 0 outcome on account of
the label.

### (e) T34's ranker is purely defensive (the premise's one correct claim)

`research/t34_exit_b8/teacher_v2.py:253-287`: if bare survives → bare first, other survivors by Q; else survivors by Q;
else by alive steps. Mass is not an ordering key. Search is admitted only on a confirmed doomed shadow
(`teacher_v2.py:756-767`). Any mass-seeking behaviour requires a new trigger *and* a new ranker; it is a different
controller, not a scoring tweak (Astra-search and Astra-mechanics both say this correctly).

## 2. Where is the biggest credible mass gain — the answer

**Defense, by the arithmetic of the endpoint.** The skill bar is H5000 mass integral with dead frames zero
(`docs/research/final_run_20261019/SUCCESS_CONTRACT.md:27-30`). B8 s3's terminal-hero failure mode is a large body
dying late (census: 15/32 scripted deaths, median frame 3283, median length 972 —
`docs/research/panel_mac_answers_2026-10-14.md:79-96`; the 36-world diagnostic above: 12/36 deaths, stored lengths
738–1,454). Holding a length-1,000 body alive from frame 3,300 to 5,000 is worth ≈ 340 mass-integral units in that
world even with zero further growth, and a surviving body keeps collecting. That is why 75 changed decisions produced
+122.8 on s3. Nothing offensive in this economy has a payoff of that shape.

**Offense is bounded by the rules, which the owner's text mis-models.** In our v2: a corpse of any size leaves at most
150 pellets (`mechanics_constants.py:46-64`), oldest-first eviction biases survivors to the victim's tail, pickup is
one pellet per snake per tick, and opponents respawn immediately so deaths are already plentiful (22–75 per game).
A length-1,000 hero cannot double from one kill; the optimistic single-corpse ceiling is +15 % length, collected over
≥ 150 ticks, worth ≤ +75 mass-integral if taken at the midpoint. The only measured hunting census (champion+v8,
2026-10-03) found forced cut-offs ≈ 0.6/episode with an upper bound of ≈ +4.7 mass, and 81 % of credited kills
passive. B8 already draws 27–60 % of its gross intake from death drops without trying. Nothing in the repository
supports "kills are where most of the upside is" for this simulator — and the served game runs the same rules, so
the conclusion transfers to the product.

**Where offense could still pay: harvesting, not hunting.** Frozen/mixed death-food share is 27–30 % while the board
carries up to 150 corpse pellets at a time; B8 has no notion of racing to a fresh corpse before eviction/competitors.
A mass-ranked one-action deviation at opportunity states is cheap to test with the exact engine (branch mass already
sums post-step hero length — `research/t34_exit_b8/engine.py:320-355` per Astra-search). My prior, consistent with
the census and the 1-pellet/tick limit: **+10 to +40 mass if it works**, concentrated on frozen/mixed; well under the
defensive lever but additive to it and, usefully, *not latency-critical* the way rescues are (a corpse persists for
many ticks; a rescue has a 4–16 decision window). That makes it a natural Path B add-on rather than a competitor.

**Is the owner's quoted analysis right?** No. Right that T34 is defensive and that search could in principle
score mass. Wrong that v2 is unimplemented, that B8 has ~0 kills / 89 % boost, that +92 is defense's ceiling (served
row +122.8), and — most importantly — wrong about the game's economics: our corpse cap and pickup rate make
"encircle, kill, double" a slither.io intuition that does not transfer. Changing the rules to make it true would
create a new game and a new benchmark, requiring B8 to be re-baselined; that is a product decision for the owner and
not a path to "beats B8 s3".

## 3. What the campaign should do now

1. **Keep the direction reset** (`CAMPAIGN_LOG.md`, 10-20 ~21:58 DIRECTION RESET): Path B deliverability smoke
   tonight, operator-02 chain if it passes; Round 0 Mac replay + analysis as soon as chunk 002 publishes — **after
   the erratum in §1(d)**. Claim A D0, Round 0-D stage 1, unpaced Path B arm on Colab as adopted. Nothing in this
   question changes those decisions.
2. **No mechanics change, no kill-reward sweep, no offensive RL/league restart, no "fix the sim first".** The sim is
   fixed; the premise was stale. T23's league fine-tune (−80.8) and every training-based lever since B8 have lost.
3. **Add one bounded offense lane, O1 (below).** Sol implements; Claude operates; freeze before data; one look.
4. **Update the stale notes** the premise came from: the memory/blueprint line "mechanics v2 NOT yet implemented" and
   the "~0 kills / ~89 % boost" characterisation should be bannered as June-2026/Apex-era in `docs/STATUS.md` or the
   INDEX, so no future advisor rebuilds this question. (Doc-only; not a registration.)

### O1 — harvest-branch census (adopted; the only new evidence lane)

- **Hypothesis.** At decisions where bare B8 s3 survives its H500 continuation, a single legal first-action deviation
  followed by B8 yields ≥ +5 mean branch mass (`delta_A500 = (Σ len_alt − Σ len_bare)/500`, dead frames zero) with
  equal horizon survival, and the gain is attributable to death-drop intake (per-branch food-provenance ledger), in
  at least 8 of 48 worlds including ≥ 4 frozen/mixed.
- **Design.** 48 fresh collision-audited B8 s3 terminal H5000 worlds, 16/mix, same constructor/profile/checkpoint/
  rosters as Round 0 (mechanics v2, and *labelled* v2). Roots per world ≤ 8: 4 time-stratified (frames 1000/2000/3000/
  4000, hero alive) + up to 4 event roots (opponent death within Manhattan 30 of the hero head, 64-decision dedupe,
  fixed reservoir). At each root: all advisory-legal first actions (≤ 6), H500 continuation, cloned opponent policy/
  RNG state (reacting, not replayed), rank by `delta_A500`, ties to bare. Provenance observer (Astra-measure's
  instrumentation pattern, parity-checked instrumented vs bare on 3 worlds) tags ambient / death-drop / trail intake
  per branch. Rescue behaviour is not part of O1.
- **Compute.** One L4 lane via the qualified CLI supervisor. Cap **8 u / 4 runtime-h total**, of which ≤ 1 u is a
  disjoint timing fixture (2 worlds, 2 roots each) that must project the full census inside the remaining cap before
  the bank is admitted (branch work ≤ 48×8×6×500 = 1.15 M row-steps + base trajectories; at the measured 377 rows/s
  that is ≈ 1.2 runtime-h). Mac: none (the Mac is Path B's). Reserve 20 % inside the cap; cap expiry = INCOMPLETE.
- **Kill rule (frozen before data).** INVALID/INCOMPLETE first (parity, provenance, accounting, missing worlds,
  cap). **KILL this controller family** if < 8/48 worlds (or < 4 frozen+mixed) contain a surviving branch with
  `delta_A500 ≥ 5` whose gain is death-drop-attributed. Otherwise GO to O2. A KILL closes single-action H500
  harvesting, not "offense"; no extension, no second look.
- **What it must beat to matter (O2, conditional, ≤ 24 u / 12 L4-h, registered separately on O1 GO).** Three frozen
  controllers on 96 fresh paired worlds (32/mix): bare B8, rescue-only D (the Path B v2 rule), and D+O (rescue
  priority; harvest trigger only while bare survives; take the top harvest branch only if `delta_A500 ≥ 5` and
  death-drop-attributed; same total branch budget as D). Unit = paired world; one look; Holm over the two contrasts.
  O2 advances only if **(D+O − D) has LB90 > 0** *and* D+O vs B8 meets the contract's pooled +40 / LB90 > 0 and
  survival guards. A gain over B8 alone would be D's; the skeptic is right that attribution needs the D contrast.
  Anything that passes then enters the normal Mac paced test → strict gate → serving qualification → owner release.
- **Operational rule.** O1/O2 never touch the Mac calendar, the Round 0 bank, or Path B's registration. The direction
  reset's Colab rule applies (pre-approved spend, price per decision). Combined offense exposure this tranche: 8 u
  now, 32 u worst case.

## 4. Who is right where the advisors disagree

| Disagreement | Ruling |
|---|---|
| Round 0 bank "v1": different economy (skeptic) vs label defect (mechanics, measure) | **Mechanics/measure.** `protocol.py:106-107` makes v1 impossible under this profile; run01's assertion failure proves the constructor yields v2. |
| "Measure first" needed before anything (owner text) | **Done** — Astra-measure's 36-world census answers it adequately for planning; do not fund a larger Mac census (mechanics' step A, 128 worlds + 32 Watch sessions) — the Mac is the bottleneck and the number it would produce is not decision-bearing beyond what O1's ledger gives. |
| Size of the first offense probe: 96 roots/H300/≤1 u (measure) vs 48 worlds×8 roots/H500/8 u (search) vs 96 worlds×24 roots/16 u (pm) | **Search's size, with measure's mass ranking and pm's attribution predicate.** H300 is too short to intercept *and* collect ≥ 100 pellets at 1/tick; 96×24 roots is more than a mechanism screen needs. |
| PM's 72-hour, ~427-u portfolio envelope | Out of scope for this question; I adopt only the offense line items (8 u + 24 u conditional). The remaining burn goes to lanes already adopted (Round 0-D, D0, unpaced Path B arm, R1 labels on a Round 0 GO). |
| Should offense be scored by kills / victim length | **No** (all five agree, correctly): score realised hero mass with dead frames zero; kills and corpse size are diagnostics. |
| "Boost only pays for kills" | Rejected; mechanics and skeptic are right that an escape boost preserves mass integral. B8's 0.12–0.17 % boost is not "abuse"; whether it is under-use is untested and not this campaign's lever. |

## 5. What would change my mind

- **O1 GO with ≥ 12/48 worlds and gains concentrated on frozen/mixed** → fund O2 immediately and plan D+O as the Path B
  candidate identity for the *next* registration (not the running one).
- **Path B smoke fails deliverability at both N=1 and N=6** → served search is dead for rescues *and* harvests; the
  pivot is then to a trained prior (Claim A with a working loss), not to offline offense. Offense-by-search inherits
  the same deadline problem, only more leniently.
- **Owner decides to change the game** (raise/remove the 150 corpse cap, slither-style self-pass-through) → new
  benchmark; re-baseline B8 under the new rules in a 2×2 (policy × rules) before any "beats B8" claim. Not this
  campaign.
- A fresh, same-rules, paced candidate that farms the served FRP3+v8 roster for ≥ +40 equal-mix mass with LB90 > 0 and
  passes the product death-rate bar would put offense on the critical path — until then it is an add-on.

## 6. Receipts

Verified by me this session: `web/backend/session.py:323-345,755-760`; `configs/mechanics_v2.yaml:32-40`;
`research/apex_safety_20260926/deployment.yaml:32-40`; `src/core/mechanics_constants.py:21-100`;
`src/simd_env/grid_sim.py:430-445,505-520,830-900`; `src/game/game_state.py:598-632,697-726`;
`src/evaluation/protocol.py:98-112,264-287`; `src/scripts/tournament_eval.py:169-223`; `src/simd_env/eval_engine.py:435-475`;
`research/t34_round0/{BANK_ROUND0.json:75-83,bank.py:120-150,runner.py:320-340,PREREGISTRATION_ROUND0.md:35-75,REGISTRATION_ROUND0.json:42,526-537,721}`;
`research/t34_exit_b8/{trajectory.py:100-118,teacher_v2.py:250-290,750-790}`; `research/phase0_search_20261019/teacher.py:95-120`;
`research/t3_rollout_labels_20261015/bank_engine.py:367-470`; `research/redesign_scope_20261007/grid_h5000_identity.py:250-290`;
`research/apex_safety_20260926/dev_screen.py:85-115`; `docs/research/t3_ceiling_test_2026-10-15.md:100-135`;
`research/t13b_six_snake_20261017/results/summary.json:385-412`; `docs/research/t13b_six_snake_roster_2026-10-17_results.md:20-40`;
`docs/research/final_run_20261019/{CAMPAIGN_LOG.md:55-100,SUCCESS_CONTRACT.md:1-60}`; `docs/STATUS.md:20-45`;
`git show 10ca76af:docs/research/kill_opportunity_census_2026-10-03.md`; Astra-measure artifacts
(`measure.py` full read, `run02/{receipt,audit,plan,world-00}.json`, `run01.log`, `source-checks.json`; SHA-256 recomputed).
