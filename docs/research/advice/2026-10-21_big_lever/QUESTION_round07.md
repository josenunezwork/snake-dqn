# Big-lever research rounds — Round 7 question (feasibility round 4: one critical path, two lines, one Mac)

Judge: Fable (round-6 judge wrote this, 2026-10-21 ~04:45 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` … `../round-06/JUDGMENT.md` (each round's §(a) and §(c) supersede the previous where they say so), then
`../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over everything below) and `../QUEUE.md`. Repo:
`/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`; start with `AGENTS.md`). Never open
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Rounds 1–6's verified facts and negative record hold with the
corrections in the six judgments; cite them, do not re-derive. Rounds 7–10 converge: Round 7 fixes trust and the calendar, Rounds 8–9
turn that into stages with kill points and budgets, Round 10 writes `PLAN.md`.

## 1. Goal (unchanged) and the two constraints that now bind

One big-lever push producing a **served snake that beats bare B8 s3 by ≥ +40 pooled H5000 mass at equal-or-better survival on the
Mac strict gate**, then serving qualification and owner OK. Colab (~1,100 units, speed over thrift) is for training, labels and
evaluation only; **serving stays on this Mac** (6 P + 12 E cores, MPS; owner decision 00:30 PT). Two lines survive Round 6:

- **Line S — served exact-sim search on this Mac** (T3 rule as a certificate controller with a reachable-root burst schedule; Claim B
  artifact whose identity includes planner + scheduler + deadlines + fallback + load envelope).
- **Line R — the Colab RL/representation line** (PPO backbone from B8 with the E8 objective; geometry residual; E16 → E17 → 32M futility
  look → 100M decision by declared-model forecast; UNCALIBRATED ⇒ balanced endpoints). Its funding rules are fixed and not re-litigated.

Both produce artifacts that must pass the same Mac-only strict gate and the same Watch6 product bars, and both need the Mac's **paced**
hours (24-fps closed-loop runs that cannot overlap): ≈ 41 Mac-h for Line S's smoke + development efficacy if it passes (L3 + L4), plus
strict banks for every nominee, plus E14 and L1′ quiet windows. The Mac calendar — not units — is the week's scarce resource.

## 2. What Round 6 established (per-advisor record in `../round-06/JUDGMENT.md`)

- **The served form that failed was mis-targeted, not merely slow.** From the smoke's 75 N=1 T3 records: 57/75 first improving
  evaluations arrived inside the *last-decision* lead (22/34 rescue-capable, 0/5 M3), but **0/75 completed before their own target
  frame** — median own-target slack 0.625 s against 5–12 s of compute (median compression needed 10×). Chronological doom→LAD scanning
  is retired; "k× faster" alone delivers nothing below ~10×.
- **The T3 rule is a certificate rule.** Bare-continuation-survives ⇒ B8 retained (`ceiling.py:140-143`; `engine.py:47-50`). In the dense
  ceiling only **440/36,996 batches (1.19 %)** had the bare continuation die within H300 — all 75 changes among them — in 22/64 worlds,
  with runs of up to 76 consecutive roots (≈ 15 s). Demand is a **burst of 4.8 roots/s for ≤ 15 s inside a doom window**, zero otherwise
  (shadow advance only). Round 5's "4.8 batches/s continuous" is withdrawn.
- **Three conditions replace it:** (i) shadow lead — 7/34 rescue intervals were already behind the live frame at detection; cold fill up to
  9.2 s; (ii) per-root latency S — the D300-rescue LAD is reachable in 27/34 at S ≤ 5 s, 17/34 at 12 s (today's one-thread latency),
  the exact H500 LAD in 61/75 at 5 s, 1/75 at 18 s; (iii) burst coverage — the reachable window holds a median 38.5 periodic roots at
  S = 5 s and closes within ≈ 12 s, so searching half of it needs ≥ 2.4 roots/s: **W = 4 ⇒ S ≤ 1.67 s; W = 2 ⇒ S ≤ 0.83 s; W = 1 ⇒
  S ≤ 0.42 s**, zero overruns, update p95 ≤ 11.688 ms. S on Watch6 opponents (FRP3 + v8) from a live-shaped world is unmeasured.
- **The owner's pending clause** (one extra CPU thread → registered W ≤ 4 worker processes + K ≤ 8 outstanding requests): under "yes",
  L1′ tests W = 1..4; under "no", only W = 1 counts and the lever is dead this week unless S ≤ 0.42 s on one thread. Check
  `../LIVE_FACTS.md` for the answer.
- **L2 is a rebuild and was never priced**: the T3 harness caps 32 worlds / 3 workers, loads census baselines, and rejects stateful
  opponents (`engine.py:99-104`, `replay.py:185-189`). L2′ = fresh 64/64/32 banks + isolated per-branch opponent runtime + parity audit
  (`research/bl_e16/PARITY_RULE.md` §1–2) + repriced envelope ≤ 32 L4-h / 64 u; labels PACKAGE_KILL (pooled UB90 < 40) / FUND_SYSTEMS
  (frozen & mixed LB90 > 0, pooled point ≥ 40) / INCONCLUSIVE (≈ 60 % likely at a true +40 with SD 200). Unpaced search is not a
  mathematical ceiling on a paced controller, but it stays the funding gate.
- **Arithmetic corrected:** with zero frozen/mixed gain the s3 row gives **+40.9 pooled** (not +31) — 0.94 of margin, 97.7 % retention
  needed; E17 evaluates **2,304 episodes** (not 864); E8 is still not landed (`bl_ppo/model.py:101`); E16's `100M_ADMISSIBLE` reads as
  32M admission.
- **Identity and claims for a wall-clock controller (adopted):** three oracles (semantic; availability-trace replay reproducing executed
  actions and scores; fresh paced runs, no best-of-N); "delivered" = consumed by the tick before its frozen cutoff for the exact
  session/life/frame/checkpoint/shadow epoch; every miss and fallback counts; one final α = .05 family across one bare and one assisted
  artifact (Holm .025/.05) with a reserved third slot for planner-over-RL-winner; development prints (E16, E15 looks, L2′, L3, L4) fund,
  never claim; six zero-overrun sessions leave a 39 % upper bound on session overrun probability.
- **Experiments in flight / queued** (results, if any, in `../LIVE_FACTS.md`): L0′ (0 u forensics), L2′ parity audit + run, L1′ burst-coverage
  probe, L3 paced smoke (DELIVERY_READY), L4 paced development efficacy (32 Mac-h), E8, D0, E4, E5, E10, E6/E7, E1b, E3, E2, E12, E14
  (five cells), E16/E17, E20 (RL endpoint as planner continuation, conditional). Treat any reported number as authoritative.
- **Shortlist (Round 6 §d):** 1 served search on this Mac (35 / 20); 2 representation (30 / 18); 3 PPO backbone (25 / 10); 4 GPU engine
  (30 %, Colab only); 5 search-in-the-loop (15 / 10); 6 archive (6); 7 opponent add-on (4); 8 world model / league (2); 1c RL endpoint as
  planner continuation (conditional).

## 3. The gap Round 6 opened

Each line now has a measurable kill point, but nobody has written the **trust rule and the calendar** that would let the owner accept a
"beats B8" claim from either artifact by the end of the week. Three things are missing. (i) **Parity/trust doctrine per artifact type:**
a Colab-trained bare net must satisfy E13's four receipts (engine identity, policy agreement with first-divergence audit, durability,
`ΔMac − ΔColab` within ±10 pooled / ±20 per mix) — none has been produced yet, and the existing CLI qualification excludes exact
continuation; a Mac-native wall-clock controller must satisfy the three oracles — none has been run; a planner-over-net inherits both.
(ii) **The Mac paced calendar:** L1′ + E14 quiet windows, L3 (≤ 9 h), L4 (32 h), strict banks for every nominee (a fresh 384/128/128
paced bank is ≈ 33 Mac-h per controller pair), serving qualification sessions — these cannot overlap, the Mac also hosts the served
game, and the owner's decision on the clause may arrive mid-week. (iii) **Kill points that release Mac time:** when Line S dies (L2′
PACKAGE_KILL, L1′ shortfall, L3 STOP, L4 UB90 < 40) its reserved hours must go to Line R's gates, and vice versa, without either line's
registered rules bending. The direction judge's "2+ weeks to first signal" and the owner's "spend it all in a week" meet here.

## 4. The open question for Round 7

**Given the measured facts above and the two lines' fixed kill points, what is the single critical path for the remaining week — the
ordered sequence of Mac paced hours, Mac quiet windows and Colab jobs, with the trust receipts each artifact type must file before its
strict gate — such that by day 7 the owner holds either a strict-gate-qualified Claim B served-search artifact, a strict-gate-qualified
bare-net artifact, or a documented kill of both, and how must the Mac's paced hours be allocated and re-allocated at each kill point so
that neither line's registered rules bend and no paced hour is spent on an artifact that cannot reach the gate?**

Answer from your discipline, not as a committee. Give the calendar in Pacific-time days and Mac-hours, the Colab jobs in units and
allocation-hours, and the Sol work in builder-days; name the published systems or methods you rely on (paper/system + year); state what
may differ between a Colab-trained or Mac-pipelined artifact and the Mac reference (batching, process placement, float order) and what
may not (chosen actions, scores, mechanics, opponents, H5000 accounting, the serving contract's identity fields); give a kill rule for
each stage you propose and say which later stages it releases; and say where Round 6's judgment — its reachable-root schedule, its
burst-coverage bars, its PACKAGE_KILL renaming, its L2′ pricing, or its one-family claim rule — is wrong and why. Separate known results
from conjecture. The release-qualification persona must say exactly what a strict pass plus serving qualification requires for a
wall-clock-dependent controller (precedents: T7 STRICT_PASS, T12/T15 serving checks) and what a Watch6 product-bar failure does to a
strict pass. The parity persona must write the receipt list for a Colab-trained bare net that has never been produced. The calendar
red-team persona must argue that the week cannot hold both lines' Mac hours and name which line to drop on day 3 under each L2′/L1′
outcome. Changing the game's rules, the opponents, or the benchmark is not beating B8. Keep to the output format in your task prompt.
