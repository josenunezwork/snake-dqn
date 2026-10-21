# Big-lever research rounds — Round 8 question (plan round 1: stress-test the draft plan)

Judge: Fable (round-7 judge wrote this, 2026-10-21 ~05:45 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` … `../round-07/JUDGMENT.md` (each round's §(a) and §(c) supersede the previous where they say so), then
`../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over everything below) and `../QUEUE.md`. Repo:
`/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`; start with `AGENTS.md`). Never open
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Rounds 1–7's verified facts and negative record hold with the
corrections in the seven judgments; cite them, do not re-derive. Rounds 8–9 turn the critical path into stages with kill points and
budgets; Round 10 writes `PLAN.md`. **This round stress-tests the draft below.**

## 1. Goal (unchanged) and what Round 7 fixed

A served snake that beats bare B8 s3 by ≥ +40 pooled H5000 mass at equal-or-better survival on the Mac strict gate, then serving
qualification and owner OK. Colab (~1,100 u) for training/labels/evaluation; serving and gates on this Mac (6 P + 12 E cores, MPS).

Round 7 established (`../round-07/JUDGMENT.md` §a, §c):
- **Bare-net strict gates are unpaced**: T7 ran 5,760 H10000 episodes + 45 H5000 controls in **6.83 h** on ≤ 6 single-thread Mac workers;
  paced and unpaced bare outcomes are byte-identical. Only a Claim B (wall-clock) candidate arm is paced (≈ 22 h at 384 worlds, ≈ 37 h at
  640; bare comparator unpaced). 384/128/128 is the E13 transfer bank, not a sized strict design (P7: ≥ 16 worlds/mix, sized from
  development variance). The "both lines need 74 paced hours each" collision is withdrawn; the Mac is serial, not short.
- **Both lines are blocked at stage 1 on day 1:** L2′ is `dispatchable=false` under its frozen 16-u / 384-min envelope and no long-body
  Watch6 batch has ever been timed (preflight fired 0 batches); E8 is still absent (`bl_ppo/model.py:101`); E16 as frozen cannot dispatch
  (its 2×-timeout clause forces forecast ≤ 4 h; one treatment costs ≈ 5.4 h). Fixes X0/X1/X3/X4 are 0–4 u, pre-data, day 1.
- **Trust doctrine (unanimous):** Colab bare net → E13's four receipts (Mac half ≈ 1 unpaced h, Colab half ≈ 3 u per audit); wall-clock
  controller → three oracles (semantic; availability-trace replay; fresh paced) with identity = planner + scheduler + W/K + deadlines +
  fallback + load envelope; planner-over-net inherits both. May differ: batching, placement, declared float order. May not: chosen
  actions, scores, mechanics, opponents, H5000 accounting, identity fields.
- **Release:** a skill STRICT_PASS survives a product failure as evidence (T12 precedent) but a timing repair that changes the Claim B
  scheduler/deadlines/fallback is a new artifact needing fresh paced efficacy; each new registration decides its own boost clause.
- **Final family:** frozen when the first strict bank opens; contains exactly the artifacts with a frozen nominee then; Holm .05/m;
  composite p = max of clause p-values; bare wins ties; no slot added later.
- **Clause arithmetic:** under K ≤ 8, W = 4 needs S_p99 ≤ 1.458 s (W = 3 ⇒ 1.25 s, W = 2 ⇒ 0.83 s); under owner "no", W = 1 **and** K = 1
  ⇒ S ≤ 0.42 s (2.4 roots/s) or 0.63 s (1.6 roots/s); the roots/s bar is fixed by L0′'s per-doom window table (pending). The owner's
  answer is still pending — check `../LIVE_FACTS.md`.
- **Day 7 has four outcomes:** qualified / scientific KILL / INCONCLUSIVE / NOT_READY. Day-3 decision **Fri Oct 23 23:59 PT**; day-5 check
  Sun Oct 25 08:00 = integrity and remaining hours only.
- Shortlist: served search 30/18; representation 30/18; PPO backbone 22/9; engine 30 %; search-in-the-loop 15/10; archive 6; opponent 4;
  world model / league 2.

## 2. The draft plan to stress-test (skeleton of PLAN.md; Pacific time; days = Wed Oct 21 … Tue Oct 27)

**Stage 0 — Unblock (Wed–Thu, 0–8 u, three builders in parallel, one Mac ledger).** Sol-S: X0 (L2′ envelope → 64 u / 32 h, label
PACKAGE_KILL) → X1 (L4 batch-rate ruler on long-body Watch6 fixtures that fire ≥ 30 batches/mix; dispatch only if forecast ≤ 80 % of
32 h) → L2′ parity audit (4 u) → L2′ dispatch Thu night. Sol-R: X4 (E8) → X3 (E16 cap → 40 u / 20 h; E2 screen may run on L4) → E4/E5
→ D0 → E3 (16 u) → E2 (20 u). Third owner: X7 (`bl_paced_identity`, 24 Sol-h + 8 Codex-h) and X8 (L0′ window table). Claude/Codex: X5
(strict-runner + E13 timing receipt, ≤ 2 unpaced Mac-h), X6 (final family, burn manifest, look ledger, boost clauses). Kill: any stage-0
item not landed by Thu 20:00 ⇒ its dependents slip one day; L2′ not dispatched by Thu 24:00 ⇒ Line S cannot be admitted Fri 23:59.

**Stage 1 — Measure (Thu–Fri).** L2′ runs on Colab (≤ 32 h). X2 = L1′ in the first 3-h quiet Mac window (cells per clause; W 1 / K 1 only
if silent). E14 bare cells (3 Mac-h). E6+E7 (24 u, after E10), E12 (16 u) on Colab. Kill rules unchanged: L2′ PACKAGE_KILL (pooled UB90 <
40) / FUND_SYSTEMS (frozen & mixed LB90 > 0, pooled ≥ 40) / INCONCLUSIVE; L1′ no authorised cell at the published bar with zero overruns ⇒
served search unservable this week.

**Stage 2 — Day-3 decision, Fri Oct 23 23:59.** S holds the paced block (Sat 00:00 → Mon ~20:00: L3 9 h → L4 32 h → Claim B strict with
candidate paced / bare unpaced, sized from L4 variance, ≈ 22–37 h) **only if** L2′ FUND_SYSTEMS ∧ authorised L1′ PASS ∧ X7 oracles pass ∧
finish-time invoice fits. Otherwise S is deferred (keeps its status) and the block is released. R runs unpaced work (E2 screen 8.5 CPU-h,
X5, E13 Mac half) in the gaps before the block and its paced product/serving block (≈ 17 h) after it or in its place. L3 INVALID/STOP ⇒
release L4+; L4 UB90 < 40 ⇒ artifact closed; DELIVERY_READY buys L4 only, never strict admission.

**Stage 3 — Line R funding path (Fri–Mon, Colab).** E16 (40 u after X3) → E17 (20 u, 2,304 episodes, ≥ 900 steps/s) → 32M futility look
(E15 amended; no efficacy α) → declared-model 100M decision (UNCALIBRATED ⇒ balanced endpoints, 160 u envelope; pause if unfundable) →
E13 receipts (≈ 7 u + 2 Mac-h) → nominee frozen. A bare nominee exists Sun night–Mon at the earliest. Kill rules: E16 tripwires/KILL;
32M stops; E13 interval outside ±10/±20 ⇒ no strict entry.

**Stage 4 — Gates (Sun–Tue).** Each nominee: fresh strict bank (bare: unpaced ≈ 7–10 h; Claim B: candidate paced) → Watch6 product
sessions (32 paired 10-min, ≈ 11 h paced) → serving regressions (T12/T15 checks, ≈ 6 h) → owner packet (hashes, clause table, rollback
matrix, pin schema, monitoring plan). Family frozen at the first strict bank. Day-5 check Sun 08:00: hours only.

**Stage 5 — Day 7, Tue Oct 27 23:00.** One of: qualified artifact(s) awaiting owner OK / scientific KILL(s) / INCONCLUSIVE with the
registered next look priced / NOT_READY with the measured remaining work. Colab ceiling this week ≈ 240 u + E15's 160; A100/H100: none.

## 3. The open question for Round 8

**Where does this draft plan break — which single stage, receipt, assumption or dependency, if it is wrong or late, makes the week
produce nothing the owner can trust, and what is the cheapest pre-data change to the draft that removes that failure without bending any
registered rule?**

Answer from your discipline, not as a committee. Attack the draft, do not restate it: name the stage and the day it fails, give the
arithmetic (hours, units, builder-days, probabilities where a registered rule or a measured SD supports them), say what the plan should
say instead, and give the kill rule and release behaviour of your replacement. Name the published systems or methods you rely on
(paper/system + year). Separate known results from conjecture. Say where Round 7's judgment — its unpaced-bare ruling, its serial Mac
ledger, its Fri 23:59 decision point, its three-slot family freeze rule, its K/W arithmetic, or its X0–X8 list — is wrong and why.
Changing the game's rules, the opponents, or the benchmark is not beating B8. Keep to the output format in your task prompt.
