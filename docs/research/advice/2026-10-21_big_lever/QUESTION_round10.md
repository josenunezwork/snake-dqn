# Big-lever research rounds — Round 10 question (final: red-team the PLAN skeleton the round-10 judge will turn into PLAN.md)

Judge: Fable (round-9 judge wrote this, 2026-10-21 ~03:15 PT — real wall clock; Rounds 7–8's "~05:30/06:30 PT" stamps were ~3.5 h fast).
Charter: `../CHARTER.md` (read it first), then `../round-01/JUDGMENT.md` … `../round-09/JUDGMENT.md` (each round's §(a) and §(c)
supersede the previous where they say so), then `../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over
everything below) and `../QUEUE.md`. Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`;
start with `AGENTS.md`; it may be behind main — Round 9 §(top) lists what landed after `6e05c52f`: L2′ AMENDMENT_08 prose at
`research/bl_served_search/AMENDMENT_08.md`, the E1b CUDA dtype fix). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`.
Rounds 1–9's verified facts and negative record hold with the corrections in the nine judgments; cite them, do not re-derive.
**This is the last round. The round-10 judge writes `../PLAN.md` from your answers. Attack the skeleton below as executable text;
every sentence you cannot sign must come back with replacement text an operator can act on alone at 03:00.**

## 1. Goal (unchanged) and what Round 9 fixed

A served snake that beats bare B8 s3 by ≥ +40 pooled H5000 mass at equal-or-better survival on the Mac strict gate, then serving
qualification and owner OK. Colab (~1,140 u; cap 3 measured; E4 running, E3/E1b relaunches pending, O2 being built) for training/labels/
evaluation; serving and gates on this Mac (6 P + 12 E cores, MPS). Owner's W/K serving clause **still pending** (silence ⇒ W 1 / K 1);
Round 9 adds a separate **probe-only** ask for L1′'s expanded cells.

Round 9 established (`../round-09/JUDGMENT.md` §a, §c; all verified against code):
- **Stage 2 is three dated admissions, not one Friday predicate.** X7's "three oracles PASS" is L3 evidence (`receipt.py:160-163` needs
  108 fresh Darwin runs); the Friday instrument is `l3_identity_prerequisites = PASS` (`receipt.py:412-417`). L4's variance exists Sun
  17:00 at the earliest, so Friday carries a **strict-time ceiling** `C_strict = 95 − 9 − 32 − 10.67 − 6 − 2 − 6 − (post-Friday X7 replay +
  semantic hours) ≈ ≤ 29 h`; the post-L4 gate sizes from paced variance and defers (NOT_READY) if the sized bank exceeds it — N is never
  shrunk. 384 candidate-paced worlds = 22.2 h (7.1 h left); 640 = 37 h (refused).
- **Two pre-data defects in `research/bl_paced_identity/`:** one `cutoff_ns` is checked at visibility (`controller.py:60`) and at
  consumption (`:268-277`) while the two-tick publication margin lives in the unbuilt adapter — fixed by two cutoffs
  (`publish_cutoff_ns = T_r − 2 ticks`, `consume_cutoff_ns = T_r + 1 tick`) and a fake-clock four-case test (Z-2); and the W/K guard
  (`events.py:144-157`, `transport.py:66-71` forces CPU) means expanded L1′ cells need a recorded owner **probe-only** authorization and an
  MPS device adapter, else NOT_MEASURED.
- **Y0 is prose-only** (`AMENDMENT_08.md`): `readiness.py:41` and `manifests.py:66-95` still enforce the single-plan 16-u / 384-min
  envelope; Z-0 lands the chunk-aware code. **R ≥ 2.18/s is per occupied 12-worker runtime regardless of parallelism** (Round 8's
  "1.09/s on two slots" withdrawn); Y1's ruler decides; R < 2.18 ⇒ NOT_DISPATCHABLE.
- **Line R is blocked at its root:** E9 BLOCKED (.270/.307 < .60; escalation exhausted), so "E2 when a slot frees" is dead text; a
  **prospective E9 replacement** (Z-4: 192/64/64 fixed; 128/48/48 with E6 reported not Holm-coupled; two-stage sequential) must be
  admitted by the funding judge before any E2. E3 may GO with `selected_architecture = null`; E16 is A/G-only; **earliest executable
  E16 is next week under any outcome.** "Funded E15 at 192 u" withdrawn; 320-u envelope (64 reserve, 2× liability, Y3 pause points) is
  the ratification target (Z-5). R's day 7 = disposition packet with explicit blockers.
- **Day-7 prints a tuple per line** — scientific verdict / readiness / release status — with the terminal reducer (registered stop ⇒
  scientific KILL scoped to that study; mechanical ⇒ NOT_READY; completed strict FAIL ⇒ KILL of that artifact; product or serving FAIL
  ⇒ RELEASE_HOLD with SKILL_PASS preserved). Owner front page: exactly one of two signature texts (release authorization vs
  acknowledgment).
- **Week 2: one family per artifact, no retry, no alpha split.** Handoff records FAMILY_UNOPENED / FAMILY_OPENED{members, p, outcome};
  week-2's family is registered prospectively from artifacts without a prior strict look; an S deferred *unopened* re-enters with the
  same identity; a repaired S is a new artifact; fresh banks always.
- **Stage 0 has item-specific gates:** Thu 12:00/20:00 for Y0-code/Y1/X3/Y4/Y6/Y7/Y9; **Fri 12:00/20:00 for X7 adapter + clock fix +
  replay adapter + Y5**; Y3 before E17/32M; Y8/Z-5 before any 100M; X8 before Friday. One writer + different named reviewer per item;
  Codex-I builds, Codex-V verifies. A slip blocks dependents only.
- **Power statement (must open PLAN.md):** with measured SDs (365/263/212) L2′ pooled SE 25.7, L4 SE 29.3 (half-width 37.6), strict
  another point filter ⇒ unconditional qualification ≈ 5–8 % at a true +40, ≈ 60–65 % at a true +80. The week qualifies a T3-scale
  effect; it cannot distinguish +40 from 0 on any single stage.
- **O1 = GO** (35/48 fresh worlds hold a corpse-harvest branch ≥ +5); **O2** (bare vs D vs D+O, ≤ 24 u) is registered and unpaced; a
  harvest-aware served operator is a *different Claim B identity* and a week-2 item. Shortlist: served search 25/14 (+O2 upside);
  representation 25/14; PPO backbone 18/8; engine 28 %; search-in-the-loop 15/10; harvest operator 5 add-on; archive 6; opponent 4;
  world model / league 2. A100/H100: none (14 rounds unanimous).

## 2. The PLAN skeleton (what PLAN.md will say; Pacific time; Wed Oct 21 … Tue Oct 27; week-2 handoff)

**§0 Scope and power.** One release track this week (S = served exact-sim search, Claim B "Assisted hero — [controller ID]"); R
(long-horizon PPO + geometry) is research-only and cannot nominate before week 2. Power statement verbatim as above. Release claim =
Mac strict PASS (candidate paced, bare unpaced under its registered clock-independence receipt) ∧ product PASS ∧ serving PASS ∧
independent audit ∧ owner signature; nothing else.

**§1 Stage 0 — dispatch readiness (Wed–Fri).** Sol-S: Z-0 (chunk-aware readiness/forecast) → Y1 (nine-fixture ruler, L4 half ≤ 4 u + Mac
adapter) → Y7 → parity audit (4 u). Sol-C: Z-2 (two cutoffs + fake-clock test) → X7 production adapter (root selection, EDF, 32→96→192→300
checkpoints, lead cap 432, K accounting) → Y5 (packet / pin schema / label / rollback matrix). Sol-R: Z-4 (E9′ designs → funding judge)
→ Y3 (exact-continuation durability ≤ 8 u) → Z-5 (E15 320-u envelope). Codex-I: Y6 (family automaton + OC incl. L4 transitions), Z-6
(create-only admission receipts, L4 tuple, day-7 reducer, typed E13 index, signature texts), Z-7 (handoff schema), Y9/Y10 templates;
Codex-V: independent acceptance of every item. Gates: Thu 12:00 code / 20:00 acceptance (Z-0, Y1, X3, Y4, Y6, Y7, Y9, Z-6); Fri 12:00 /
20:00 (Z-2, X7 adapter, replay adapter, Y5); Y3 before E17/32M; Z-5 before any 100M; X8 before Fri 23:59. Kill: a missed gate blocks
its dependents and never extends the week's reservation.

**§2 Stage 1 — measure (Wed–Fri).** Colab, slots per Z-1 (record terminal export before reassigning): Y1 → E3 relaunch → E1b relaunch →
L2′ chunks 1–4 (immutable 16/16/8 manifests, ≤ 384 timeout-min each, R ≥ 2.18/s per runtime, one look after four terminal chunks,
chunk fault ⇒ study INCONCLUSIVE) → O2 → E2 only after E9′ admission. Mac: Phase 0 idle *receipt* → L1′ K1 cells (Z-3; expanded cells +
MPS only with probe-only authorization) in the first window after Y1's Mac adapter is accepted → X5 (2 h) → E14 bare cells on the E3
architecture (Thu, 3 h, after E3's architecture receipt). Labels as registered (PACKAGE_KILL / FUND_SYSTEMS / INCONCLUSIVE /
NOT_DISPATCHABLE; L1′ PASS / FAIL / NOT_MEASURED).

**§3 Stage 2 — three admissions.** (A) **Fri 23:59 provisional reservation** (`admission/friday.json`): four L2′ chunks terminal+valid with
aggregate FUND_SYSTEMS ∧ authorised L1′ cell PASS (clause branch as answered; silence ⇒ W1/K1, C p99 ≤ 333/542 ms) ∧ production
`l3_identity_prerequisites = PASS` ∧ Y5 + L3/L4 registrations frozen ∧ Y6 reviewed ∧ X8 table published ∧ provisional invoice with
`C_strict` ≥ smallest admissible bank. Else `reservation = DEFERRED`, `release = NOT_READY`, scientific labels preserved, **no strict bank
opens this week**, the Mac runs R's unpaced work and quiet-window probes. (B) **Post-L3 gate** (`post-l3.json`): DELIVERY_READY ∧
complete three-oracle PASS incl. replay of every registered attempt; else L4 blocked, downstream hours released. (C) **Pre-strict gate**
(`pre-strict.json`): L4 tuple (`mass_status` CLOSED_BELOW_BAR / BAR_MET / INCONCLUSIVE; `transfer_status` FILTER_MET / TRANSFER_UNRESOLVED;
guards; timing), deterministic sizing from L4 paced variance (historical SDs a labelled sensitivity), `slack = (Tue 23:00 − now) −
remaining serial work − 2 − 6 ≥ 0`, sized bank ≤ `C_strict`; `strict_admission = ADMITTED` only on BAR_MET ∧ FILTER_MET ∧ guards ∧
slack ≥ 0; else BLOCKED (NOT_READY), reservations released, no smaller N. Family {Claim B} freezes at the first strict episode
(comparator-only counts).

**§4 Stage 3 — S paced block (Sat 00:00 → Tue 23:00 if reserved).** L3 (≤ 9 h; INVALID / STOP / INCONCLUSIVE / DELIVERY_READY) → gate B →
L4 (32 h; tuple) → gate C → strict (≤ `C_strict`; candidate paced, bare unpaced; pooled ≥ +40 ∧ LB90 > 0; survival band v2; scripted
no-death ≥ +.03; boost clause as registered; every miss charged) → product (32 paired 10-min sessions ≈ 10.7 h; death-rate paired 95 %
UB < 0; mass relative LB ≥ −5 %) → serving regressions (6 h; identity, hero-only assistance, stale/wrong-life/restart, pins, label,
Watch8/12 + Play isolation, rollback drains and shows resulting identity) → Tue 21:00 seal → 2 h independent packet review → Tue 23:00
disposition. Any scheduler/deadline/fallback repair = new artifact, fresh paced efficacy.

**§5 Stage 3′ — R research track (Wed–Tue, Colab + Mac gaps).** Z-4 → (if admitted) E2 (20 u) → E6/E7 (+E10, 24 u) → E12 (16 u); E3
relaunch → E14; Y3; Z-5. E16/E17/E15 are NOT_READY this week by construction (need E3 architecture + E14 PASS + E12-qualified parent +
E9′). Day-7 deliverable: disposition packet — E9′ ruling, E4 receipt, E2/E6/E7/E12 prints or blockers, E3 print, E14 receipt, Y3 receipt,
E15 envelope ratification, burn manifest, next-week E16 registration *text* (not dispatch). R enters no strict bank this week even if S
defers.

**§6 Stage 4 — day 7 (Tue 23:00) and week-2 handoff.** Per line: tuple (scientific verdict / readiness / release status) via the
reducer; owner front page with exactly one signature text. Handoff index (Z-7): identity, evidence, family (UNOPENED / OPENED), worlds
+ burn manifest, resources (balance, debits, outstanding 2× liabilities, reserve), decisions (verbatim owner answers or PENDING).
Week-2 plan: family registered prospectively from artifacts without a prior strict look (R nominee if E16 → E17 → 32M → 100M → E13
complete; planner-over-R; S if deferred unopened); harvest-aware operator (if O2 D+O > D) as a *new* S identity needing its own L2′-class
evidence; fresh banks; Colab ceiling week 1 ≈ 130 u dispatched + E15 320 u ratified for week 2.

## 3. The open question for Round 10

**Read the skeleton as the person who will execute it, audit it, or sign it. Which sentence of §0–§6 — quoted exactly — would still
let a wrong artifact reach the owner's signature, a right artifact be lost to a clerical rule, or an operator stall at 03:00 with no
lawful next action — and what is the exact PLAN.md text that replaces it, with its kill rule, its receipt path, and its day-7 label?**

Answer from your discipline, not as a committee. Quote the sentence you attack (section, words), give the arithmetic or the code fact
that breaks it (hours, units, SE with the measured SDs 365/263/212, line numbers in the read-only repo), and write the replacement as
text an operator can act on alone at 03:00 with no judge available. Name the published systems or methods you rely on (paper/system +
year). Separate known results from conjecture. Say where Round 9's judgment — its three-admission Stage 2, its `C_strict` ceiling, its
two-cutoff clock fix, its probe-only ask, its Z-4 E9 replacement, its 320-u E15 target, its one-family-per-artifact week-2 rule, or its
power statement — is wrong and why. Changing the game's rules, the opponents, or the benchmark is not beating B8. Keep to the output
format in your task prompt.
