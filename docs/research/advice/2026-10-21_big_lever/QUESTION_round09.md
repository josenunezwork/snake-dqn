# Big-lever research rounds — Round 9 question (plan round 2: final adversarial review of the near-final plan)

Judge: Fable (round-8 judge wrote this, 2026-10-21 ~06:45 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` … `../round-08/JUDGMENT.md` (each round's §(a) and §(c) supersede the previous where they say so), then
`../LIVE_FACTS.md` (measured results Claude adds between rounds; authoritative over everything below) and `../QUEUE.md`. Repo:
`/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`; start with `AGENTS.md`; note it may be
behind main — Round 8 §(top) lists what landed after `57483b2f`: E8 at `research/bl_ppo/model.py:149`, Colab cap 3 measured, E9 with
measured SDs, D0 KILL). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Rounds 1–8's verified
facts and negative record hold with the corrections in the eight judgments; cite them, do not re-derive. **Round 10's judge writes
`PLAN.md` from this round's answers; this round is the last adversarial pass. Attack executable text, not direction.**

## 1. Goal (unchanged) and what Round 8 fixed

A served snake that beats bare B8 s3 by ≥ +40 pooled H5000 mass at equal-or-better survival on the Mac strict gate, then serving
qualification and owner OK. Colab (~1,142 u at 06:20 PT, cap 3 assignments measured, all three occupied by E3/E1b/O1 at writing) for
training/labels/evaluation; serving and gates on this Mac (6 P + 12 E cores, MPS). The owner's W/K clause is **still unanswered**:
keep both branches (W ≤ 4 / K ≤ 8 ⇒ S_p99 ≤ 1.458 s at W = 4; silence or "no" ⇒ W 1 / K 1 ⇒ submit→consumed ≤ 333 ms at 2.4 roots/s
or 542 ms at 1.6 roots/s).

Round 8 established (`../round-08/JUDGMENT.md` §a, §c):
- **One release track per week.** S's first strict bank would open Sun Oct 25 17:00 and freeze the family; R cannot nominate before
  Tue (100M endpoints earliest Mon–Tue). So **S is the conditional release track; R is the research track** whose day-7 deliverable
  is decisions + an executable funded E15 registration + a durability receipt + an E13-ready export — NOT_READY for release by design.
  No bounded wait for R; if S is deferred Fri 23:59, **no strict bank opens this week**. No 4M/32M checkpoint enters strict.
- **Serial Mac ledger:** Sat 00:00 → Tue 23:00 = 95 h; S alone 80–95 h; the 37-h strict case is refused; admission needs
  `slack = exclusive hours − serial remaining − 2 h packet − 6 h recovery ≥ 0` at the L4-variance-sized strict bank.
- **L2′:** four immutable 40-world chunks (16 frozen / 16 mixed / 8 scripted each), timeout ≤ 480 min, `timeout ≥ 2 × estimate`,
  aggregate **R ≥ 2.18 batches/s** on X1's fixtures, one look after all four, chunk timeout ⇒ study INCONCLUSIVE; PACKAGE_KILL /
  FUND_SYSTEMS / INCONCLUSIVE labels unchanged. With measured SDs (365/263/212) pooled SE ≈ 25.7: FUND_SYSTEMS ≈ 20–30 % at a true
  +40, ≈ 80 % at a true +80 — powered for a T3-sized effect only. Latest single-session dispatch for a Fri print was Thu 22:23;
  chunks relax this to "as slots free".
- **X1 does not price Mac S.** One shared fixture/operator harness, separate platform receipts; L1′ runs on the Mac **Wed evening**
  (Phase 0 ends Wed; Round 0 owes no replay), all W cells recorded, authorised cells per the clause; E14 bare cells Thu (3 h each).
- **Stage 0 = dispatch readiness** (Thu 12:00 code cutoff, Thu 20:00 reviewed manifests), not "everything lands". Owners: Sol-S
  (Y0/Y1/Y7/X8), Sol-R (X3/Y3/Y4/Y8/E2 admission), Sol-C (X7 recorder + L3 controller + Y5), Codex (Y6/Y9/Y10/replay reducer/audits).
  X7 + L3 are due before L3 (Sat 00:00 earliest), not Thu.
- **Trust moved into Stage 0:** Y3 exact-continuation durability (≤ 8 u) before any 32M pause; Y5 owner packet / Claim B pin schema /
  "Assisted hero — [controller ID]" label / rollback matrix before L3; Y6 family state machine ("opens" = first strict evaluative
  episode incl. comparator-only; late nominees excluded; frozen-unavailable p = 1) + pipeline OC on measured SDs; Y7 world-level
  ledger + L4 **TRANSFER_UNRESOLVED** (strict admission needs frozen point > 0 ∧ mixed point > 0 paced, no per-mix UB90 < 0; budget
  status, not a scientific label); Y4 E17 → 40 u / 20 h; Y8 E15 → 192 u chunked at Y3 pause points, E13 → 18 u, 128/mix fixed.
- **E16 executes A/G only** (`bl_e16/admission.py:13,131,195-227`) and requires E3 + E14 receipts: no admitted G ⇒ no E16/E17/E15
  dispatch this week; E2/E6/E7/E12 decisions stand alone; any A-only study is a new registration.
- Day 7 (Tue Oct 27 23:00) per line: qualified / scientific KILL / INCONCLUSIVE / NOT_READY; day-3 Fri Oct 23 23:59 (admission
  record Y10); day-5 Sun 08:00 = integrity + hours only. A100/H100: none (13 rounds unanimous).
- Shortlist: served search 28/16; representation 28/16; PPO backbone 22/9; engine 30 %; search-in-the-loop 15/10; archive 6;
  opponent 4; world model / league 2.

## 2. The near-final plan (skeleton of PLAN.md; Pacific time; Wed Oct 21 … Tue Oct 27, with a week-2 handoff)

**Stage 0 — Dispatch readiness (Wed–Thu 20:00).** Sol-S: Y0 (chunked L2′ amendment) → Y1 (shared harness; L4 R receipt ≤ 4 u) → Y7
(ledger fields) → parity audit (4 u) → L2′ chunks dispatched as slots free (two in parallel when two are free). Sol-R: X3 (E16 cap) →
Y4 (E17 cap) → Y3 (durability, ≤ 8 u) → Y8 (E15/E13 repricing) → E4/E2 admission under frozen E9. Sol-C: X7 events frozen Wed, recorder
+ L3 controller against a fake clock, Y5 packet/pin/label/rollback schema. Codex: Y6 state machine + OC, Y9 owner table, Y10 admission
template, independent replay reducer, all audits. Mac: Phase 0 confirmed idle → Y2-A L1′ (Wed eve, 3 h, all cells) → X5 strict-runner
receipt (2 h) → Y2-B E14 bare cells (Thu, 3 h). Kill: an item not reviewed by Thu 20:00 is NOT_READY; its dependents do not dispatch.

**Stage 1 — Measure (Thu–Fri).** L2′ chunks on Colab; E2 (20 u) when a slot frees; E3 print (running) → E14 receipt; E6/E7 (24 u,
after E10 and E2 endpoints); E12 (16 u). L0′ window table published (sets 1.6 vs 2.4 roots/s). Kill rules as registered.

**Stage 2 — Fri 23:59 admission record (Y10).** S holds the paced block Sat 00:00 → Tue 23:00 **only if** L2′ FUND_SYSTEMS (all four
chunks terminal) ∧ authorised L1′ cell PASS (clause branch as answered; silence ⇒ W 1 / K 1) ∧ X7 three oracles PASS ∧ Y5 schema frozen
∧ slack ≥ 0 at the L4-sized strict bank (37-h case refused). Otherwise S is deferred (status preserved), no strict bank opens this week,
and the Mac runs R's unpaced work (E2 screen, X5, E13 Mac half) plus any quiet-window probes.

**Stage 3 — S paced block (Sat–Tue, if admitted).** L3 (9 h; DELIVERY_READY / INVALID / STOP / INCONCLUSIVE) → L4 (32 h; UB90 < 40 ⇒
closed; TRANSFER_UNRESOLVED ⇒ reservation released) → Claim B strict (candidate paced, bare unpaced; bank sized from L4 variance
with measured SDs; family frozen at the first strict episode = {Claim B}) → product sessions (32 paired 10-min) → serving regressions →
owner packet. Monday product failure ⇒ SKILL_PASS / PRODUCT_FAIL / RELEASE_HOLD; a scheduler/deadline/fallback repair is a new artifact.

**Stage 3′ — R research track (Fri–Tue, Colab).** E16 (40 u; only with E3 architecture + E14 PASS + E12-qualified F parent) → E17 (40 u)
→ 2 × 32M futility look (after Y3) → declared-model 100M decision (UNCALIBRATED ⇒ balanced endpoints under the 192-u envelope; pause at
the exact checkpoint if unfundable) → E13 receipts on the final export. Deliverable Tue: decisions printed, E15 registration executable
and funded, durability receipt, E13-ready export, burn manifest. No strict entry this week.

**Stage 4 — Day 7 (Tue 23:00) and week-2 handoff.** Per line: qualified (awaiting owner OK) / scientific KILL / INCONCLUSIVE (next look
priced) / NOT_READY (measured remaining work). Handoff: frozen identities and hashes, family state, burned-world manifest, units spent /
remaining (~1,142 u start), open owner decisions (W/K clause; any release), and the week-2 plan under which both lines may enter one
family. Colab ceiling this week ≈ 220 u + E15's 192 u; A100/H100 none.

## 3. The open question for Round 9

**Which single clause of the near-final plan above would you refuse to sign as written — because at the moment it fires an operator
could not execute it (wrong instrument, wrong timestamp, missing owner, missing receipt path, ambiguous label, undefined joint
outcome) or because it would let a development print, a Colab receipt or a repaired artifact pass as release evidence — and what is
the exact replacement text PLAN.md should carry for that clause, with its kill rule and release behaviour?**

Answer from your discipline, not as a committee. Quote the clause you attack (stage, sentence), give the arithmetic or the code fact
that breaks it (hours, units, builder-days, SE with the measured SDs 365/263/212, line numbers), and write the replacement as text an
operator can act on alone at 03:00 with no judge available. Name the published systems or methods you rely on (paper/system + year).
Separate known results from conjecture. Say where Round 8's judgment — its single-release-track ruling, its chunked L2′, its 2.18/s bar,
its weakened TRANSFER_UNRESOLVED clause, its "no strict bank this week if S is deferred" rule, its owner table, or its day-7 and week-2
text — is wrong and why. Changing the game's rules, the opponents, or the benchmark is not beating B8. Keep to the output format in your
task prompt.
