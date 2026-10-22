RULING: Fill the idle L4 slot with DENSITY-2 — a pre-sized natural-arm-A replication of the round-14 density test (A 96/72/72, X arms 24/36/24, fresh seed namespace, pilot excluded from the statistic, ≤ 5 u / ≤ 180 min, one look, no third enlargement); the P1 lane (canary → Y3 source-latebody-03 → 1 M) continues in its own slot unchanged; nothing else is admitted to fill Colab — O2 / E6 / E7 / E2 / Round 0 stay parked, and an idle slot is lawful once DENSITY-2 is terminal.

# Spare-slot judgment — what fills idle Colab capacity this week

Judge: independent Fable (spare-slot judge), 2026-10-22 08:20 PT. Inputs read in full: `LIVE_FACTS.md` (tail, newest
07:50 PT), `QUEUE.md` (tail), `round-14/JUDGMENT.md`, `exp/bl-l2/JUDGMENT_MAC_AMENDMENT.md`, the density pilot
`exp/bl-density/round14/production-l4-output-02/exports/density/summary.json` (every cell), one episode export, the
production manifest and run log, `research/bl_density/{runner,manifests}.py` (read-only, main `383fa395`), P1
`qual-1m/prepare-latebody-02/DISPATCH.txt`, `exp/bl-l2/native-l4-prepared-20261022-02` listing. Sealed study not opened.
No jobs, no Colab calls, no commits.

## 0. Facts the ruling rests on (verified this morning)

- **Pilot (run-02, COMPLETE, 90 episodes):** 4,532 CPU-s total, ≈ 1,100 s supervisor wall on 6 workers (≈ 4.1 effective
  cores), well under 1 u. Arm A: 36 episodes → doom-bearing episodes frozen 1 / mixed 2 / scripted 2 (0.139 per episode
  pooled; 0.083 frozen); witnessed-avoidable doom roots 0 / 1 / 1. Every X×mix cell printed
  `INSUFFICIENT: A_fewer_than_4_doom_bearing_episodes`; every realism classifier `INSUFFICIENT` (< 4 episodes per class);
  no AUC exists. Directional per-CPU-second yields (C-P1 ≈ 4.8× A, B70m ≈ 3.7× A) are **pilot observations, not
  verdicts** — the pre-registered rule never fired.
- **Per-episode CPU (attributed, L4 host, stock Python hook; `s2_native_v8: null`):** A 67.7 / 48.6 / 62.5 s
  (frozen / mixed / scripted); B70m 42.6 / 44.2 / 33.8; B70p 43.1 / 56.9 / 20.8; C-P1 50.0 / 57.6 / 48.9.
- **Free by-products already in the pilot:** ρ95 natural total-occupied fraction frozen .122 / mixed .105 / scripted .122
  (n = 85 / 51 / 94 roots — small); arm-A cheap-prefilter recall **0.6 (6 of 10 true dooms captured)**; doom roots are
  27/29 `DEATH_SELF` at L 801–2,094 (median 1,186).
- **The production allocation is frozen in code, not in the manifest:** `runner.py` refuses anything but
  `--episodes-per-cell 6` (A = 2×), `--workers 6`, and seeds derive from the constant `NAMESPACE =
  "bl-density-round14-20261022"` by `(arm, mix, index)` — so a plain re-run with more episodes would **reuse the pilot's
  seeds for indices 0–11 / 0–5**. A replication therefore needs a small Sol change; it cannot start this minute.
- **P1 lane:** `qual-1m/prepare-latebody-02` is PREPARED_NOT_DISPATCHED behind "one P1 L4 at a time"; the running
  P1 canary (`y3/source-latebody-03-canary-01`) owns that lane. The L4 native+exact-stop ruler owns the third slot;
  the Mac amendment (§5) already ruled its print is not needed for L2′ this week — let it finish, do not restart it.
- **Nothing else judged is dispatchable:** O2 v3 refuses its own timing forecast until rewired on native+exact-stop and the
  round-14 shortlist places it after an L2′ FUND; E6/E7/E10 are blocked on E2's endpoints and E2 on E9 (joint funding
  .27–.31 < .60); Round 0 is CLOSED unless a lane commits to consume it (none does); D0 KILL; E1b STOP; E3 KILL.

## 1. Ranked jobs (at most three)

### 1 — DENSITY-2: natural-arm-A replication at pre-fixed N (idle slot; the decision-producing job)

**What it decides.** For each X ∈ {B70m, B70p, C-P1} × mix, the pre-registered round-14 §2b verdict
(REJECTED / ADMITTED_FOR_A_TRANSFER_TEST / INSUFFICIENT) that the pilot could not reach; the natural
witnessed-avoidable-doom rate and ρ95 at ≈ 2,000 natural late roots (sizes any week-2 restart archive, row 9); and
the cheap-prefilter recall at ≈ 30 true dooms (pilot 0.6 — if it stays < 0.8 the prefilter cannot be the trigger for
any critic or served use, which is itself a design-closing fact).

**Sizing (from the pilot rates; sizing is not evidence).** Floor = ≥ 4 doom-bearing A episodes per mix. A frozen 96
episodes (λ ≈ 8 at 0.083/episode ⇒ P(≥ 4) ≈ .96; still ≈ .57 if the true rate is half the pilot's), A mixed 72 and
scripted 72 (λ ≈ 12). X arms must also reach ≥ 4 doom-bearing episodes for their realism classifier: B70m 24/mix
(pilot 5/18 ⇒ λ ≈ 6.7), **B70p 36/mix** (the owner's "smaller + more opponents" arm; pilot 3/18 ⇒ λ ≈ 6), C-P1 24/mix
(pilot 11/18 ⇒ λ ≈ 15). Total **492 episodes**: CPU ≈ 14,500 (A) + 2,900 + 4,350 + 3,750 ≈ **25,500 CPU-s ≈ 7.1 CPU-h**.
Wall at the pilot's 6-worker efficiency ≈ 105 min; with 10 workers on the 12-vCPU host ≈ 70–80 min (unmeasured —
the 180-min cap absorbs a 1.7× miss). **Units ≤ 5 u incl. 20 % reserve** (1.54 u/h × 3 h = 4.6 u); first cap wins;
INCOMPLETE retains per-episode units and is **not** extended.

**Pre-registered rule (unchanged thresholds).** Per X × mix, only where new-A has ≥ 4 doom-bearing episodes: REJECTED
if witnessed-avoidable yield per total CPU-second < 2× A, or avoidable fraction < ½ A's, or AUC ≥ .80; else
ADMITTED_FOR_A_TRANSFER_TEST (admits only the registration of a separate N-vs-D transfer screen with 09's falsifier —
never a P1 change, never a training arm). A realism classifier that is still INSUFFICIENT (< 4 episodes per class on
either side) ⇒ the cell is INSUFFICIENT_REALISM (may be REJECTED on yield, can never be ADMITTED). Pooled-over-mix A
avoidable rate and ρ95 are secondary descriptives, reported, not decided on.

**Kill / stop rules.** (i) Mechanical (transport, CUDA, export, hash) failure ⇒ one new attempt from a new prepared
manifest, never a resumed one. (ii) If new-A still has < 4 doom-bearing episodes in a mix after the full allocation,
that mix is INSUFFICIENT and **row 9/9a (densification) is CLOSED for that mix this campaign — no third enlargement.**
(iii) Pilot and replication are never pooled.

**How it avoids the peeking problem (summary.json has been read).**
1. The pilot's 90 episodes are used **only to size N**; they are excluded from every statistic, classifier and
   decision of DENSITY-2 (new `NAMESPACE`, fresh seeds, collision-checked against the pilot's 90 seeds).
2. All four arms × three mixes stay; none dropped, none added, no weight change. The pilot's favourable-looking X
   yields do not alter the allocation beyond the per-cell ≥ 4-episode floor, which is derived from the *rates* alone
   and written here before any new data exist.
3. Rule, thresholds, features, prefilter, stride, horizon, B8 bytes: byte-identical (same METRICS / amendment SHAs);
   only `episodes_per_cell` by arm, `NAMESPACE`, `workers`, `max-seconds` and manifest caps change.
4. One look, at terminal state; no interim reads of exports; no extension on INSUFFICIENT.
5. This file is the pre-data registration; Sol's `METRICS_AMENDMENT_04.md` quotes the pilot numbers so a later reader
   sees N was fixed on rates, not on outcomes.

**Package / Sol change (≤ 0.25–0.5 Sol-day, then Codex-V review).** `research/bl_density`: a second frozen production
allocation (A 96/72/72 by mix; B70m 24; B70p 36; C-P1 24), new `NAMESPACE = "bl-density-round14b-<date>"` with a test that
asserts zero seed overlap with the pilot namespace, workers ≤ 10 allowed in production mode, `--max-seconds 10800`,
manifest `unit_cap 5.0 / reserve 1.0 / work_wall_seconds 10800 / session_minutes 180 / min_vcpu 12`, `forecast.py` and
tests updated, `METRICS_AMENDMENT_04.md` sibling (pre-data). Round-12 §4.1 still applies: the 12-episode tiny canary
through the real supervisor first (≤ 0.5 u, ≤ 20 min; PASS = exports recovered + hash-verified + per-stage CPU counters),
then production. **Cannot start immediately** — earliest dispatch is this afternoon PT; until then the slot stays idle
lawfully (an unjudged filler is worse than an idle hour).

### 2 — P1 lane continuation (already judged; its own slot; no new approval)

Canary terminal receipt ⇒ `y3/source-latebody-03` under round-14 (c)1–6 (≤ 6.4 u remaining of the 8-u Y3 cap) ⇒ on
Y3 PASS, `qual-1m/prepare-latebody-02` (est 40 min, timeout 80 min, cap 4.0 u, reserve 0.8 u). Kill rules as registered:
INSUFFICIENT_LATE_BODY_SUPPORT after the one bounded continuation ⇒ NOT_READY; numeric gate failure ⇒ NOT_READY with
rows exported ⇒ run (c)6 on the Mac (0 u). "One P1 L4 at a time" stands — this lane never takes the idle slot. Can start
the moment the canary's receipt is verified (operator = Claude).

### 3 — Conditional, after DENSITY-2 prints: natural late-root bank by x86 replay (≤ 1 u, ≤ 30 min L4)

Only if DENSITY-2's new-A reaches the floor in ≥ 2 mixes. Each episode export already carries `(seed, plan, decision
index, sim_config, mechanics)`, so natural L ≥ 800 roots are reconstructible by deterministic replay **on the same
x86/L4 platform** (Mac↔x86 float drift forbids cross-host reconstruction — E3 recovery-03). Sol ≤ 0.5 day: replay and
export full snapshots for every confirmed doom root and witnessed-avoidable root plus 1-in-10 of the other eligible A
roots (≈ 250 snapshots, ≈ 25 MB, under the 64-MB export cap). Label: DEVELOPMENT bank, unregistered for any gate; its
only consumers are the round-14 (c)6 critic diagnostic's cross-check and the week-2 archive realism check (row 9,
archive ≤ 12 u), each of which needs its own registration before it reads the bank. It produces an input, not a
verdict — hence ranked last and conditional.

## 2. Explicitly refused as slot-fillers this week

A larger pilot-pooled "extension" of run-02 (peeking); B50 or any map-shrunk / extra-opponent *training* arm; O2 v3
before its native+exact-stop rewiring and an L2′ print; E6/E7/E10 before a new E2 design passes E9; a speculative
Round 0 re-registration; a second L4 native ruler; A100/H100; anything on the Mac during Sol's L2′ wiring or the
approved ~14-h L2′ run (serving, L2′ and strict gates stay on this Mac — Colab is for training/labels/evaluation only).

## 3. Honest limits

Sizing uses Poisson arithmetic on 5 pilot events; a true frozen rate half the pilot's leaves P(≥ 4) ≈ .57 — the no-third-
enlargement rule is what keeps that from becoming an open-ended spend. Worker scaling beyond 6 on the 12-vCPU host is
unmeasured (E4 saw no CPU scaling 6 → 12 for PPO; the density runner is a different workload). Nothing here is efficacy,
a served claim or a release; B8 s3 stays the served hero.
