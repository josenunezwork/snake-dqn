# Round 15 — from an offline FUND to a served, strict-gate-qualified snake on THIS Mac (2026-10-22 ~15:45 PT)

Written by the round-15 Fable judge from `../CHARTER.md`, `../PLAN.md`, `../PLAN_v2.md`, `../round-11..14/JUDGMENT.md`,
`../LIVE_FACTS.md` (newest 15:35 PT, authoritative), `../QUEUE.md`, the L2′ analysis, the K-1 and native-ruler receipts,
`RESULT_L1.md`, the two L2′ amendment judgments, and `SUCCESS_CONTRACT.md` (repo `b416719e`, read-only). Every number below
names its receipt; verify in the repo/artifacts before relying on it. Never open `snake-dqn-artifacts/t34-round0-study-01/`.

## Goal (unchanged, charter)
A served snake on THIS Mac (Apple M5 Pro, 18 cores, no CUDA) that clearly beats bare B8 s3 (`3e649783…`) as the highlighted
hero of the **mixed Watch6** roster, with a claim the owner can trust: Mac strict gate (SC §2 skill bars on fresh paired banks)
∧ product PASS (SC §3, continuous-respawn Watch6 vs FRP3 s12+v8 opponents) ∧ serving qualification (SC §4 + the owner's §4
amendment: up to **W = 4 worker processes / K = 8 outstanding requests on this Mac**, 24 fps, zero overruns, update p95
≤ 11.688 ms, deadline miss → bare action) ∧ owner release OK. Serving and gates stay on this Mac (owner 10-21 00:30). Colab
(3 slots, L4 only, ~1,100 u, "spend them") is for training / labels / evaluation only; nothing is GPU-bound; A100/H100 are out.

## Owner rules for this round
Budgets 2× generous; never idle (Mac, slots, builders); resume-by-default on mechanical failure; **advisors before design**
(a judge rules before Sol builds anything that changes an identity, estimand or environment); release needs owner OK.
Everything else is pre-delegated. B8 s3 stays served until a same-digest packet qualifies.

## What landed today (facts, with receipts)

**F1 — L2′ printed FUND_SYSTEMS** (`docs/research/results/L2prime_mac_analysis_20261022.json`; commit `b416719e`;
`exp/bl-l2/mac-amendment-01/analysis.json`). Registered 64/64/32 fresh paired bank, H5000 terminal hero, dense T3 overlay vs
bare B8 s3, one look. Equal-mix pooled mass **+83.0 (LB90 +64.1, UB90 +101.9)**; frozen **+56.1 (LB90 +34.8)**; mixed
**+27.6 (LB90 +12.1)**; scripted +165.3 (LB90 +114.8). No-death episode rate **+19.8 pp pooled** (frozen +20.3, mixed +7.8,
scripted +31.3); alive-frame fraction +7.5 pp. **The gain is survival, not productivity:** `mass_ledger.dense_only_survival_mass`
is 3,585.5 of the 3,591.9 frozen paired delta, 1,765.4 of 1,766.4 mixed, 5,284.5 of 5,288.6 scripted; common-alive
productivity delta ≈ 0. World ledger: productive rescues 13 / 5 / 10 (of 64 / 64 / 32 worlds), re-deaths 3 / 2 / 6, D300-only
escapes 3 / 2 / 6, newly caused deaths 0, baseline deaths 23 / 11 / 17 (sealer strata all "unresolved"). **Action changes:
54 / 21 / 79 out of 295,355 / 304,693 / 153,048 decisions = 0.18 / 0.07 / 0.52 per 1,000**; of those, delay-only 26 / 7 / 47 and
horizon-surviving 28 / 14 / 32. Interventions "proposed = completed" 30,534 / 30,159 / 19,037 (every dense batch, unpaced);
`consumed_before_deadline: UNAVAILABLE_UNPACED_L2`. `development_only: true`; `prospective_l4_transfer: TRANSFER_READY`
(explicitly "not authoritative").
Execution (`chunk-00/operational.json`, AMENDMENT_MAC_L2.md): this Mac, **W2 one math thread, nice 10 / caffeinate, mode A
`exact_stop`, `native_mode = stock`**, `DenseTrigger(600, 5)` (every 5th decision at L ≥ 600), ≤ 6 legal H300 branches,
`_choose` ranking (survival/delay, bare preference, Q desc, index), counters validated, lease-bound.
**Opponents: the historical T3 rosters with NO opponent veto** (frozen = Apex checkpoints champion_a5_freespace /
best_apex_fs / best_apex_stage1_fs / best_apex_pre_fs × 5; mixed = three of those + 2 `random_safe`; scripted = `greedy_food`
× 5; `veto_slots=()` at `eval_engine.py:910`) — see `exp/bl-l2/JUDGMENT_OPPONENT_PREMISE.md`. Boost share of the overlay is
not reported in the analysis (SC §2 requires [0.5 %, 10 %] at strict).

**F2 — The served game is a different game.** Watch6 mixed = bare B8 s3 at slot 0 vs **5 × FRP3 s12, each wrapped with
the exact v8 veto (λ = 8)** (`web/backend/session.py:622-705`, `safety_veto_serving.py:508-527`; `docs/STATUS.md:29-42`);
continuous respawn; product bars are paired deaths/1k (95 % interval excluding zero, target ≥ 20 % fewer) and full-horizon
mass relative LB95 ≥ −5 % (SC §3). No FRP3 s12 appears anywhere in BANK_L2. **No exact-search value against vetoed
FRP3+v8 opponents has ever been measured**; L2′'s print says nothing about SC §3 (opponent-premise judgment §2, "honest
limit"). The strict gate (SC §2) uses the T3/strict roster family — the game L2′ did play — on *fresh* banks.

**F3 — Speed on this Mac, measured (fixtures, not production).**
- K-1 (`exp/bl-k1/timing.json`): `K1_PASS`, native kernel vs stock opponent-veto hook **20.13×** on frozen-1500 (810 batches,
  warm-29 medians, W1, one thread, `foreign_load_observed: true`, `serving_or_release: NOT_GRANTED`, `wp4: ADMITTED_BUILD_ONLY`).
- Native + exact-stop paired ruler (`exp/bl-l2/y1-native-mac/ruler-receipt.json`, `PAIRED_PASS`, identity 270/270, quiet host,
  **W1 one thread, kernel mode, FRP3 s12 + v8 opponents on slots 1–5** = the Watch6 pricing case): all-branches aggregate
  **0.611 batches/s**; exact-stop **1.793 batches/s**. Exact-stop warm medians / p99 per 6×H300 batch: frozen-1200 0.63 / 0.78 s,
  frozen-1500 0.68 / 0.84 s, **frozen-800 (the only doom fixture, bare dies 30/30) 1.07 / 1.27 s** (2 branches evaluated), mixed
  0.50–0.54 / 0.62–0.69 s, scripted 0.29–0.30 / 0.34–0.38 s. All-branches: frozen 1.8–2.5 / 2.1–3.1 s, mixed 1.6–2.0 / 2.0–2.5 s.
- Contract ceilings (RESULT_L1.md; PLAN §1): W1/K1 valid-consumption **C p99 ≤ 333 ms** (2.4 roots/s) or 542 ms (1.6/s, only if
  X8 publishes it); expanded cells **S p99 ≤ 0.830 s (W2) / 1.250 s (W3) / 1.458 s (W4, K8 tail cap)**; coverage ≥ 29 valid
  consumptions per 12 s (≥ 20 at the 1.6 sensitivity); zero overruns; Watch6 update p95 ≤ 11.688 ms. **On paper** one
  exact-stop root fits under the W3/W4 caps (even the doom fixture's p99 1.27 s) and misses the W1/K1 ceilings. **Projected,
  not measured:** these are single-root fixture micro-timings under an exclusive lease with no game running; the Y1 fixtures are
  constructed serpentine roots (not natural histories); W > 1 concurrency, K > 1 queues, cold starts, the 24-fps game sharing
  the cores, Spotlight/analyticsd load and the clock seam were never timed with the native path.
- The only end-to-end latency receipt (`exp/bl-l1-20261021/RESULT_L1.md`, 10-21) used the **stock** hook: S p50 19.7–23.7 s
  (CPU, W1–W4), MPS 32–38 s, **every cell had in-burst overruns (280–390) and update p95 37–48 ms**, zero eligible consumptions,
  `NOT_READY(INSTRUMENT_MISSING)` — "the requested production reachable-root provider is absent". A formal L1′ on the native +
  exact-stop path has not been run. The earlier Path B v2 dense wrapper smoke (10-20) delivered 0/34 rescues; doom-to-last-
  avoidable-decision leads were 224–433 frames (≈ 9–18 s).
- Dense arithmetic: at 24 fps, "every 5th decision" is 4.8 roots/s per hero; bare dies in ≈ 1.19 % of dense batches on the
  T3-dense distribution (440 / 36,996; round-13 §0), and L2′ changed 0.07–0.52 actions per 1,000 decisions — the value is in a
  few rescues per world, each worth ≈ 334 mass at census medians (round-14 §2d).

**F4 — What a release claim requires (PLAN §0, unchanged; SC §1–§4).** One controller digest C (weights, planner, root rule,
checkpoints, lead cap, both cutoffs, W/K, device, threads, recorder, fallback, load envelope) bound through: L1″ latency PASS
with independent audit → L3 paced smoke (9 Mac-h) → L4 paced dev efficacy (32 Mac-h; 32 fresh worlds/mix, candidate and B8
each twice at absolute deadlines, every miss charged) → strict on a fresh collision-audited bank (candidate **paced**, bare
unpaced; pooled ≥ +40 ∧ LB90 > 0, survival band v2, scripted no-death ≥ +0.03, boost clause; family {Claim B} frozen before the
first strict episode) → product (32 paired 10-min Watch6 sessions, ≈ 11 h) → serving regressions + rollback matrix (6 h) →
sealed packet + independent review → owner signature text A. ≈ 80 serial Mac-hours after the controller exists; PLAN_v2 §5
priced "L2′ FUND ∧ K-1 PASS ⇒ served sparse-burst identity program, earliest qualification **Nov 3–7**, probability of a
+40-at-bar qualified release ≈ 10–15 %". Power (PLAN §0): L4 paced 32/mix SE ≈ 29; strict 128/mix SE ≈ 14.7; three stacked
point filters pass ≈ 12.5 % at a uniform true +40, ≈ 60–65 % at +80. L2′'s mixed +27.6 is below the +40 point bar on its own;
the bar is equal-mix pooled.

**F5 — Identities that exist on paper or in code, and their evidence status.** (a) The **dense T3 certificate controller**
(what L2′ measured): exact, unpaced, 4.8 roots/s, every batch consumed — never served, never paced. (b) **WP-4 sparse
future-root burst** (PLAN_v2 §1, round-11 §c4): production root provider targeting a reachable future root ≥ 144 frames
ahead from the exact bare shadow; bare + two Q-ordered alternatives; resumable H32→96→192→300; first alternative surviving
H300 published when bare dies; W = 4 / K = 4; visibility timestamp + publication/consumption cutoffs (Z-2′ clock seam,
PLAN §2); **build was suspended by round 12 pending L2′'s print (now printed); K-1 marks it ADMITTED_BUILD_ONLY; it is a new
identity with no inherited T3/L2′/K-1 evidence and its fixture bars (first valid H300 certificate ≤ 5 s on ≥ 50 % of
dense-rescuable frozen/mixed opportunities, before each root's own deadline) were registered pre-data in PLAN_v2 §3.**
(c) X7 paced-identity package (built, not qualified; `l3_identity_prerequisites` unmet); Z-2′ clock seam fixture: synthetic
proof only (`clock-preflight-01/clock.json`), no production acceptance. (d) V0 latency ladder (round 13) and the tail-critic
V1: V0 not run; learned components are research only until a served exact identity exists.

**F6 — Everything else, in one line each.** P1 (100M PPO) is in Y3 recovery (`source-latebody-05` prepared on the requalified
warm-up transport; no training yet). DENSITY-2 terminal twice on supervisor telemetry failures (fix requalified). O2 parked.
X5 bare strict timing still NOT_READY. Process rules in force: premise register (measure before plan), target-platform canary,
execution attestation by counters, Mac exclusive lease (`exp/mac-lease.json`), judge ruling before environment/identity-changing
builds, pre-claim published-ledger check, no timing transfer Mac↔L4, no partial looks, no retries of bank chunks.

## Proven vs not (read this before answering)
Proven (receipt-backed): exact-sim dense search has frozen/mixed offline value against the T3 roster family (F1); the value
is survival via rare rescues (F1); the native veto kernel is action-identical and 20× faster on the hook (F3); exact stop is
decision-identical and ≈ 2.9× on batches (F3); one exact-stop root on constructed fixtures with vetoed FRP3+v8 opponents
takes 0.5–1.3 s at W1 on a quiet Mac (F3); the owner's W ≤ 4 / K ≤ 8 clause is in force (SC §4 amendment).
Not proven (no receipt): any search value against the **served** vetoed FRP3+v8 roster or in continuous-respawn play (F2);
any paced/served delivery of a rescue at 24 fps on this Mac with the native path (F3; the only paced receipt is the stock-hook
FAIL); W > 1 / K > 1 behaviour, overrun-free ticks and update p95 with search workers alive; the production root provider and
consumption path (absent per RESULT_L1); that the sparse-burst identity preserves L2′'s rescues (its deadlines, lead windows and
root rule are different from "every batch consumed"); the boost share of a search controller; the strict-gate power to clear
+40 pooled when mixed sits at +28; that ≈ 80 serial Mac-hours fit before the owner's patience, with the Mac also wanted for
X5, V0 and P1 probes.

## The open question
**What is the fastest credible path from this FUND to a qualified served controller on this Mac and a strict-gate verdict —
and what must be measured first?** Answer with the identity you would freeze, the one measurement (platform, code path,
receipt, bar, kill rule, ≤ units / Mac-hours) that must precede any build, the sequence of receipts to a strict verdict with
honest calendar and probability, what evidence from L2′ / K-1 / the native ruler may lawfully carry over and what may not,
and what you would do with the Colab slots and the Mac so nothing idles. Say what would make you stop.
