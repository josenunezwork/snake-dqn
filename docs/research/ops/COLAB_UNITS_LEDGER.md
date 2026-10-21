# Colab compute-units ledger

_As of **2026-10-20 19:18 PT**. Read-only snapshot: `colab usage` / `colab sessions` (CLI 0.7.4) plus supervisor
receipts. No runtime was created, assigned, stopped or exec'd to build this ledger._

## Headline

| Quantity | Value | Source |
|---|---|---|
| Live balance | **1,153.84 u** | `colab usage` 19:18 PT (rate 1.54 u/h, 1 active assignment = Round 0 chunk 000) |
| Spent since CLI era began (10-18 19:41 PT, 1,174.37 u) | **20.53 u** | balance delta |
| Spent since the stale 1,240 u starting balance (includes Q0/Q0b Drive-queue work, 10-17 eve to 10-18) | **~86 u** | 1,240 − 1,153.84; Q0 period ≈ 65 u (Q1 memo) |
| Days to owner goal (balance → ~0 by ~2026-11-20) | ~30.2 days | |
| **Burn needed from now** | **~38.2 u/day** (~1.6 u/h around the clock; ≈ 1 L4 running 24/7 at the observed 1.54 u/h) | |
| Actual burn, CLI era | **~10.4 u/day** (20.53 u / 47.6 h); 10-19 = 13.1 u, 10-20 = 7.4 u so far | |

Pricing notes: L4 receipts consistently observe **1.54 u/h** (planning price 2.0 u/h). The Colab CPU runtime showed
no measurable debit (≤ 0.01 u per job). Q0 A100 work was priced at 5.3 u/h.

## Ledger: every Colab job since 2026-10-18

Times are PT (start = receipt mtime − receipt `elapsed_seconds`). "Units" = account-balance debit across the job
(`usage_before` → `usage_after` in the receipt; provider balance updates in ~0.01–0.1 u steps and lags, so short jobs
can read 0.00). "Own" = supervisor-integrated own-elapsed units. Receipts live under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/` at the path given per block.

### Q0 / Q0b / Q1 (legacy Drive queue + browser notebooks, 10-17 evening to 10-18 19:00 PT)

| When (PT) | Job | Runtime | Wall | Units | Decision |
|---|---|---|---|---|---|
| 10-18 00:33 – 14:04 | Q0 T23 mass audit (`q0-audit-t23-mass*`, 7 attempts incl. 354 min + 242 min A100 runs) | A100 (+1 CPU) | ~12 GPU-h claimed | claimed 62.6 (at 5.3 u/h) | T23 loss = foraging-rate drop by 2M |
| 10-18 14:02 – 14:41 | Q0 B8-init audit (`q0-audit-b8-init-a100r5/r6`) | A100 | 9 min | 0.79 | B8-init fidelity COMPLETE |
| 10-18 14:05 – 17:56 | Q0b capacity packets (A100 r5/r6, L4 r6) | A100, L4 | 86 min | 5.2 | both packets qualified (A100 N=3 ~13.5k, L4 ~7.5k decisions/s) |
| 10-18 19:00 | **Q1** (B8 2×2 factorial) | — | — | **0 (never dispatched)** | REFUSED under A5: upper-bound accounting 363 u (idle runtimes). INCOMPLETE (budget), not a result |
| **Q0 period total** | | | | **~65 u real balance drop** (claims 68.6 u; policy upper bound 317 u) | idle allocations were the dominant cost |

Source: `q1-a5-20261018/allocation-accounting-v2.json` (12 historical attempts, `historical_claim_totals.q0_units`
= 68.63). Real drop per `memory/q1-a5-refused` and the 1,174.9 u balance quoted in
`docs/research/advice/2026-10-18_q1_next_*.md`.

### Colab CLI supervisor era (10-18 19:41 PT onward, start balance 1,174.37 u)

| Start (PT) | Job id | Runtime | Wall | Units (Δbal) | Own | Status | Decision |
|---|---|---|---|---|---|---|---|
| 10-18 19:41 | acceptance-1792377693-cpu | CPU | 1.1 m | 0.00 | 0.00 | INCOMPLETE (upload refused) | fix upload path |
| 10-18 19:58 | acceptance-1792378723-cpu | CPU | 14.2 m | 0.01 | 0.03 | INCOMPLETE (upload time cap) | fix transfers |
| 10-18 20:31 | acceptance-1792380711-cpu | CPU | 2.8 m | 0.00 | 0.01 | INCOMPLETE (CLI timeout) | fix timeouts |
| 10-18 21:27 | acceptance-1792384051-cpu | CPU | 19.3 m | 0.02 | 0.03 | INCOMPLETE (kernel completion file) | fix exec completion |
| 10-19 00:20 | acceptance-1792394444 cpu + l4 | CPU, L4 | 4.0 + 4.3 m | 0.11 | 0.09 | INCOMPLETE (L4 telemetry deadline) | fix telemetry recovery |
| 10-19 00:42 | acceptance-1792395748 cpu + l4 | CPU, L4 | 4.2 + 4.1 m | 0.05 | 0.02 | INCOMPLETE (L4 upload) | fix upload |
| 10-19 01:19 | acceptance-1792397953 cpu + l4 (run 7) | CPU, L4 | 4.0 + 7.9 m | 0.13 | 0.18 | **PASS** | CLI acceptance passed → S1 admitted |
| 10-19 07:46 | s1-mature-n3-223213833 (dispatch-01) | L4 | 8.3 m | 0.19 | 0.00 | INCOMPLETE (transfer refused on symlink) | admission-source repair |
| 10-19 09:00 | acceptance-1792425613 cpu + l4 (public release) | CPU, L4 | 4.0 + 10.1 m | 0.25 | 0.25 | **PASS** | re-acceptance for the S1 public release |
| 10-19 09:17 | s1-mature-n3-223213833 (retry) | L4 | 5.7 m | 0.04 | 0.10 | INCOMPLETE (foreign generation refused) | settlement-basis fix |
| 10-19 10:12 – 18:32 | **S1** 10 jobs, serial 1 lane: n3 × 5 seeds (~30 m each) + n1 × 5 seeds (~68 m each) | L4 | 8.3 h | **12.13** | 15.62 | 10 × PASS | **S1 KILL both arms** (−12.5 / −14.9 pooled mass vs B8); audit PASS |
| 10-19 22:19 | acceptance-1792473566 cpu + l4 (carry-over, run 9) | CPU, L4 | 4.1 + 10.2 m | 0.16 | 0.25 | **PASS** | CLI **QUALIFIED on dab00823** |
| 10-20 12:07 | **A5 #1**: t34-exit-b8-v2-a5-s0-a1 / s1-a1 | 2 × L4 | 15–16 m | 0.71 | 0.95 | PASS (run), benchmark STOP | ~50 row-steps/s/lane vs 3,000 floor (serial shadow fill bug); 0 decisions → shadow batching |
| 10-20 13:15 | **A5 #2**: t34-exit-b8-v2-a5-s0-a3 / s1-a3 | 2 × L4 | 29 m | 1.45 | 1.89 | INCOMPLETE (30-min wall cap) | STOP by telemetry (~56 row-steps/s/lane, 0/256 worlds) → speed review → engine fan-out |
| 10-20 14:52 | **L0** CPU profile: t34-exit-b8-v2-a5-s0-a3-cpu-profile | CPU | 6.7 m | 0.00 | 0.05 | PASS | CPU runtime nproc=2 (37 rows/s) → CPU runtimes not used |
| 10-20 14:59 | **L0** A5 on L4: t34-exit-b8-v2-a5-s0-a3 / s1-a3 (final-3ebf3991) | 2 × L4 | 33 m | 1.49 | 1.74 | PASS | **A5 v2 GO**: 57.2 / 54.4 worlds/h/lane, ~0.036 u/teacher world → Round 0 sized |
| 10-20 16:46 – running | **Round 0 chunk 000** attempt-01 (`snake-t34-round0-chunk-000-attempt-01-l1-…`, endpoint `gpu-l4-s-kkb-ass1a1-eqee8x199zm4`) | L4 | ≥ 2 h 31 m | **3.78 so far** (1,157.62 → 1,153.84) | — | remote job `succeeded` at 7,859 s; supervisor export/settle in progress; assignment still active at 19:18 | pending (all 3 chunks terminal before any analysis) |
| — | Unpaced retention arm (Path B, Colab) | — | — | **0 (not dispatched)** | | merged, cap 42.5 u | waits for Round 0 lane |

Receipt paths: `colab-cli-acceptance/acceptance-*/outputs/{cpu,l4}/receipt.json`;
`s1-mature-20261018/{admission-source-repair,admission-public-release,settlement-basis-fix}-20261019/production/training/*/receipt.json`;
`colab-cli-carryover-dab00823/acceptance-1792473566/`; `t34-a5-20261020/l4-{0,1}-output/`;
`t34-shadow-batch-20261020/final-l4-{0,1}-output/`; `t34-engine-fanout-20261020/manifests-final-3ebf3991/{cpu-profile,a5-l4-s0,a5-l4-s1}-output/`;
Round 0: `t34-round0-study-01/` (active record `~/.config/colab-cli/snake/active-*.json`: cap 11.07 u, reserve 2.21 u).

### CLI-era totals by purpose

| Purpose | Units | Share |
|---|---|---|
| CLI acceptance / qualification (9 runs) | 0.73 | 4 % |
| S1 admission failures (2) | 0.23 | 1 % |
| S1 production (10 fine-tunes) | 12.13 | 59 % |
| A5 #1 + A5 #2 (STOPs) | 2.16 | 11 % |
| L0 (CPU profile + A5 GO) | 1.49 | 7 % |
| Round 0 chunk 000 (running) | 3.78 | 18 % |
| **Total** | **20.53** (1,174.37 → 1,153.84) | |

Decision yield: 3 decisions with units behind them (S1 KILL, A5 v2 GO, CLI qualified), ~2.4 u of engineering STOPs that
redirected work, and Round 0 pending.

## Planned lanes (registered or drafted)

| Lane | Units | Condition | Source |
|---|---|---|---|
| Round 0 remainder (chunk 000 tail + chunks 001/002: 100 + 40 pairs) | **~5.6 expected** (scaled from chunk 000 ≈ 3.8–4 u per 100 pairs); plan says ~9; caps ≤ 21.6 (≤ 43 with one mechanical rerun each) | registered, 48 h dispatch deadline | `research/t34_round0/REGISTRATION_ROUND0.json` |
| Round 0-D dense-teacher premise test | **26.7–39.3** | only on Round 0 NO-GO; DRAFT, not ratified | `research/t34_round0d/PREREGISTRATION_ROUND0D.md` |
| Claim A L4 qualification | **≤ 5** | Claim A package | Claim A specs |
| Unpaced retention arm (Path B, descriptive) | **≤ 42.5** (cap; expected unknown) | merged; after Round 0 frees the lane | `docs/research/specs/2026-10-20_t34_unpaced_arm_colab.md` |
| Dense R1 at 560 fresh worlds | **49–75** | needs own registration + Fable approval; after Round 0 GO / 0-D | `research/t34_round0d/VERIFICATION_ROUND0D.md` |
| _Also named in REVIEW_B §6, outside this brief_ | G4 probe ≤ 13.2, CUDA fit qual ≤ 10, R1 labels ~29, A100 fit ≤ 12 | conditional | `docs/research/advice/2026-10-20_a100/REVIEW_B_decision.md` |

Totals for the five lanes in this brief:

- **Everything at its upper bound** (treating 0-D and dense R1 as both happening): 9 + 39.3 + 5 + 42.5 + 75 ≈ **171 u**.
- **Lower / expected**: 5.6 + 26.7 + ~2 + 42.5 (cap; real could be lower) + 49 ≈ **126 u**.
- Round 0 **GO branch** (no 0-D): ~9 + 5 + 42.5 + dense R1 49–75 ≈ **106–132 u**.
- Adding every REVIEW_B extra at cap: up to **~235 u**.

## Gap versus the owner's goal (honest)

- Goal: spend the remaining **1,153.84 u** by ~2026-11-20 → **~38 u/day** for 30 days.
- Everything currently planned, at its caps, is **~171 u** (~235 u with all REVIEW_B extras): **15–20 % of the balance**.
  That leaves **~920–1,030 u (80–85 %) with no lane assigned**.
- Time view: the planned lanes span roughly Tue–Sat (10-20 → 10-25). At 38 u/day that window "should" use
  ~190 u; the plan uses ~106–171 u, so even the near term runs at **~55–90 %** of target pace, and nothing is planned
  after ~10-25.
- Pace to date: **~10 u/day** in the CLI era (~27 % of target). Most wall time went to engineering fixes, reviews and
  Mac-bound work, not Colab.
- Structural reason: the T34 decision lanes are cheap on L4 (~0.036 u per teacher world; one L4 at 1.54 u/h burns only
  ~37 u/day running 24/7). Hitting the goal needs **about one L4 busy around the clock for 30 days**, or 2–3 parallel
  L4 lanes for 8–12 h/day, on **decision-producing** work. That needs a portfolio beyond T34 (the portfolio advice
  lists ~745 u of named envelopes with ~300 u uncommitted — `docs/research/advice/2026-10-20_compute_portfolio_*`),
  not idle or padded allocations.
- Missing the goal is the expected outcome on the current plan unless new registered lanes are added by ~10-25.
