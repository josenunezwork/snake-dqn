# PLAN_A100.md — the A100/H100 program (parallel research lane; does not touch the Mac-serial chain)

Written by the round-20 Fable judge, 2026-10-22 ~20:15 PT, from `round-20/JUDGMENT.md` (verified facts and rulings there), `PLAN_v3.md`,
`LIVE_FACTS.md` (18:40 PT). Authority: owner > registered rules/amendments > PLAN_v3 (Mac chain) > this plan (A100 lane) > advisor memos.
Owner opt-in (round-20 QUESTION): A100/H100 wanted, no budget limit; serving stays on this Mac; GPU-trained artifacts may be served if they
fit 24 fps / update p95 ≤ 11.688 ms. Owner rules applied: budgets 2× once; never idle (every Colab slot has a consumer or is released);
advisors before design (every identity/estimand change below is ruled in round 20 or goes Astra → Fable); dispatch only via
`tools/ops/dispatch.sh`; Sol implements, Codex-V reviews, Claude dispatches and takes the one look.

## §0 The program in one paragraph

Rebuild the exact rollout loop as GPU-resident batched worlds (sim + v8 + vector61 + five opponents + B8 + collection), starting from the
existing `research/gpu_sim_spike/` engine, whose 0.217 steps/s was a Python corpse-candidate loop (62,400 iterations/tick) plus a dense
free-space BFS — not the chip. Use the engine for three things, in this order: (1) a **label factory** producing counterfactual, CPU-rescored
H300 outcomes for every legal action at hundreds of thousands of natural and doom-enriched roots; (2) a **prediction package** (exact-H64-prefix
leaf evaluator, discrete death hazards, rescue availability) whose only lawful roles are ordering/scheduling exact work and, after a
selection-conditioned audit, a cutoff with exact fallback; (3) a **scale + distillation study** (1.64 M vs ~6.5 M vs ~25 M teacher) producing a
bare Mac-servable student, screened on GPU development banks and then handed to the Mac chain as a new identity. P1's collector is rebuilt on
the same engine (WP5). Every package has a kill number frozen before data; no GPU result is gate evidence.

## §1 Work packages (ordered; Sol builders, Codex-V review; GPU = allocated hours incl. setup/export)

| WP | Name | Builder → review | Builder days | GPU (A100-h) | Base → effective units | Replaces / accelerates |
|---|---|---|---|---|---|---|
| **WP0** | Mac runtime/identity assay (sizes S/M/L; PyTorch CPU / ONNX / MPS / Core ML fp32; full-update p95 with live Watch6) | Sol-K 0.75 d → Codex-V 0.25 d | 1 | 0 (Mac quiet gap) | 0 u | gates every served-size claim |
| **WP1a** | E1b event-path repair + stage timers + launch counts; A1 ruler | Sol-G 0.5–1 d → Codex-V 0.25 d | 1 | 1 | 10 → **20 u** | reopens E1b (PLAN_v3 §7 narrowed) |
| **WP1b** | Sparse free-space (vector61) flood on the existing CUDA queue; grouped opponent inference; fused food maintenance; graph-scheduled tick | Sol-G + Sol-G2 → Codex-V | 4–6 (2 builders) | 4 | 40 → **80 u** | — |
| **WP1c** | Engine qualification: action-tape parity (100 k ticks), closed-loop parity (100 natural H5000 worlds), ties classified; three warmed 10-min runs at 256/1,024/4,096 incl. resets/deaths/refills; optional reference-RNG restoration (path A) | Sol-G/G2 → Codex-V (independent oracle fixtures) | 4–6 | 6 A100 + 2 L4 | 70 → **140 u** | — |
| **WP2** | Label factory: natural roots (pinned B8 s3 + Watch roster and T3 rosters), doom enrichment by natural late-state restarts, all legal actions × exact H300 (B8 continuation), 10 % H1440 audit; **every consumed label CPU-rescored from its action tape**; lives counted | Sol-M 2–3 d → Codex-V 1 d | 3 | 100 k roots: 2.5 h at 20 k/s (+1 h CPU rescoring on host); 1 M roots: 25 h | tranche 1: 40 → **80 u**; per 1 M roots ≈ 190–250 u forecast | replaces Phase-0/Round-0 label production (1 M decisions ≈ 129 Mac-h) and B2 as root source after Oct 30 |
| **WP3** | Prediction package: one trainable encoder (S or M), three heads — per-action remaining-horizon survival + mass after exact H64 prefix; censored hazards at 30/60/100/200/300; rescue availability (only where all legal actions were tested); world-grouped fit/calibration/audit; selection-conditioned audit on sealed exact outcomes | Sol-M 2 d → Codex-V 1 d | 3 | 4 fit + 2 audit | 60 → **120 u** | V0/V1 leaf critic (bl_v0 C3/C4 concept) becomes this; T35 lineage closed |
| **WP4** | Scale + distillation: S vs M (vs L teacher) on nested 2 k / 10 k / 100 k / 1 M roots; cost-sensitive categorical action loss with B8 retention on non-decisive rows; one on-policy DAgger relabel round (student visitation) through WP2; GPU development screen 128 paired H5000 worlds/mix (+ 24 paired 10-min Watch6 sessions) | Sol-M + Sol-D → Codex-V | 4–6 | 12 fit + 6 screen | 180 → **360 u** | reopens Claim A with new labels + representation (new registration) |
| **WP5** | P1 resident collector: A3 learner ceiling first; then rollout tensors on device, duplicate teacher forward removed, time-flattened learner forward, resident env at the registered 128-world shape (wider is a new recipe) | Sol-R 3–5 d → Codex-V | 4 | 2 + 8 | 100 → **200 u** | replaces L4 collection (1,440/s → target ≥ 10 k/s with learning); 100 M = 2.8 h at target |
| **WP6** (optional) | Prediction-powered development evaluation: predictor of paired candidate − B8 mass + random exact audit worlds; GPU development banks (needs WP1c path A or per-world CPU rescoring) | Sol-M 1 d → Codex-V 0.5 d | 1.5 | 1 + banks | 20 → **40 u** | 2–4× fewer exact development worlds; strict/product unchanged |

Totals: ≈ 22–30 builder-days across 3–4 Sol lanes + ~5 Codex-V days; ≈ 50 A100-h + ~25–60 h of label production in weeks 1–3
(≈ 120–200 A100-h ⇒ ≈ 650–1,500 u forecast at 5.4–7.5 u/h, caps 2× once). H100 substitution: only for a package whose A100 profile is
memory-bandwidth-bound (flood/BFS kernels plausibly are) after a matched 30-min packet shows ≥ 1.5× wall and a live rate reading exists.

### Guardrails that stop a surrogate lying silently (apply to WP2–WP4, WP6)
1. **Exactness boundary.** Collision, body release, food order, RNG and opponent carry are simulated, never predicted. Device labels are
   "simulator D" labels until WP1c path A passes; every label a fit consumes is CPU-rescored from its recorded action tape (`cpu_rescored_label`
   path, ~0.3 s per H300 branch). Unexplained mechanics divergence on a rescored root ⇒ the tranche is INVALID.
2. **Splits by ancestry.** Parent world → all branches, restarts, H1440 extensions and shadow descendants on one side; audit worlds hash-selected
   before fitting; exact outcomes of audit roots sealed until the model is frozen.
3. **Selection-conditioned audit (09).** On ≥ 256 world-disjoint audit roots stratified ordinary/decisive/roster, every legal action gets an
   exact H300 continuation; report predicted − exact on the *selected* action, lost-rescue rate (rescue-capable roots where the top-1 dies),
   top-2-contains-survivor rate, calibration (Brier/log loss at natural prevalence), and harmful-override rate. Tier promotion rules:
   ordering/scheduling allowed if completed useful roots/s improves ≥ 25 % in matched replay with ≥ 95 % rescue retention; cutoff-with-fallback
   allowed only if lost rescues ≤ 5 % and top-2 ≥ 95 %; certification never.
4. **Uniform audit stream.** 10 % of development roots are evaluated exhaustively regardless of any score, so false negatives stay observable.
5. **Lives, not rows.** Every fit/claim reports independent rescue-bearing lives (≥ 64, ≥ 16 held out) and parent-world counts; INSUFFICIENT
   is final for that tranche, never extended to success.
6. **Gameplay decides.** Agreement/AUC/regret admit a candidate to GPU development screening only; a served claim needs fresh paired Mac
   evidence under the contract (PLAN_v3 §4 chain), with misses/fallbacks counted.

## §2 Expected speedups and gains (arithmetic; assumptions named)

- **Engine.** Unrepaired: 4.6 s/tick at E = 6 (host-launch-bound: 1.5 M eager ops × ~3 µs). After WP1a the corpse path is one fused launch per
  tick; the next wall is the dense vector61 BFS (~160 × ~10 whole-board passes at E = 1,024 ≈ 1–1.5 TB/tick ≈ 1 s ⇒ ~1 k steps/s). After WP1b
  (sparse flood, grouped opponents, graph-scheduled tick) the conjectured rate is 20–100 k complete steps/s at E = 1,024 (51 → 10 ms/tick).
  **Hinge:** B8 + five opponent forwards for 1,024 worlds (~6 × 11 M MACs × 1,024 ≈ 70 GFLOP) are ≪ 10 ms on an A100; floods and observation
  painting dominate — unmeasured until A1's trace.
- **Labels.** 1 M six-action H300 roots = 1.8 B steps = 25 A100-h at 20 k/s (5 h at 100 k/s) vs ≈ 129 Mac-h per 1 M *searched decisions* on the
  Phase-0 path; CPU rescoring 1 M × 6 × 0.3 s / 12 vCPU ≈ 42 h — so rescoring, not GPU search, bounds the 1 M tranche: rescore every audit/decisive
  label and a 10 % uniform sample of ordinary labels (ruled lawful for training labels; audit labels 100 %).
- **Search-time prediction (WP3 tier ii).** If 80 % of branches stop at the H64 prefix: 64 + 0.2 × 236 = 111 frames vs 300 ⇒ 2.70× fewer frames;
  with evaluator overhead 5 % ⇒ 2.38×; with 20 % full fallbacks ⇒ 1.75×. Must beat the already-exact-stop baseline (bare-first, first survivor),
  so the realistic saving is on all-die and late-survivor roots only — measure, do not assume.
- **Served gain.** No numerical forecast. The prior: L2′ +83 pooled (frozen +56, mixed +28) is survival-only, 28 rescues, historical rosters;
  a bare student that captured half of frozen+mixed (+21) would fail the +40 bar; the hypothesis WP4 tests is that counterfactual labels across
  ordinary states (growth, boost) add mass beyond rescues (O1 GO: 35/48 worlds had a ≥ +5 corpse-intake branch). Kill rules below.
- **P1.** Collector 1,440/s → ≥ 10 k/s with learning (06's inequality: C + L + O ≤ 1.638 s per 16,384-transition rollout) ⇒ 100 M in 2.8 h; only
  if A3 shows the learner ≤ 1.6 s per rollout.

## §3 First 24 h — dispatchable specs (Thu Oct 22 20:30 → Fri Oct 23 20:30 PT)

**A0 — E1b event-path repair (Sol-G, 4–8 h, 0 u; gates A1).**
- In `research/gpu_sim_spike/device_sim.py`: replace the per-column Python loop in `_append_many` (`:440–458`) with device-side processing:
  (i) early device reduction — if no world has a death this tick, skip corpse dispatch entirely while advancing nothing that affects the RNG
  counter; (ii) for ticks with deaths, compact candidates to the actual dead segments (`corpse_mask.nonzero` per world, padded) and perform
  ordered insertion/dedup/eviction in one kernel or a bounded tensor program with a static per-world loop over *actual* candidates
  (≤ 1,500 per dead snake × dead count), never over `S × cap`. (iii) `_maintain_food` (`:498–510`): compute `want` once, run trials only for
  worlds that need food, and advance the counter RNG by exactly the trials the reference stream would have consumed, so the E1b RNG stream is
  unchanged (prove with a fixture: identical food cells before/after on 1,000 ticks).
- Add exclusive stage timers (movement, deaths/corpse, food, vector61, v8 floods, opponent forwards, B8 forward, observation paint, collection)
  with `torch.cuda.synchronize()` only in an instrumented mode, plus CUDA launch counts per stage; receipts keep `host_stages` non-empty.
- Codex-V fixtures: full-food, refill, single death, simultaneous deaths, capacity-eviction, at production capacity 10,400; before/after
  byte-identity of food/corpse state on recorded action tapes; no capacity truncation.

**A1 — Repaired full-tick ruler (Claude dispatches on A100-40; base 1 h / 10 u, effective 2 h / 20 u; `resume_exemption: acceptance`).**
- Shapes: E ∈ {6, 256, 1,024} × mixes {scripted, frozen, mixed}; six bodies 800–1,500 (realistic_benchmark template); real B8 s3 hero, five
  real opponents (FRP3 s12 + v8 where the mix says so), vector61 + v8 features, observation paint, movement, food/reset and collection writes;
  three warmed repeats × 8 complete ticks per cell; one unrepaired tick at E = 6 on the same allocation as the control; synchronized end-to-end
  wall per tick; separate instrumented pass for stage attribution; parity self-check on one recorded action tape vs the CPU reference
  (classification, not a pass bar).
- Report per cell: median and p95 tick seconds, complete steps/s, stage shares, launch counts, peak memory, setup/export seconds.
- **Kill tiers (frozen):** oracle mismatch unexplained ⇒ INVALID. Full-tick speedup at E = 6 < 5× vs unrepaired ⇒ event-path hypothesis rejected;
  no further kernel work without a trace naming a stage ≥ 50 %. Median complete steps/s at E = 1,024 on any mix **< 2,000 ⇒ engine program
  STOP** (WP1b/WP1c/WP2 not funded; WP3/WP4 continue on CPU-labelled banks only). 2,000–20,000 ⇒ fund WP1b (sparse free-space + grouped
  opponents) with a second ruler at the same shapes; stop if still < 20 k. ≥ 20,000 on every mix ⇒ WP1c + WP2 tranche 1 funded.
- Transport: A100 runtime is a new manifest family ⇒ same-pin real-transport canary (≤ 30 min, ≤ 3 u) first; a kernel-RPC stall ⇒ INCOMPLETE,
  one outcome-blind mechanical retry, second ⇒ judge. Assignment count ≤ 3 at admission.

**A2 — Scale + target falsifier on existing labels (Sol-M 1 d → Codex-V 0.5 d; A100-40 base 2 h / 20 u, effective 4.8 h / 48 u).**
- Data: T5/T8 bank (SHA `e4ecbe99…`, 19,976 states, 103,264 legal branches, 342 worlds, five frozen world folds, label = recorded branch dies
  within 300) and, as a second table, the permitted excluded-A5 diagnostic cache (190,779 anchors, 225 decisive rows / 21 worlds) — never
  sealed study outputs. Nested training-world subsets 25/50/100 % inside each fold.
- Arms (fixed endpoint, 2 seeds each, no checkpoint selection): C0 frozen-linear D300 (control; reproduces T35/T5b class); S trainable
  B8-shape encoder from scratch and from B8 init; M ≈ 6.5 M encoder (towers 64/128/128 + one block each); each with (a) single-horizon D300
  head and (b) censored discrete hazards at 30/60/100/200/300 with survival S(h) = Π(1 − h_k). Report D300 AUC/Brier, multi-horizon scores,
  decisive-row agreement on the A5 table, world-bootstrap CIs, and the 50 % → 100 % slope per arm.
- **Kill (frozen):** M − S D300 AUC < +0.02 (LB ≤ 0) ∧ slope(50→100 %) < +0.01 ⇒ "scale/volume on this label type" closed: WP4 may only be funded
  on WP2's counterfactual labels. Hazard arm < 10 % Brier improvement over single-horizon ⇒ single-horizon kept in WP3. Any result here is
  exploratory (reused historical bank, source continuation policies), not evidence for a served gain.

**A3 — P1 learner ceiling (Sol-R 4 h → Codex-V; A100-40 1 h + L4 1 h; base 2 h / 20 u, effective 4 h / 40 u).**
- On the frozen P1 128 × 128 rollout shape with actor enabled: (1) cached learning only (rollout tensors pre-staged on device), (2) collection
  only, (3) complete PPO cycle; phase timers (collection, staging/H2D, fwd/bwd/Adam, scoring, export) and sampled GPU-busy; identical on L4
  for the ratio. Also remove the duplicate teacher forward (`run.py:89–92`) and flatten the per-step learner forward (`learner.py:228–240`)
  as a *separately labelled* numerical comparison (gradient equality on a fixed batch within fp32 tolerance; reassociation reported).
- **Kill:** cached actor-enabled learning > 1.6384 s per rollout ⇒ the learner cannot reach 10 k/s with free collection; WP5 pauses for a learner
  fix before any collector port. Unexplained gradient change ⇒ optimization rejected. No recipe change; no interaction with P1 amendment-04's
  envelope or its one-look rule.

**A4 — Mac runtime/identity assay (Sol-K 0.75 d → Codex-V 0.25 d; 0 u; first quiet gap ≥ 45 min after X5 on Fri, never during M1/X5/C1).**
- Extend `research/t34_shadow/runtime_probe.py`: shapes S/M/L (random weights are fine for timing; B8 s3 weights for identity); backends
  PyTorch CPU (1 thread), ONNX CPU, MPS (with `torch.mps.synchronize()`), Core ML fp32 CPU / CPU+GPU; batch 1 and branch batches 6/24; 10,000
  held-out observations grouped by world incl. long bodies, boost, masks, near ties; warm timing includes raster/v8 featurisation, conversion,
  output materialisation and argmax; cold load separately; one 10-minute window with the live mixed Watch6 game ticking at 24 fps and the
  full-update p95 recorded.
- **Kill:** a size/backend with any registered overrun or update p95 > 11.688 ms ⇒ teacher-only; action disagreement vs the fp32 reference
  not explained by a classified tie ⇒ that backend is a different controller (requalify or drop).

Dispatch order: A0 build now; A2 and A3 packages to Codex-V tonight, dispatch when PASS (slots permitting, P1 Y3 canary/source first if
ready); A1 when A0 PASSes (target Fri afternoon); A4 in Friday's gap. Owner report 09:00/21:00 PT carries A1's number the evening it prints.

## §4 Week-by-week (PT), decision points, kill rules

| When | Decision | Rule |
|---|---|---|
| Fri Oct 23 evening | **G1 engine** | A1 tiers (§3): STOP / fund WP1b / fund WP1c + WP2 |
| Sat Oct 24 | G2 scale-on-old-labels | A2 kill ⇒ WP4 only on WP2 labels; A3 ⇒ WP5 go/pause |
| ~Thu Oct 29 – Fri Oct 30 | **G3 engine qualified?** | WP1c: ≥ 20 k/s per mix (three 10-min runs), parity PASS with ties classified, ≥ 3× completed roots/h vs the native Mac pipeline ⇒ WP2 production; else STOP engine, WP3/WP4 continue on CPU labels |
| ~Sat Oct 31 | G4 tranche 1 | 100 k roots: lives ≥ 64 (≥ 16 held out), CPU-rescoring mismatches 0 unexplained, H1440 sign reversals ≤ 10 % ⇒ WP3 fit + WP4 2 k/10 k doses |
| ~Tue Nov 3 | G5 predictions | WP3 audit tiers (§1 guardrail 3); scale pair: ≥ 10 % regret reduction, LB > 0, overrides ≤ +1 pp |
| ~Thu Nov 5 – Sat Nov 7 | G6 GPU development screen | one frozen student (or learned-ordered assisted rule) vs bare B8: 128 paired H5000/mix: pooled point ≥ 40 ∧ LB90 > 0 ∧ no mix UB90 < 0 ∧ guards ⇒ Mac-chain candidate registered (new identity); UB90 < 40 ⇒ candidate closed |
| Nov 10 | campaign close (PLAN_v3) | this lane continues as the post-campaign charter with ≤ the slots the owner allows; a live GPU-screened candidate enters the Mac chain under a new PLAN |

Standing kill for the whole lane: two consecutive packages ending INVALID for mechanics/parity reasons ⇒ engine lane paused for a judge;
any package that reads sealed/gate data ⇒ lane STOP; any Mac overlap with a leased window ⇒ that attempt INVALID and the builder's Mac
access removed.

## §5 Interaction with the live Mac chain and Colab slots

- Mac: untouched except A4's gap. M1 attempt-05 → X5 → D1 (Sat 12:00, ≤ 24 h slip) → C1 → D2 → C15/B3 → D3 → L1″/L3/L4 → strict → product →
  D6. No A100 artefact enters C15 (no learned component this campaign).
- Colab: 3 assignments. Priority when contended: P1 Y3 canary/source (ratified) > A1 > A2 > A3 > WP2 tranches > WP5. P1 100 M tranche HELD
  until G1. B2 production not before its transport remedy (round-17 §(d) permission on one A100 stall census stands, ≤ 3 u).
- Units: this lane is the consumer of the ~1,100 u balance the owner wants spent; ≈ 90–200 u in week 1, ≈ 300–600 u in weeks 2–3 at forecast
  rates; every manifest still declares base/effective caps and a verified-ledger resume.

## §6 What must not change (copied from round-20 §(g); binding)

B8 s3 served until a strict pass and the owner's OK; serving on this Mac (W ≤ 4 / K ≤ 8); every SUCCESS_CONTRACT bar; honest paired
evaluation (paired worlds, fresh namespaces, ancestry splits, one look, no N shrink, Holm, pre-frozen kill numbers, sealed exact outcomes,
published ledgers, 2× once, first cap wins, INCOMPLETE ≠ pass); no learned transition model in any label/fixture/gate; CPU rescoring of every
consumed GPU label; strict/product exact on this Mac; PLAN_v3 §1–§6 and all prior rulings/prices; three-assignment cap; `tools/ops/dispatch.sh`
only; never two Mac jobs; never kill daemons; `t34-round0-study-01/` closed; `../snake-dqn-l2mac` untouched. Not committed by the judge.
