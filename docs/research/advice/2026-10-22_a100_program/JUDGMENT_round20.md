# Round 20 judgment — "Move work onto A100: where can prediction and GPU batching replace slow computation?"

Judge: Fable (round-20 judge), 2026-10-22 **~19:30–20:30 PT** (`date` = 19:27 PDT at the first receipt read). Read-only: no python, no jobs,
no Colab, no tmux, nothing written outside this file and `../PLAN_A100.md`. `t34-round0-study-01/` not opened. Inputs read in full:
`QUESTION.md` (incl. the owner's model-scale follow-up), `personas.txt`, `astra_01..10.md` (all ten), round-19 `QUESTION.md` + all ten
bottom lines, `../round-17/JUDGMENT.md`, `../round-18/JUDGMENT.md`, `../PLAN_v3.md`, `../LIVE_FACTS.md` (18:40 PT), `../QUEUE.md`,
`../exp/JUDGMENT_M1_ATTEMPT5.md`. Verified in code/receipts (file:line, repo-relative unless noted):

- **E1b corpse-candidate loop — VERIFIED, and stronger than the advisors wrote.** `research/gpu_sim_spike/device_sim.py:440` is
  `for col in range(n)` with ~20–30 eager tensor ops per iteration (`:441–458`); `step()` calls it unconditionally with
  `n = S × cap` (`:839–843`, `body.reshape(self.E, self.S * self.cap, 2)`), no early return on an empty death mask; `cap` defaults to
  `cfg.body_storage_capacity or cfg.max_capacity` (`:53`) = **10,400** (`research/apex_safety_20260926/deployment.yaml` `pqn.max_capacity`)
  with `num_snakes: 6` ⇒ **62,400 iterations ≈ 1.2–1.9 M eager ops per tick**. The L4 receipt
  (`exp/bl-e1b/cuda-dtype-fix-20261021-01/l4-output/exports/decision-0000.json`) shows `device_loop_seconds` **8,294.06** and
  `host_launch_cpu_seconds` **8,296.32** for **1,800** full world-steps (0.217/s = **4.6 s/tick**), `host_stages: {}`, `policy_timing: {}`.
  Host CPU time ≈ device-loop wall ⇒ the host was saturated for the whole loop; 1.5 M eager ops × the ~3 µs eager-dispatch floor on a Colab
  vCPU ≈ **4.5 s/tick** — the loop alone reproduces the measured tick without any other cost. This is an inference from a receipt plus
  code, not a stage timer (none exists); it is the strongest causal candidate on the board. Secondary: `_maintain_food` runs
  `max_food` = 300 trials per tick each with a full `occupancy()` rebuild (`:498–510`).
- **The device engine is already a different simulator.** `port_limitations` (`device_sim.py:101–106`): counter RNG, bounded rejection
  sampling, cell-centre geometry. The same receipt's `comparison.branches` show `alive_steps_error 0` but `mass_integral_error` up to
  **−4.86** on a short scripted root: first action agreed, branch outcomes did not. Its `cpu_rescored_label` replayed the recorded action
  tape on the CPU simulator in **0.305 s** for an H300 branch — the cheap exact-rescoring path the program below relies on.
- **Predicted second wall (not in any advisor note):** the vector61 free-space feature runs a dense whole-board BFS
  `for _ in range(FREE_SPACE_BFS_CAP - 1)` over a `(E, S, 3, H, W)` bool tensor (`research/gpu_sim_spike/vector_device.py:180–186`), with
  `FREE_SPACE_BFS_CAP = 160` (`src/simd_env/vector61_featurizer.py:80`). At E = 1,024 that is ~160 × ~10 whole-board passes over ~0.9 GB
  ≈ 1–1.5 TB of traffic per tick ≈ **~1 s/tick on an A100 ⇒ ~1 k steps/s**, i.e. fixing the corpse loop alone cannot reach 20 k/s; the
  free-space count must also move to the existing sparse CUDA flood (`research/gpu_sim_spike/v8_device.py:134–144`, `flood_cuda.flood`).
  Opponents are also dispatched one slot at a time (`device_policy.py:153–168`).
- **T35 KILL — VERIFIED** (`research/t35_outcome_head/RESULT.md:20–34`; `PREREGISTRATION.md:20–32`; `model.py:1–7`): linear probe on
  frozen 256-D features, 296 searched rows / 225 decisive / 21 worlds; AUC 0.747 [0.705, 0.792], agreement 46.2 %, chosen action survives
  60.9 %. Related: T8 raw-raster CNN (294,353 params, 19,976 states / 103,264 branches / 342 worlds) AUC **0.742** vs frozen-fuse GBM 0.765
  (`docs/research/t8_probes_2026-10-17.md:31–33,55–66`); T10 horizon decay D30 0.933 → D300 0.749 (`t10_certificate_horizon…:172–177`).
- **B8 architecture/size — VERIFIED** (`src/model/ego2s_network.py:33–66`, inputs 7×31×31 local, 4×37×37 global, 60 scalars): params =
  local tower 1,106,816 (of which **Linear(4096→256) = 1,048,832**) + global tower 415,216 (Linear(3200→128) = 409,728) + fuse 114,432 +
  heads 1,799 = **1,638,263 ≈ 1.64 M**. The conv towers hold only ~60 k params; the model is ~11 M MACs per forward. Mac batch-1:
  PyTorch CPU p50/p99 0.344/0.609 ms, ONNX 0.138/0.300 ms, one fixed input (`research/t34_shadow/RESULTS.md:224–231`).
- **P1 collector — VERIFIED:** per-frame barrier + pipe scatter (`research/bl_ppo/collector.py:720–751`); teacher forward repeated in the
  adapter (`run.py:89–92`) after `policy.act` already ran it (`model.py:204–206`); learner forwards one time slice at a time
  (`learner.py:228–240`). Round-17's measured learner share 15 % (critic-only) stands.
- Contracts: `SUCCESS_CONTRACT.md` §2 (+40 ∧ LB90 > 0, guards), §4 (24 fps, update p95 ≤ 11.688 ms), §5 amendment (W ≤ 4 / K ≤ 8 on this
  Mac); `P7_STATISTICS.md:14–39` (paired world unit, Holm, one look); `AGENTS.md:117–121` (2× once); `exact_stop.py:20–60`
  (bare-first, first survivor); `bl_v0/REGISTRATION.md:16–25` (H64-prefix cutoff, C3/C4 already registered as a concept);
  `CAMPAIGN_LOG.md:105` (A100 rule), `:121` (L2′ FUND), `:127` (natural doom 0.058/1 k); owner decision 2026-10-21 00:30 (serving on this
  Mac; Colab GPUs for training/labels/evaluation only). M1 attempt-05 admitted with in-attempt retry; D1 may slip to Sun Oct 25 12:00.

## (a) Bottom line — for the owner (12 lines)

1. **Yes — fund an A100 program now, as a parallel research lane.** The thing to move is not "predictions instead of simulation"; it is the
   **exact simulator + opponents + v8 + B8 inference as thousands of GPU-resident worlds**, which turns label production, counterfactual
   data and development evaluation from Mac-days into GPU-hours. Predictions then *order, schedule and distil* that exact work; they never
   certify a survival. All ten advisors converge on this; the red team (09) and the ambitious case (10) differ only on pace.
2. **E1b was not a chip failure; it was a Python loop.** Verified: 62,400 corpse-candidate iterations per tick, ~1.5 M eager ops, host CPU
   saturated the whole run — that alone reproduces the measured 4.6 s/tick. One Sol-day fixes it; one A100-hour tells us the real engine rate.
3. **First kill number (Fri Oct 23 evening):** after the repair, **< 2,000 complete long-body branch-world steps/s at 1,024 worlds** ⇒ the
   engine program stops. ≥ 20,000/s ⇒ label factory funded. In between ⇒ one more kernel (sparse free-space flood), then re-measure.
4. **Predictions: three roles, tiered.** (i) ordering/scheduling exact work — allowed after audit; (ii) H64-exact-prefix cutoff with exact
   fallback — allowed only if ≤ 5 % of rescue-capable roots lose every rescue; (iii) certifying a branch as safe — **never**.
5. **Model scale (your question):** nothing measured says 1.64 M was ever the limit — T3 fit 99.6 % and lost 75 mass; T35 had 225
   decisive rows; a 294 k-param raster CNN matched the frozen trunk. B8's conv towers are only ~60 k params, so there is a cheap, specific
   place to add spatial capacity. **Test it, don't bet on it:** a 2-h matched 1.64 M vs ~6.5 M fit on existing labels this week, then on
   fresh counterfactual labels. ~6.5 M very likely serves on the Mac; ~25 M is a teacher until the Mac assay says otherwise.
6. **Units:** week 1 ≈ 12–20 A100-h (≈ 90–200 u); the full program ≈ 120–200 A100-h over three weeks — inside the balance, no cap needed, every
   packet still carries an effective cap (2× once). H100 only after a matched 30-min packet and a live rate reading.
7. **Nothing on the Mac-serial chain moves:** M1/D1, C1/D2, C15, B3, KEEP_B8 Nov 10 all stand. Deprioritised: the P1 100 M L4 tranche is
   **held** until Friday's engine number prints (a qualified engine makes it a 2-h job); B2 stays on its transport remedy.
8. **Honest odds:** ≈ 0 % that this changes the Nov 7 served verdict; ≈ 30 % (judgment) of a GPU-screened candidate beating B8 s3 on
   development banks by Nov 14; a served release still needs the Mac strict/product chain and your OK.

## (b) Per-advisor verdicts (verified facts in **bold**)

| ID | Persona | Verdict | One line |
|---|---|---|---|
| 01 | GPU-resident sim | **keep — anchors WP1** | **Corpse loop real (`:440`, `:839–843`, cap 10,400)**; "share unmeasured" is right in letter (no stage timer) but the receipt's host-CPU ≈ wall + launch arithmetic makes it the dominant cause beyond reasonable doubt. Missed the dense vector61 BFS (second wall). Repair-first over warp-flood ✓; 20 k/s as target not capability ✓; its 1.64 M vs 6.5 M scale test adopted (WP4). |
| 02 | Learned search / MuZero | **keep; prop 3 deferred** | Leaf value after exact H64 prefix (= `bl_v0` C3/C4 already registered as concept) is the right surrogate shape ✓; 2.70×/2.38× arithmetic ✓. EfficientZero macro model **not funded** this round (ruling 2). Its agreement ≥ 90 % / ≤ 5 % lost-rescue gates adopted (WP3). |
| 03 | Distillation | **keep — WP4's hypothesis** | **T3 99.64 % / −75.145 (`t3_diagnostics:17–28`), D0 KILL** are the right negatives; "representation + cost-sensitive labels + two DAgger rounds" is the one distillation hypothesis not yet refuted; **TTV clip at 255 (`full_sight.py:62`)** ✓. Its "no production without ≥ 20 k verified steps/s; ≥ 64 rescue-bearing lives (16 held out)" adopted verbatim. 14–21 days ✓ — lands after Nov 10. |
| 04 | Doom prediction | **keep — target change adopted** | T35 = tiny-data frozen-trunk selector ✓; **T8 0.742 / T10 D30 0.933 → D300 0.749** ✓ (counter-evidence to "just more data"); time-to-death hazards + rescue-availability targets adopted as WP3 heads; 6.7 % precision arithmetic ✓. Its scheduler (prop 3) is Mac-side and waits for a candidate identity. |
| 05 | Speculative cache | **partly — prop 2 not funded** | Digest/shadow facts ✓ (`shadow.py:427–`); exactness-compounding warning ✓. A GPU **future-root service for serving** is a serving-platform change the owner closed on 2026-10-21 00:30 (ruling 6). Prop 1 (prefix-survival replay) is cheap but duplicates what M1/B3 measure; prop 3 (sub-computation cache) folded into WP1 as optional. |
| 06 | RL systems | **keep — WP5** | **Barrier/pipe collector, duplicate teacher forward, per-step learner (`collector.py:720–751`, `run.py:89–92`, `learner.py:228–240`)** ✓; feasibility inequality C+L+O ≤ 1.638 s ✓. Its 2-h actor-enabled learner ceiling is the WP5 falsifier; full resident PPO only on a qualified shared engine and after P1 amendment-04 passes. |
| 07 | Mac serving | **keep — WP0 and the scale answer's binding constraint** | **0.344/0.138 ms batch-1; p95 ≤ 11.688 ms is the bar, not 41.667 ms** ✓; Core ML fp16/ANE identity caveat ✓. Runtime/identity assay adopted as the 0-u Mac companion (first quiet gap). H64 evaluator numbers ✓. |
| 08 | Statistician | **keep — guardrails** | **T35 world split correct; P1 held-out consumed** ✓; nested 2 k/10 k-root doses with ≥ 500/2,000 parent worlds adopted (WP4); 6.7 % precision ✓; prediction-powered inference adopted as optional WP6 for development banks only; strict stays exact ✓. |
| 09 | Red team: predictions lie | **keep — its audit is the program's guardrail** | Four silent failures (selection bias, rare-event blindness, compounding error, leakage) adopted as the WP3 audit design; "training and auditing against the same faulty port manufactures agreement" ✓ → CPU action-tape rescoring is mandatory (ruling 3). Its ≤ 5 % lost-rescue / top-2 ≥ 95 % kills adopted. |
| 10 | Red team: go big | **partly — pace, not substance** | Same corpse loop ✓; 2 M roots / 24 M student / 84 A100-h is the right *shape* of ambition; **overruled** on dispatching four builders and 24 A100-h before Friday's engine number (one day of waiting buys the whole decision), on 24 M as a served model before the Mac assay, and on rows as evidence (lives count). Its "Claim A lane prospectively ratified, no sealed evidence borrowed" adopted (ruling 8). |

Shared blind spots: none of the ten read the E1b receipt's `host_launch_cpu_seconds` or the dense vector61 BFS; eight of ten priced
label volume in roots, not independent rescue-bearing lives (03/04/08/09 did). Arithmetic spot-checks: 01's 1.8 B steps for 1 M dense
H300 roots ✓ (1 M × 6 × 300); 02's 2.70×/2.38× ✓; 05's 8,640 steps/s for every-fifth-decision search ✓; 06's 1.6384 s ✓; 07's 0.371/0.571
✓; 08/09's precision figures ✓ (0.54–0.80 % at 99 % specificity, 6.7 % at 99.9 %); 10's 7.9872 B ✓. Factual errors that change a conclusion:
none; one omission that does (vector61 dense BFS).

## (c) Rulings on disagreements

1. **Repair-first vs rebuild-first (JAX/warp flood).** *Repair-first.* The native CUDA floods already exist (`v8_device.py:134–144`,
   `flood_cuda.cu`); the pathology is Python dispatch. No framework migration; extend the existing PyTorch+CUDA engine. The repair must
   preserve E1b's RNG stream (device-side empty-event rejection and food early-out that advance the counter RNG identically), so the
   before/after tick is a controlled comparison.
2. **Learned dynamics / world model (02 prop 3, 10) vs exact collision (04, 07, 08, 09).** *Collision, body release, food order and RNG stay
   exact in every label and every gate.* No learned transition model enters a label, a fixture or a bank this program. A latent/macro model
   may be proposed later only to *order* exact queries, after the WP3 leaf evaluator has shown value (02's own ordering).
3. **"Exact" GPU labels vs a different simulator (01, 05, 09).** *Two lawful paths, choose per package:* (A) restore the reference RNG on
   device and pass full-trajectory parity on action tapes + closed loop (ties classified separately) — then the engine is an oracle; or
   (B) keep the device simulator as "simulator D" and **rescore every consumed label by replaying its recorded action tape on the CPU
   reference** (`cpu_rescored_label`, 0.3 s per H300 branch ⇒ 100 k roots ≈ 1 CPU-hour on 12 vCPU). Under (B) a label means "under this
   recorded opponent tape, action a survives/attains mass m" — a legitimate counterfactual for training and ordering, **not** gate evidence.
   Path (B) is mandatory from the first tranche; path (A) is a WP1 stretch goal that would also unlock GPU development banks (WP6).
4. **Role of predictions (02/07 cutoff; 04 scheduler; 09 order-only).** *One prediction package, three heads, three tiers* (bottom line 4).
   Promotion between tiers needs the selection-conditioned audit (WP3 §audit) on world-disjoint roots with exact outcomes sealed until the
   model is fixed. Tier (iii) never.
5. **Model scale (10: 24 M served; 01/02/08: matched pair).** *Matched pair first, in the conv towers.* Sizes S = 1.64 M (control), M ≈ 6.5 M,
   L ≈ 25 M (teacher/upper bound). Served size = whatever passes the Mac assay (full update p95 ≤ 11.688 ms under Watch6 contention,
   action identity on 10 k held-out observations, ties classified). Distil L → M/S if L wins offline. Details in §(d).
6. **Speculative GPU future-root cache for serving (05 prop 2).** *Not funded.* It is a GPU-host-assisted serving path; the owner ruled serving
   stays on this Mac (LIVE_FACTS 2026-10-21 00:30) and this round's brief repeats it. Reopening needs an owner decision, not a judge.
7. **Data volume: rows (10) vs independent lives (03/04/08/09).** *Lives.* Tranches of 100 k roots; natural doom 0.058–0.09/1 k decisions
   ⇒ 20 M natural decisions ≈ 1,200–1,800 doom events; enrichment by natural late-state restarts on the real map (round-14 rule, no smaller
   maps); every claim reports independent rescue-bearing lives, ≥ 64 (≥ 16 held out) before any fit is read; otherwise INSUFFICIENT, no
   extension to success.
8. **PLAN_v3 §7 stop list ("Claim A descendants; E1b; A100/H100").** *Narrowed by owner opt-in, with result honesty kept:* E1b reopens as WP1
   under a new charter (this file); Claim A reopens as WP4 only with new labels + representation and a new registration; A100/H100 are
   lawful for the A100 program's packages. Everything else in §7 stays. No sealed study output is reused; `t34-round0-study-01/` stays closed.
9. **P1 100 M tranche (PLAN_v3 Sat 24 → Tue 27 on slot A, ~19 h L4, ≤ 120 u).** *HOLD until A1 prints (Fri evening).* If the engine ≥ 20 k/s,
   100 M waits for WP5 (a 1.4–2.8 h job in days); if < 2 k/s, 100 M proceeds on L4 as PLAN_v3. P1 Y3 canary + source (2.00 + 9.62 u) run as
   ratified. Cost of the hold: ≤ 1 day on a chain that cannot reach serving by Nov 7 anyway (round-17 §(d)).
10. **Pricing.** Round-17 stands: A100-40 ≈ 5.4 u/h, A100-80 ≈ 7.5 u/h measured third-party (`colab/SKILL.md:61–67`); forecasts at 3.5× L4,
    caps at 5× (7.7–10 u/h); 2× applied once; H100 unpriced until a live reading. No cap needed by the owner; every manifest still declares one.
11. **Evaluation on GPU (07 round-19, 08).** *Development banks only, and only after ruling 3 path (A) or with per-root CPU rescoring; strict
    and product gates stay on this Mac, exact, paired, one look.*

## (d) Model scale — the owner's question "wouldn't a larger model learn more general?"

**What is measured.** B8 s3 is 1,638,263 params; 64 % of them sit in one `Linear(4096→256)` and 25 % in `Linear(3200→128)`; the two conv towers
together hold ~60 k params and see 7×31×31 and 4×37×37 byte planes (`ego2s_network.py:33–66`; `t8_probes:31–33`). Every distillation /
prediction failure on record failed on **labels, data or distribution, not capacity**: T3 reached 99.64 % decisive agreement and lost
75.1 mass (`t3_diagnostics:17–28,65–69`); D0's corrected losses broke the anchor guards (`CAMPAIGN_LOG.md:100`); T35 had 225 decisive rows
from 21 worlds (`RESULT.md:20–34`); a 294 k-param raw-raster CNN (T8) matched the frozen 1.64 M trunk (0.742 vs 0.765). The only hint that
information is missing is spatial (T8/T10: short-horizon danger is readable, D300 is not; TTV planes clip at 255), which is exactly
where B8 is thinnest. So: **more parameters cannot repair missing information or a bad loss, but the conv towers are a plausible place
where capacity *and* new deployment-visible planes could help together.** That is a hypothesis with a cheap test, not a lever to bet on.

**What the Mac can serve (estimate, to be measured in WP0).** Hero forward is ~11 M MACs and costs 0.34 ms (PyTorch) / 0.14 ms (ONNX) at
batch 1; the 11.688 ms update p95 is dominated by raster/v8 featurisation and the five opponents' own v8 floods, not the hero forward. A
~6.5 M model at ~50–80 M MACs should cost ~0.5–1.5 ms on CPU batch 1 (fits); ~25 M at ~300–400 M MACs ≈ 3–6 ms (does not safely fit under
contention; MPS batch-1 sync and Core ML fp16 identity are unresolved — Astra 07). **Rule:** served size is decided by the WP0 assay
(PyTorch CPU / ONNX / MPS / Core ML fp32; 10 k held-out observations grouped by world; full-update p95 under the live Watch6 game), never
by parameter count.

**Sizes to try.** S = B8 shape (control, 1.64 M). **M ≈ 6.5 M**: conv widths 32/64/64 → 64/128/128 with one extra 3×3 block per tower and the
local flatten kept at 256 (adds ~0.5 M conv + ~2.1 M in the wider flatten; keep the linear bottleneck so params land in the towers), optional
unsaturated TTV/ownership/heading planes (03's representation hypothesis — a separate arm, not bundled). **L ≈ 25 M**: 6–8 residual blocks at
128–256 channels, teacher only. Same scalars, same action interface, same dueling head, categorical/action-ranking loss (03) with B8
retention on non-decisive rows.

**Data each size needs.** No honest universal ratio exists (01, 02, 08 all say so). Use nested doses and read the slope: **2 k roots from
≥ 500 parent worlds, 10 k from ≥ 2,000, then 100 k and 1 M** (08's doses extended), always with ≥ 64 independent rescue-bearing lives
(≥ 16 held out) and all descendants of a parent world on one side of the split. Advance a size only on ≥ 10 % relative reduction of
held-out decisive-root regret vs the smaller size with a world-bootstrap LB > 0 and no > 1 pp rise in harmful overrides (08/01 kill rules).

**Cheapest experiment that shows whether scale helps here (A2, 2 A100-h, this week):** on the existing T5/T8 bank (19,976 states, 103,264
legal branches, 342 worlds, five frozen world folds) and the permitted excluded-A5 diagnostic cache, fit S-class and M-class trainable
encoders against the frozen-linear D300 control, with nested 25/50/100 % training-world subsets and 04's censored multi-horizon hazard
heads. Two numbers decide: the **M − S** held-out D300 AUC/Brier gap, and the **50 % → 100 % data slope**. Flat slope and M ≈ S ⇒ scale is not
the lever on this label type; fund labels of a *new type* (counterfactual, WP2), not more rows or more params. Positive slope ⇒ WP2 label
volume is worth buying and WP4's matched pair on fresh labels is pre-funded.

## (e) First A100 experiments within 24 h (hand to Sol now) — full specs in `../PLAN_A100.md` §3

| # | What | Builder → review | GPU | Base / effective envelope | Kill number |
|---|---|---|---|---|---|
| A0 | E1b event-path repair (fused device-side corpse dispatch with empty-event rejection; food-maintenance early-out; stage timers + CUDA launch counts; RNG stream preserved) | Sol-G (4–8 h) → Codex-V fixtures | — | 0 u | oracle fixture mismatch ⇒ not dispatchable |
| A1 | **Repaired full-tick throughput ruler**: real B8 + 5 opponents + v8 + obs + movement + food/reset + collection writes at E = 6/256/1,024 × 3 mixes, long bodies 800–1,500; 3 warmed repeats × 8 ticks; unrepaired tick once at E = 6 on the same allocation | Claude dispatches; Codex-V reads | A100-40, ~1 h | 1 h / 10 u → **2 h / 20 u** | < 5× full-tick speedup ⇒ event-path hypothesis rejected (trace next); **< 2,000 complete steps/s at E = 1,024** ⇒ engine STOP; 2 k–20 k ⇒ fund the sparse free-space kernel only; ≥ 20 k ⇒ WP2 funded |
| A2 | **Scale + target falsifier** on existing labels (T5/T8 bank + excluded-A5 cache): frozen-linear D300 control vs trainable S vs trainable M, each with single-horizon and censored multi-horizon hazard heads; nested 25/50/100 % world subsets; 2 seeds | Sol-M (1 d) → Codex-V label/censor/split check | A100-40, ~2 h | 2 h / 20 u → **4.8 h / 48 u** | M − S D300 AUC < +0.02 ∧ 50→100 % slope < +0.01 ⇒ scale/volume on this label type closed; hazard arm < 10 % Brier gain ⇒ single-horizon kept |
| A3 | **P1 learner ceiling** (06): cached actor-enabled PPO update time on the frozen 128×128 rollout, three matched modes, phase timers, GPU-busy sampling | Sol-R (4 h) → Codex-V | A100-40 1 h + L4 1 h | 2 h / 20 u → **4 h / 40 u** | cached actor-enabled learning > 1.6384 s per rollout ⇒ the current learner cannot reach 10 k/s even with free collection |
| A4 (Mac, 0 u) | Runtime/identity assay for S/M/L shapes: PyTorch CPU, ONNX, MPS, Core ML fp32; batch 1 and branch batches; 10 k held-out observations; full-update p95 with the live Watch6 game | Sol-K (0.75 d) → Codex-V | — | first quiet gap ≥ 45 min after X5, Fri; never during M1/X5/C1 | any registered overrun or p95 > 11.688 ms for a size ⇒ that size is teacher-only |

Order of dispatch: A0 builds first (it gates A1); A2 and A3 dispatch as soon as their packages pass Codex-V (independent of A0); A1 dispatches
when A0 lands. Three assignments max: A1 + A2 + A3 concurrently if P1 Y3 is not running; else A3 waits. All via `tools/ops/dispatch.sh`
with resume declarations (`AGENTS.md:76–116`), same-pin real-transport canary for the new manifest family (A100 runtime type: expect the
L4 control-path stall census to apply; a stall is INCOMPLETE, not a result).

## (f) Interaction with the Mac-serial chain — what stops, what is deprioritised

- **Unchanged:** M1 attempt-05 tonight (judged retry policy), X5 after M1, M0-B aggregated, **D1 Sat 12:00 (≤ 24 h slip allowed)**, C1 Sat
  12:00 → Sun, D2 Mon 26, C15 build on GO, B3/D3 Tue 27, L1″/L3/L4, strict, product, **KEEP_B8 Nov 10 23:59**. The A100 program uses no Mac
  hours except A4's single quiet gap and must never overlap a leased window; a Sol build that spawns Python on the Mac during M1/X5/C1 is the
  foreign-load class that aborted attempts 03/04 — builders work in worktrees without running the simulator on this Mac during leases.
- **Held:** P1 100 M L4 tranche (ruling 9). **Deprioritised:** B2 transport remedies beyond the supervisor stall fix (WP2 supersedes B2 as the
  root/label source once the engine qualifies; B2's C15-fixture role by Mon 26 is unaffected because the engine cannot be ready by then).
- **Stopped (confirming PLAN_v3 §7 where it still applies):** L4/Colab L2′ follow-ups, Round 0 re-registration/0-D, O2, E14/E16, WP-4, any
  learned component in C15 this campaign, RunPod, quoting "+83".
- **Builders:** Sol-S (C1/B2/B3), Sol-C (C15), Sol-K (M1 → A4), Sol-U (label) keep PLAN_v3 §3 priority. The A100 program gets its own
  builders: Sol-G (engine), Sol-M (models/labels), Sol-R (P1), Codex-V shared. If builder capacity is short, C15 and B3 win: they are the only
  path to a served verdict before Nov 10.
- **Calendar reality:** engine decision Fri Oct 23 (A1) → qualification by ~Oct 30 → first 100 k-root tranche ~Oct 31 → WP3/WP4 fits Nov 1–5 →
  GPU development screen Nov 5–7 → a Mac-chain candidate (new identity: bare student or learned-ordered assisted hero) **after Nov 7**. This
  program is the post-campaign charter round 17 anticipated, started now so that the Nov 10 close hands over a live candidate instead of a plan.

## (g) What must not change

- B8 s3 served with no label change until a strict pass **and** the owner's OK; serving on this Mac (CPU/MPS, W ≤ 4 / K ≤ 8); every
  SUCCESS_CONTRACT bar (pooled ≥ +40 ∧ LB90 > 0, survival band v2, scripted ≥ +0.03, boost [0.5 %, 10 %], zero overruns, update p95 ≤ 11.688 ms,
  product §3); release OK is the owner's alone.
- Honest paired evaluation: paired worlds, fresh collision-audited namespaces, world-grouped splits with all descendants on one side, one look
  per bank, no N shrink, Holm across arms, pre-registered kill numbers frozen before data; exact outcomes sealed until a model is fixed;
  "publish ledger every phase"; verified-ledger resume; 2× applied once; first cap wins; INCOMPLETE is never a pass.
- No learned transition model in any label, fixture or gate; every consumed GPU label CPU-rescored from its action tape (ruling 3); strict and
  product gates exact on this Mac.
- PLAN_v3 §1–§6 (M0/M1/M3/C1/B2 specs, C15 identity, D1–D6, KEEP_B8 triggers, Nov 10 close); round-16/17/18 rulings and prices; P1
  amendment-04 package and its envelope; the three-assignment cap; dispatch only via `tools/ops/dispatch.sh`; never two Mac jobs; never kill
  daemons; `t34-round0-study-01/` closed; `../snake-dqn-l2mac` untouched.
- Nothing here is committed by the judge; the operator logs it (STATUS.md / CAMPAIGN_LOG.md one line) and pushes main.
