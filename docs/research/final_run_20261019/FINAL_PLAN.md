# FINAL PLAN — "ExIt-B8" expert iteration on Colab L4 lanes (lead advisor, 2026-10-19, post red-team revision)

Supersedes MERGED_PLAN.md. Every red-team fatal issue is either adopted or answered below (Section 11).

## 1. Decision

Run one expert-iteration campaign: a **deterministic verify-then-expand exact-sim teacher** (H=300, the T3-dense mechanism that is the only lever ever to beat B8 by a large margin) labels the student's own trajectories, and the labels are projected into the B8 CNN by supervised distillation with a tight KL anchor and no TD term. Hardware = up to three Colab L4 lanes (host-bound numba sim, all forwards on the GPU). **Funded now: prep + a ~2-unit L4 benchmark/two-lane rehearsal.** Round 0 (~16 u) is funded only when (a) the benchmark meets the active-row-step floor, (b) the owner has published the input checkpoints, and (c) the owner has answered the three pre-decisions in Section 8. R1+R2 (~145 u) are funded together after Round 0 GO; R3 only on an R2 improvement. Hard cap ≈ 270 priced units (~23% of ~1,160), expected observed 150–200. Verdict ≈ Nov 6–8, not Oct 28.

## 2. Why this beats the current direction (evidence)

- T3 dense (`t3-rollout-labels-20261015/dense/ceiling-result.json`): B8+teacher 54/64 scripted survival vs 38/64, alive-frame 0.942 vs 0.861, mass 700 vs 608, while changing 75 of 301,589 decisions. No bare-network lever since B8 (M5, M6, T6, T23 −80.8, FRP-v4/v5) comes close.
- The N8/H128 stochastic teacher Phase 0/T33 are built around is noisy: `calibration-v1.json` two draws agree on the argmax 98/300, 23/61 on clear-top decisions, median top-2 gap 0.6%. Distilling it distils noise; the deterministic verify-then-expand teacher has no such variance and costs ~4–10x less per decision.
- T3's post-mortem (corrected per red team): the students *learned* the bank (99.6%) and retained labelled behaviour (89–92% survivor retention); they lost through behavioural spillover (boost-mode disagreement at all lengths) and a late-growth deficit. The fix is therefore the **anchor and tripwires** (τ=0.05, margin 0.1, boost cap, all-length disagreement), not more data. On-policy DAgger and the full-arena tower are secondary, ablated arms.
- GPU ports (h100/a100) are 2–4 weeks and >500 u to the first useful unit; PPO from B8 (rl_selfplay) is the known failure mode (T23, Q0, M3-C).

## 3. The run

**Algorithm (teacher T_r on student S_{r−1}; S_0 = B8 s3, SHA 3e649783…).**
1. Triggers: (i) confirm pass — every **8th** decision at L ≥ 600 (interval relaxed from 4: confirmations carry little label information and are the KL rows anyway); (ii) death window — the 150 decisions before any student death, replayed from a snapshot at interval 1, teacher acts where it differs, **≤ 2 rewinds per world**; (iii) 3% uniform sample below L600.
2. VERIFY: one deterministic H=300 continuation of the student's greedy action (eps 0, real opponents, cloned food RNG). Survives → confirmation label.
3. EXPAND on failure: all other legal actions in lockstep, H=300. a* = highest-student-Q survivor (ties → student action); none survive → longest-surviving. Forbidden set F = actions that die while a* survives. Per-action (survive, alive-steps, mass) recorded.
4. Teacher-acts β: R1 = 1, R2 = 0.5, R3 = 0. **R1 is acknowledged as off-policy** (roots are B8+teacher trajectories); R2 is where the on-policy thesis is tested, so R1 is non-terminal (Section 8).

**Per-world cost, costed at the red team's numbers:** confirm pass ≈ 310 triggers × 420 = 130k active row-steps; death window ≈ 2 × 150 × 1,800 (100% expansion assumed) = 270k; total **≈ 0.4 M active row-steps/world**, ≈ 600 searched rows/world. Expected decisive rows: T3's flip rate (~1.2/world) gives ~1k flips per 800 worlds, plus the death windows (~300 decisive rows/world, dominant). The claim that this beats T3's 75 flips rests on the death-window rows being on the student's own failure states, not on flip count; the R1/R2 screens test exactly that.

**Student fit** (generalised `research/redesign_m2_20261008/train_student.py`, one L4):
- Rows: all searched decisions + 1:1 unsearched on-policy rows; aggregated DAgger bank (1.0 current / 0.5 older). Death-window rows weighted **per world** (each window sums to the weight of 10 rows), decisive rows 20x within that.
- Loss = KL(softmax(Q_prev/τ) ‖ softmax(Q/τ)) with **τ = 0.05** (M2b precedent; τ=1 is vacuous at B8's 0.06–0.45 Q gaps) + hinge Σ_{f∈F} max(0, Q(f) − Q(a*) + **0.1**) + small advantage-MSE. Empty F → no hinge. A4 sweeps τ ∈ {0.02, 0.05, 0.1}, margin ∈ {0.05, 0.1, 0.25} on a replayed T3 bank.
- Arms: **S1b (primary): no tower, no aux heads** — minimum-change. S1a: full-arena tower + hindsight log-free-region head (Huber 0.1; the survive-300 head is dropped, T8 ≤ 0.74 AUC). Both fit on identical labels; R2 continues with the R1 winner.
- Adam 2.5e-5 (pretrained) / 1e-4 (new), batch 1,024, 4–6 epochs; epoch chosen by held-out decisive agreement + 32-world survival probe (pre-registered, never gate data). No TD.

**Observation/network.** ego2s-bv8 unchanged. S1a adds 4 uint8 full-arena planes (152×88, own TTV / other TTV / food / heads) and a 3×3 stride-2 conv tower 16-32-64-64 → 256, fuse columns zero-initialised (max|ΔQ| == 0 vs B8 at step 0). Serving +3–5 ms, +53 KB plane per tick; `LiveFullSight` with sim/live byte-identity tests; new loader arch id.

**Simulator & throughput (rewritten in active row-steps).** Exact numba GridBatchSim on the host; no GPU sim port. The merged plan's "40 µs/row-step" amortised host cost over 196,608 resident row-steps of which only 44,844 (23%) were live (`ROUND2_MAC_MPS_LOCKSTEP.json`); verify (N=1) has no dead followers. Planning number: **~175 µs host per active row-step per Mac thread** (sim 105, obs 28, opponents 25, assembly 12, observe 11), i.e. ~5.8k/s on 3 Mac threads; T3 dense's 1,126 env-frames/worker-s independently agrees within 2x. Per L4 lane (6 physical cores, 2 processes × 5 threads, 2.4–3.7x slower per core, 30% derate): **expected 3.5–5.5k active row-steps/s ≈ 8–13 dec/s at 420 row-steps/decision**, ≈ 30–50 worlds/h. The old 20–80 dec/s bracket and the 15 dec/s go-rule are withdrawn. Resident bytes 356 KB/row (not 80 KB); RAM is measured in A5.

**Opponents/curriculum.** Gate-matched rosters, fresh collision-audited namespaces disjoint from Phase 0/S1/T3: 50% scripted, 25% frozen (frp3-s12+v8, champion_a5+v8), 25% mixed; 6 snakes; H5000. No league/PBT/restart curricula. Curriculum = DAgger + β anneal.

**Bootstrap.** B8 s3. If S1 reads GO tonight and passes a Mac screen, its final becomes S_0; nothing else changes.

## 4. Hardware & GPU utilisation

L4 only (`allowed_gpus=[L4]`, refuse substitutions; A100/H100/G4 buy the same 6 cores at 3.5–5x the units). Per lane: 2 teacher processes × 5 numba threads, W = 256 root worlds each, hero forward on the GPU in **fp32** (labelling determinism; fp16 only if A2 passes with it), frozen vector61 opponents batched on the GPU. GPU busy will read 10–25% during labelling and 40–60% during fits — **the utilisation metric is active row-steps/s per lane**, not GPU %. Mac: S1 eval tonight, Phase 0 (6 workers) to Wed Oct 21, then A2 regeneration, screens (~1 h per 96/32/32 bank) and the 7–9 h strict gate. Mac on AC, lid open, caffeinate, for ~2.5 weeks. Mac workers after Phase 0 ≈ one L4 lane at active cost.

## 5. Colab operations

- All jobs via `research/colab_cli` unchanged (no re-acceptance owed) — except if `max_export_bytes` > 64 MiB is needed (it is not: ~55 KB/world → 100-world chunk ≈ 5.5 MB).
- **Chunk sizing from the benchmark:** chunk worlds = floor(2.5 h × measured worlds/h/lane) (≈ 75–125 worlds), `est_minutes` ≤ 180, `timeout` 360, export per chunk; satisfies `timeout ≥ 2×est` and the unit cap. Unique (round, shard, attempt) namespaces; no auto-retry.
- Exports per world: seed, **all six snakes' action traces** (30 KB), food/scripted RNG stream ids, **sim-state hashes every 250 frames**; per searched decision: index, a*, F, 6 × (survive, alive-steps, mass) fp16. The fit job regenerates obs by replay and **drops any world whose hashes diverge** (divergence is detected, never silently mislabelled); drop rate reported per chunk, kill at > 2%.
- Fit job = **streaming replay-regeneration loader per epoch** (1 M rows × 65 KB does not fit RAM); regen time per epoch measured in A5b, budgeted ≤ 30 min × 5 epochs.
- Concurrency: A5 doubles as the **two-concurrent-lane rehearsal**; lane 3 opens only after two lanes complete one chunk cleanly. Never start beside S1's dispatcher (exits ~8 PM PT). Lease 600 s. Telemetry every 2 min: active/resident row-steps/s, expansion fraction per mix, worlds/h, boost share, flips/world, RAM.
- Recovery: lost chunk (≤ 6 h, ≤ 12 u) rerun under a fresh attempt id; 36 u reserve.

## 6. Budget and wall time (priced 2.0 u/h at the **floor** 3k active row-steps/s/lane; observed ~1.54)

| Item | Lane-hours | Priced u | Gate |
|---|---|---|---|
| A5 benchmark + two-lane rehearsal | 2 × 0.5 | 2 | now |
| Round 0: 240 paired worlds (96 scripted/72/72), B8+teacher acting vs bare B8 | 3 × 2.6 | 16 | A5 GO + owner pre-decisions |
| R1 labels, 800 worlds | 3 × 9.9 | 60 | R0 GO |
| R1 fits S1a + S1b | 2 × 4 | 16 | with R1 |
| R2 labels 800 + fit (one arm) | 3 × 9.9 + 4 | 68 | funded with R1 |
| R3 (conditional) | same | 68 | R2 ≥ R1 + 15 mass or + 0.03 survival |
| Rerun reserve | — | 36 | — |
| **Hard cap** | | **≈ 270** | expected 150–200 |

At the expected 4.5k row-steps/s each label round is ~6.7 h / 40 u. If the benchmark lands ≥ 8k, rounds grow to 1,200 worlds at the same cap.

Wall time: Sol prep Oct 20–28 (7–9 days; S1 needed 10 acceptance runs and 4 dispatches for an existing trainer); A5 as soon as the engine phases exist (target Oct 23); publication + banks + preflight by Oct 28; Round 0 Oct 29; R1 Oct 30–31 (labels ~10 h, fits, Mac screen); R2 Nov 1–2; R3 Nov 3–4 (conditional); strict gate Nov 5; serving qualification Nov 6. **Verdict ≈ Nov 6–8.**

## 7. Preparation (Sol implements in two worktrees; Claude reviews and operates; ordered)

| # | Work | Acceptance | Kill / hold |
|---|---|---|---|
| P0 | **Owner:** publish B8 s3 + six frozen checkpoints + parity reference to a public GitHub release (T33 `INPUTS_REQUIRED` is PUBLICATION_BLOCKED) | SHA-256 match | no publish → no Colab run |
| P1 | Engine: verify-first (N=1, H=300) and expand phases, snapshot/rewind death windows, teacher-acts mixing, active-row compaction, exports with 6 traces + hashes | **A1 sim-identity:** same seeds → identical outcomes across two runs/processes; T3 receipt agreement reported, not gated | — |
| P2 | **A5 benchmark (2 u):** the actual H=300 kernel on real root worlds, all three mixes, exact 2×5-thread layout, student-with-tower (zero-init) as rollout policy, two lanes concurrently | active row-steps/s, active/resident fraction, expansion per mix, RAM, export size, worlds/h | **< 3k active row-steps/s per lane → STOP, report** |
| P3 | A2 cross-runtime replay: label on L4 fp32 → regenerate on Mac and on a second L4 session, ≥ 150 worlds, three mixes | zero hash mismatches | any mismatch unexplained → hold |
| P4 | Fit harness: KL τ, hinge/abstention, per-world death weights, aux masking, streaming regen loader, probe-based epoch rule | **A4:** tower off + labels = B8 argmax reproduces B8 within 1e-5; τ/margin sweep on replayed T3 bank reports all-length disagreement + boost share | — |
| P5 | Tower + `LiveFullSight` + loader arch id + serving path | **A3:** max|ΔQ| == 0 on 10k states; sim/live planes byte-identical on the P2 bank; export strips heads | S1b proceeds if the tower slips |
| P6 | Three Colab entrypoints + manifests, namespace banks | **A6:** Mac preflight of each entrypoint (2 worlds, 500 frames, 200 updates) | untested entrypoint never enqueued |
| P7 | Pre-registration (banks, rules, caps, selection, Section 8 decisions) | document committed | — |

## 8. Evaluation and promotion vs B8

**Owner pre-decisions, required before Round 0:** (1) publish checkpoints (P0); (2) does **Path B** (served asynchronous verify-only lookahead on the Watch hero, ~12–30 ms on a spare Mac core) count as success? It contradicts the 2026-10-08 "no veto" standing decision and holds ~0.6 of the plan's upside; (3) is a **Watch6-only release** acceptable (B8 already failed Watch12 at T12; the tower adds 3–5 ms)?

**Round 0** (re-run of the Phase 0 premise with the deterministic teacher, interval 4, teacher acting; per-world-step paired comparison with 300-step look-ahead credit): GO = scripted no-death ≥ 0.66 AND ≥ +0.10 vs paired bare B8 AND pooled mass ≥ +40 with LB90 > 0 AND boost share ∈ [0.5%, 10%]. STOP → B8 is reported as the bare ceiling; Phase 0's verdict is a cross-check.

**R1 screen** (S1a, S1b vs B8; 96 scripted + 32 frozen + 32 mixed paired H5000; add both-alive length-delta per T3 diag §2): R1 is **non-terminal** — R2 runs on-policy (β=0.5) on the better arm unless a tripwire fires. **R2 screen** gates R3 and the gate. **Tripwires** (kill that fit): non-finite loss; > 90% one-action share; disagreement vs previous student > 5% at **any** length decile on a full H5000 trajectory; boost share > 1.5x the previous student's; held-out decisive agreement < 70%.

**Promotion:** pre-registered selection = largest LB90 pooled mass subject to scripted survival ≥ B8 + 0.03; then the Mac strict gate (Tier 1 + Tier 2, survival band v2 pooled_ni_continue) vs served B8 s3 on fresh banks; then serving qualification (live identity, no veto, rollback matrix, Watch6 path). Release = STRICT_PASS + SERVING_PASS + owner OK; rollback `SNAKE_SERVE_CHECKPOINT=b8-s3`.

## 9. Risks and early kill signals

- **Throughput** below 3k active row-steps/s per lane (A5) → stop before any round; 3–5k → 800-world rounds, ≤ 2 rounds unless R2 improves.
- **Distillation spillover repeats** (T3): detected at R1 by the all-length disagreement and boost tripwires, bounded at ~90 u.
- **Silent replay divergence** (Mac↔pod float ties ~1/50 episodes): fp32 labelling, cross-runtime A2, per-world hashes with drop-and-report; > 2% dropped worlds in a chunk → stop and investigate.
- **Round 0 GO fails to replicate** at 240 fresh worlds (precedent 64) — the designed off-ramp.
- **Prep slips** past Oct 28 → hold Round 0; never dispatch untested jobs.
- **Fit memory/time**: streaming loader; regen > 45 min/epoch → reduce epochs, not rows.
- **Three concurrent lanes unproven**: lane 3 only after the two-lane rehearsal.
- **Unbounded rewinds**: ≤ 2 per world, counted in the per-world budget.

## 10. What to stop / what to let finish

Let **S1** finish and read once (eval tonight on the Mac); let **Phase 0** finish untouched (Wed) as a cross-check. **T33**: stop host optimisation toward N8/H128 and the stochastic-teacher L4 benchmark; keep its engine, admission and export code as the base for P1. Stop: GPU/JAX sim ports, A100/H100/G4 requests, G4/TPU nproc probes, bare-B8 TD fine-tunes after S1, league/PBT/restart curricula, M7 diagnostics, RunPod, interim peeks. Phase 0 interim claims ("42% boost switches", "−254 mass") are **withdrawn** — they trace to no artifact.

## 11. Panel record

- **Proposals (ranking.json):** a100 4.5 (GPU viable, Colab viable, RL weak); rl_ongpu 4.0 (one review, Colab weak); h100 3.67 (weak ×3); rl_search, colab, game, l4g4, rl_selfplay unreviewed — **17 of 24 critic reviews were lost to a session limit**, so the merged core (rl_search + colab + game, with grafts from rl_ongpu/l4g4/a100) reached the red team with zero surviving critic reviews. The four red-team reviews above are the critic pass on the merged plan; all four returned **fund-with-changes**.
- **Adopted from the red team:** active-row-step cost model and withdrawn dec/s bracket; go-rule re-based; rounds resized to 800 worlds and chunks sized from the benchmark; cap restated at the floor (~270); full six-snake traces + RNG ids + per-world hashes; fp32 labelling; cross-runtime A2 ≥ 150 worlds; streaming fit loader; τ = 0.05, margin 0.1, per-world death weights, all-length/boost tripwires, both-alive length delta; R1 non-terminal; survive-300 head dropped, S1b primary; A1 as sim identity; rewind cap; owner publication and Path B/Watch6 as explicit decisions; 7–9 day prep; Phase 0 interim numbers withdrawn; nproc probes dropped.
- **Where I hold a different view:** (i) RT1's "near-death expansion → ~100%" is an upper bound (early-window actions mostly survive); I budget at 100% anyway. (ii) RT2's cap arithmetic kept 2,400 worlds; resizing rounds is the better fix than raising the floor to 30 dec/s, which the active model says no lane reaches. (iii) RT3 asks for β = 0.5 from R1; I keep β = 1 in R1 (the 0.8-survival regime must be in the bank) and remove R1's kill power instead. (iv) RT1 funds only A5; RT3 would fund Round 0 now — Round 0 cannot run before P0 anyway, so it is gated, not funded.
- **Open disagreement for the owner:** whether a served lookahead (Path B) is a legitimate success; the panel's own numbers put most of the expected value there.
