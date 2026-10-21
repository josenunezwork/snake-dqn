# T34 design v2 — shadow trajectory + last-avoidable-decision search (OWNER-APPROVED 2026-10-19, pre-data)

Status: owner approved switching Path B and the teacher to this design on 2026-10-19 (~21:15 PT), **before
any Path B efficacy data or teacher label data exists** (only Path B plumbing smoke-01 exists; it is excluded
from every estimate). Hardware placement may still change after the pending GPU advice
(`speed_advice/QUESTION_GPU.md`); the algorithm below does not depend on it.

Binds with [SUCCESS_CONTRACT.md](SUCCESS_CONTRACT.md) (unchanged: claims, bars, serving constraints).
Supersedes FINAL_PLAN §3 steps 1–2 (trigger/verify/death-window mechanics) and the Path B v1 protocol
(depth 40, 2-tick deadline), which will **not** be run for efficacy. Evidence: four speed advisors —
`../advice/2026-10-19_t34_speed_astra_{perf,search}.md`, `speed_advice/fable_{systems,search}.md`.

## Why

- B8 CPU forward, batch 1, one thread: 0.29–0.41 ms. A cold H300 verify needs ≥ 92–124 ms of inference
  alone: it can never fit a 42/83 ms tick. Corrected teacher estimate on an L4: ~2.1–3.3k active row-steps/s.
- The world is deterministic, so the greedy continuation from frame t and from t+1 is the same trajectory:
  v1 recomputed ~37–60x redundant work per verify, and expanded all 150 death-window decisions when only
  the neighbourhood of the last avoidable decision (LAD) is decisive (T1: late-death H_LAD quartiles
  ~[16–21, 88–100, 248–254] decisions).

## Mechanism (shared by teacher and Path B)

1. **Shadow trajectory.** Keep one exact greedy continuation of the current policy D frames ahead of the
   real world (D = 300 default, ≤ 500), advancing one row-step per real step. A real step that matches the
   shadow's predicted state (full digest: six bodies, ordered food, all RNG streams, opponent carry incl.
   v8 state, frame/life, checkpoint) confirms it; any mismatch or intervention → invalidate and re-fork
   from the live state (catch-up runs in the background). Verify labels for every decision inside a
   confirmed shadow are free.
2. **Doom → LAD search.** When the shadow's hero dies at t_d, search branch points in [t_live, t_d]:
   bisection on "some alternative action survives to the horizon", then stride-1 refinement near the
   boundary (~6–8 expansions × ≤ 1,500 row-steps). Alternatives in descending B8 Q; branch stops at death
   or horizon. Result = (t*, a*, F, per-action survive/alive-steps/mass).
3. **Action rule (Path B).** "At frame t*, take a*" — scheduled by frame, valid only while the shadow it
   was computed on stays confirmed; then re-fork. A result is stale iff its shadow was invalidated
   (exactly detectable). Late result (t* already passed) → discarded, bare action. One extra thread, ≤ 1
   outstanding request, never blocks a tick (SUCCESS_CONTRACT §4 unchanged). No hindsight rewinds.
4. **Teacher labels.** Confirm labels from the confirmed shadow (dense, free) + LAD-centred labels
   (t*, a*, F) + a sparse sample before the LAD. Same deterministic semantics as T3 dense; exports and
   replay-hash rules from FINAL_PLAN §5 unchanged. Round sizes and units re-derived from M2/A5 below.

## Measure first (Mac, no spend) — gates before rebuilding

- **M1 shadow identity:** shadow beside a live mixed-Watch6 session for 5,000 frames, three mixes; count
  divergences (watch global `random` use in game_state/food_manager). GO = zero unexplained divergences.
- **M2 work/world:** replay the T3 dense bank under shadow + LAD bisection (no new search where avoidable);
  report row-steps/world vs v1's ~0.4M and whether the LAD labels reproduce T3's 54/64 interventions.
  GO = ≥ 5x less work AND ≥ 90% of T3 interventions recovered.
- **M3 one-thread latencies:** shadow step, 300-frame catch-up after re-fork, LAD search per doom; convert
  through T1's H_LAD distribution into the fraction of late deaths reachable in time at 12 and 24 fps.
- **M4 B8 batch-1 runtime:** ONNX Runtime / Accelerate-BNNS vs PyTorch CPU (≤ 10-min probe).

Path B v2 efficacy (contract §5) runs only after M1 and M3; the teacher's A5 benchmark uses the v2 kernel.

## Amendment 1 (2026-10-19 ~22:15 PT, pre-data) — LAD search rule for teacher labels

Astra flagged, and Fable confirmed on T1's 75 late deaths, that rescue availability is **not monotone** in
time (47/75 deaths have a non-surviving decision before the latest surviving one; gap quartiles 6/38/101
decisions). Bisection always returns a surviving branch point, so it stays valid **for Path B intervention**,
but it misses the true LAD by >30 decisions in 9/75, so it is **not used for teacher labels**.

Teacher label rule (T1's validated rule): on shadow doom at t_d, expand at **stride 10** over [t_live, t_d]
(≤ 30 expansions), then **stride 1 within ±5 of every survive→die transition**; label every transition's
decisive decision (a*, F) and take the latest surviving decision as the LAD. Cost ≈ 40–50 expansions ×
≤ 1,500 row-steps ≈ 60–75k row-steps per death (3–4x cheaper than v1's 150 dense expansions, not ~20x).
Shadow confirm labels are KL-anchor rows only, not independent decisive examples.
M2 now measures BOTH rules on the T3 dense bank (work/world, recovery of T3's interventions, LAD error);
GO for the teacher rule = ≥ 90% of T3 interventions recovered.

Hardware (GPU panel, 4 advisors, `speed_advice/`): teacher on **L4 only** (host-bound ~35:1); also probe
the Colab CPU-only runtime (`nproc` + 10-world shadow run) and the free Mac; Path B on Mac CPU; fits: cache
observations first, time one full L4 fit, try A100-40 only if that fit is GPU-bound beyond ~1 h.

## Amendment 2 (2026-10-20 ~01:00 PT, operator decision under owner delegation; pre-data for every new test)

Evidence: M1–M4 results (`research/t34_shadow/RESULTS.md`, shadow branch `f5b45365`): M1 GO; **M2 NO-GO stands**
(registered 90% recovery bar failed: bisection 39/75, stride rule 37/75; not relabelled); **M3 NO-GO** for the serial
continuation path (30–63 s per doom vs 12.5–24.9 s of D=300 lead). Advice: `speed_advice/m2m3_*.md` and
`../advice/2026-10-20_t34_m2m3_astra_perf.md` (Astra perf + Fable search, dissent resolved by Fable judging Astra).
Owner delegated the call ("u make calls on how to proceed", 2026-10-20 00:15 PT).

1. **Labels.** Export the full ranked per-action outcome (T3 rule: keep bare if it survives; else horizon survivors;
   if all die, longest alive; ties B8 Q → bare → index), with horizon/censoring, alive frames and mass. **Decisive**
   rows = horizon rescues only. All-die "delay" choices are tagged and weighted ~0 (the 33 such T3 changes sat in
   worlds that died anyway, +1/2/4 frames). Amendment 1's stride-10 rule is **dropped**; bisection (radius 3) is the
   search rule; label the rescue across the escape island, not only the exact LAD. Shadow confirms stay KL-anchor rows.
2. **Path B v2 search.** No LAD search. On shadow doom, expand alternatives **earliest-first** from the first
   detection decision (all 42 truth-capable T3 intervals were already positive there), batched across alternatives
   with prepared-snapshot reuse and persistent scratch, exact opponent carry/v8, isolated RNG state (no process-global
   `random`), native work releasing the GIL, library pools capped at 1 thread; D = 500. Publish the first fully
   evaluated improving action at a still-future frame; stale/late → bare. Stop advancing a dead future after doom.
   Engineering target (not a gate): first-result p95 ≤ 2 s.
3. **Deciding sequence (no label rounds funded before it):**
   (a) **Pre-check (development, not evidence):** Path B v2 online vs bare B8 on the reused 64-world T3 dense bank
       at real 24 fps pacing; proceed only if ≥ 50/64 survive. ~1.5 h Mac.
   (b) **Registered test — "Path B v2 paced retention":** fresh disjoint namespaces, 32 paired H5000 worlds per mix +
       16 fresh continuous-respawn mixed-Watch6 sessions; arms bare B8 s3 vs frozen successor paced at 24 fps (one
       worker, one request, bounded memory, stale rejection, bare fallback) + the same successor unpaced (descriptive);
       one look, fixed N, Holm over the two active arms, P7 inference. GO / KILL / INCONCLUSIVE exactly as in
       `speed_advice/m2m3_fable_on_astra.md` §4 (GO = every contract bar under deadlines incl. +40/LB90>0, zero
       overruns, update p95 ≤ 11.688 ms). Cap 32 h elapsed / 48 CPU-h, $0, after Phase 0 frees the Mac. Its full
       pre-registration (source SHA, banks, seeds, load profile) is committed before (b) runs.
4. Teacher label rounds (R1+) wait for (b) or Round 0; the teacher v2 build continues as infrastructure with the
   §1 label rule.

### Amendment 2 addendum (2026-10-20 ~13:50 PT, pre-data, operator under owner delegation)
§3(b) sizes and caps are superseded by the power-review resize: **48 paired H5000 worlds per mix (144) + 32 continuous
mixed-Watch6 sessions; caps 40 h elapsed / 60 CPU-h, $0**. The §3(a) pre-check has its own cap (8 h elapsed) and is not
debited against the 40 h. Reason: development-only OC gave joint GO 0.27–0.32 at T3-like effects with 32/16; 48/32
gives 0.80–0.83 (`research/t34_pathb/RETENTION_OC_V2.json`, `VERIFICATION_RETENTION_RESIZE.md`). No retention or
pre-check data existed.

### Amendment 2 addendum B (2026-10-20 ~13:14 PT, before A5 run #2)
The FINAL_PLAN A5 floor (≥ 3,000 active row-steps/s per lane) priced the v1 teacher (~0.4M row-steps/world). The v2
teacher needs ~13–31k row-steps/world (M2; Mac 32-world profile), so A5 v2 is judged on useful output instead:
**GO = ≥ 50 completed worlds/hour per L4 lane** (800-world round ≈ 5.3 h on 3 lanes ≈ 32 u at 2 u/h), measured over
the registered 1,200 s steady window excluding cold fill; < 50 → STOP and report. Row-steps/s stays reported.
Evidence: A5 #1 (serial shadow fill, ~50 row-steps/s/lane, 0 decisions), batching profile
(`research/t34_exit_b8` VERIFICATION_SHADOW_BATCHING.md: byte-identical, Mac −19%, L4 untested).

### Amendment 2 addendum — label ordering (2026-10-20 ~18:20 PT, prospective, pre-data)
Replaces item 4's "Teacher label rounds (R1+) wait for (b) or Round 0" with the SUCCESS_CONTRACT §5 Amendment
(2026-10-20) rule: *R1 teacher-label collection (Colab only, bare B8 s3 + teacher, β=1) may be registered and, once its registration is approved, dispatched after a published Round 0 GO, in parallel with the Path B retention phases. R1/R2 fits, R1/R2 screens, R2 labels and any Mac work for Claim A wait for the Path B disposition (GO/KILL/INCONCLUSIVE decision.json). Round 0 GO authorizes nothing by itself.* Ruling: `docs/research/advice/2026-10-20_a100/RULING_label_ordering.md`.
