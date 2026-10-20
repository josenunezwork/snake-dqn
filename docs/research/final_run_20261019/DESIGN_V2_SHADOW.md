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
