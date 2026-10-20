# Lookahead speed: where the cost is and how fast it could be (Fable, 2026-10-19 evening PT)

Advice only. Read-only; two probes <60 s, 1 thread, nice 15. No repository file edited.

## 1. Measured facts, verified — with three corrections

- `ROUND2_MAC_MPS_LOCKSTEP.json` (1,024 worlds, N4/H32, 3 Numba threads + MPS): 179,500 active row-steps in 42.6 s; sim 113 / obs 27 / opponents 12 / assembly 12 / observe 10 us per active row-step. Correct as quoted, but (a) they are **3-thread wall** microseconds, not one-thread costs (~650 thread-us/active row-step); (b) `sim_step` is paid on all 786,432 **resident** rows (26 us/resident row-step) — a verify-first N=1 teacher has no dormant followers, so its sim term is ~26 us, not 105-113; (c) the "54 us hero forward" is MPS at batch 24k. The CPU twin (`ROUND2_MAC_LOCKSTEP.json`) spent **464 of 504 s in B8**: ~590 us/row.
- Probe (B8 s3, eager fp32, 1 thread): **0.378 ms at batch 1**, 0.35-0.53 ms/row at batch 6-256 (43-64 GFLOP/s); `torch.jit.freeze` no gain; local tower alone 0.32 ms. Phase 0's CPU profile agrees (B8 = 65% of a search; sim 13%).
- Probe (GridBatchSim, short bodies): E=1 **0.24 ms/step** (Python floor), E=48 16.6 us/env-step.
- T3 dense: 55.0 M env-frames / 48,844 worker-s = 1,126/s; 36,996 batches (~1,490 frames each), **75 action changes** (0.2% of batches).
- calibration-v1: H32/N4 0.31, H32/N8 0.38, H64/N4 0.40 agreement — as quoted.
- T1 commitment (late deaths, L>=800): H_LAD quartiles **[21, 100, 254]**; a survivor exists within 30 decisions for 32.7%, within 300 for 93.9%. "50-80 steps" is the median, not the tail.
- Sol's Path B worktree (`snake-dqn-pathb`, PROTOCOL.md): depth **40**, every 5th decision at L>=600, deadline **2/fps**, rollouts step the **live GameState** via cloudpickle (9-13 ms/frame per the 2026-10-03 study). 40 frames ≈ 0.4-0.5 s against an 83-167 ms deadline.

## 2. Where the cost really comes from

Per rollout row-step of one world, first principles: sim = six head moves + O(1) grid lookups, ~1-2 us compiled; hero raster = crops of grids the sim already maintains plus 6-action capped floods, ~10-20 us; five opponents (V61 features + v8 floods + 0.86 MFLOP MLP) ~100-150 us; hero B8 22.5 MFLOP = **0.1 ms at a decent NEON/AMX rate, 0.38 ms in PyTorch CPU, ~1 us on an L4**. So the compute floor on one Mac thread is **~0.25-0.5 ms/row-step, set by the hero CNN and the five opponent policies**, not the sim. Today: ~0.7 ms in the batched teacher (per thread), **~8-10 ms** in the live-engine Path B. On an L4 lane the host floor is ~150 us/row-step on a Mac core (opponents dominate) vs ~650 today: ~4x from kernel work, no more.

The bigger lever is the decision's structure. The world is deterministic (eps 0, cloned food RNG, frame-keyed anchors, deterministic frozen nets), so **the greedy continuation from decision t and from t+1 are the same trajectory**. Every verify at interval 8 (teacher) or 5 (Path B) recomputes 300 steps that the previous verify already simulated; the confirm pass is 37-60x redundant. And the plan's death window expands at all 150 decisions (270k of its 400k row-steps/world) although T1 says the decisive labels are the few around the LAD.

## 3. What I would do

**A. Shadow trajectory (rolling verify).** Keep one greedy continuation running D=300-500 frames ahead of the real world (1 row-step per real step). Death appears at the frontier, D frames early. Verification labels for every decision in between are free. Invalidate and re-fork whenever the real hero's action or the world hash differs from the shadow's prediction (the only ways identity breaks: an intervention, or a batch-shape tie flip of ~4e-5).

**B. Expand once per doomed trajectory, bisect for the LAD.** On a shadow death at t_d, branch at a few points in [t_live, t_d] (bisection on "some alternative survives to D", then a stride-1 refinement): ~6-8 expansions x <=1,500 row-steps ≈ 10k per death instead of 225k. Label the LAD-neighbourhood decisions (a*, F, per-action outcomes) and a sparse sample before it. Branches stop at death or D; run them in descending Q when only a* is needed.

**C. Kernel work, second order, only if A+B are not enough:** Accelerate/BNNS or ONNX-Runtime CPU graph for batch-1 B8 (expect 2-4x on 0.38 ms); scalar numba world-step for E=1 (0.24 ms -> ~20 us); on L4 lanes, fuse V61+v8+MLP per world into one kernel.

## 4. Estimates

- **Teacher (L4 lane / Mac cores):** per world 0.4 M -> ~30-50k active row-steps (shadow ~5k over a 5,000-frame world, deaths ~10-20k each, a few sampled expansions). At today's expected 3.5-5.5k row-steps/s/lane that is **~250-500 worlds/h/lane**, not 30-50; an 800-world round is ~2 h per lane. Same deterministic label semantics; confirm labels become denser, window labels become LAD-centred (the ones with Q-gap <2 at 87% of LADs).
- **Path B, one Mac thread, live Watch6:** shadow 1 frame/tick ≈ 1.5-2.5 ms with GridBatchSim + CPU B8 + batched opponents (~10 ms with the live engine) — 4-25% of a spare core at 24 fps. Catch-up to a 300-frame lead after a re-fork ≈ 0.5 s (3 s live engine). Five-branch expansion at E=5 ≈ 1-1.5 s (≈15 s live engine). Detection lead is ~300 frames = 12.5 s at 24 fps / 25 s at 12 fps. The binding constraint is **search latency vs (t_LAD - t_live)**, which is >=46 frames (1.9 s at 24 fps) for 75% of late deaths; the 42/83 ms tick deadline never binds. Action rule: "at frame t*, take a*", scheduled by frame, then re-fork; a result is stale iff the shadow was invalidated — exactly detectable, never guessed.

## 5. Measure first (hours, Mac, no spend)

1. Shadow identity: run a GridBatchSim shadow beside a live mixed-Watch6 session for 5,000 frames; count divergences (T12 check 1 predicts zero; the 2026-10-03 study notes there is no live-GameState -> BatchSim loader, so start from the seed and replay actions, or keep Sol's cloudpickle engine for the shadow).
2. Row-steps/world under shadow+bisection by replaying the T3 dense bank (no new search); confirm ~10x and that the LAD-centred labels reproduce the 54/64 interventions.
3. Three latencies on the spare thread: shadow step, 300-frame catch-up, 5-branch expansion; convert through T1's H_LAD distribution into the fraction of late deaths reachable.
4. B8 batch-1 on ORT/BNNS, 10-minute probe.

## 6. Flags

- **The Path B deadline check as built will fail for reasons unrelated to lookahead's value**: depth 40 reaches ~1/3 of T1's avoidable late deaths before any deadline, and the live engine misses a 2-tick deadline on essentially every request. Under SUCCESS_CONTRACT §5 that rejects Claim B. Re-freeze the protocol to the shadow design before efficacy data exists.
- The "~50 ms verify / ~260 ms expand" figures are 3-thread wall numbers with MPS; on one CPU thread verify is ~0.3 s (GridBatchSim) to ~2.5 s (live engine).
- FINAL_PLAN's 8-13 dec/s and 30-50 worlds/h/lane are right for its mechanism but the mechanism is ~10x wasteful; the A5 floor (3k active row-steps/s/lane) can stand, the worlds/h and round sizes should be re-derived.
- Cross-platform float ties (~1/50 episodes) break shadow identity Mac<->L4; the teacher's per-world hash-drop handles it, and live Path B is single-machine.
