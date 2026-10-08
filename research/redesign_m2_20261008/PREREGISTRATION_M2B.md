# M2b pre-registration: ego2s-b student non-inferiority on fresh worlds

**Committed:** 2026-10-08, before any M2b distillation data or ego2s-b student exists (the
only M2b data so far is the coverage-audit set, data round 9, never used for training).
**Owner decision (2026-10-08):** Option 1 — close the observation gap, regenerate, refit,
probe, then this check with the SAME rule as M2 on FRESH worlds. The M2 FAIL
(`results/ni_check/verdict.json`, pooled Δ −29.5, LB −43.7) stays on record.

Executable rule: `ni_spec_m2b.py` (pinned by `tests/test_m2b_ni_spec.py`).

| Item | Pre-registered value |
|---|---|
| Hypothesis | ego2s-b student **without veto** non-inferior to frp3-s12 **without veto** on dev H5000 mass |
| World / rosters / engine | as M2 (`PREREGISTRATION.md`): pinned config, `promotion-v2-watch-rect`, strict rosters, `run_simd_eval(sim_engine="grid", vector61=True)`, frp3-s12 rowwise, student numba featurizer (`ego2s-b`) + batched CPU forward, greedy over the resolved mask |
| Worlds | 48 per mix from the fresh namespace `redesign-m2b-ni/v1`, shared by all mixes and both arms (144 pairs); disjoint from M2's NI worlds, all distillation rounds (0–2, 9–12), probe worlds (50, 51), M1 identity worlds (tested) |
| Decision | **PASS iff pooled one-sided 90% t LB of (student − frp3) > −15**, all 144 worlds complete |
| Student under test | the fit on data rounds 10–12 (DAgger rounds 0–2 of M2b), selected within the fit by held-out v8 agreement; sha in the intent file before the first NI episode |
| Discipline | no probe-based stopping (probes on reserved worlds, round index 51, are reported, not used to decide whether to run); no tuning on NI worlds; one run; `decide` provenance audit as M2; independent review before the run |
| Reported, not gated | per-mix deltas; student+v8 vs frp3-s12+v8; death causes; survival; boost fraction; held-out agreement on boost decisions |

## Addendum (2026-10-08, before the final M2b fit exists; from the pre-check review)

* **Final fit recipe (fixed now; no probe result can change it):** data rounds 10, 11, 12;
  warm start from `fits/r11/student.pth`; loss = raw-Q Huber + DQfD margin (λ 1, m from
  round-10 data) + advantage Huber ×10 (β 0.1) + softened KL ×1 (τ 0.05); Adam 3e-4,
  batch 1024, cosine, **6 epochs**; the epoch with the best held-out v8 agreement
  (held-out = `world_seed % 10 == 0`) is the student under test.
* **Lineage as executed:** round-10 data by frp3-s12+v8; recipe chosen from held-out
  metrics on distillation worlds (`exp_kl` 0.777, `exp_adv` 0.780, `exp_advkl8` 0.794 vs
  M2 recipe 0.783); round-11 data driven by `exp_adv` (β 0.5); the round-11 fit warm-started
  from `exp_advkl8` (8 epochs on round 10); round-12 data driven by the round-11 fit (β 0.25).
* **Probe seen:** the round-0 student's reserved-world probe (round index 51) was seen before
  this addendum (scripted +23.8, frozen +109.1, mixed −19.8; sd 150–195). It does not decide
  whether the check runs; the final student goes to the check regardless.
* `ni_check.py --spec` is now required; for `m2b`, `intent` refuses unless the student's
  rounds are exactly [10, 11, 12] and its obs spec is `ego2s-b`.
