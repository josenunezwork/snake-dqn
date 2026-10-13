# M4 Phase R result — B8 without a veto vs frp3-s12 + v8 (2026-10-13)

**Outcome: PARTIAL** (stopped at the final look, n = 32 per seed × mix; `python -I` audit PASS).
Study `redesign-m4-phase-r-nv` (pre-registration `research/redesign_m4_20261013/PREREGISTRATION_M4.md`
on branch `m4b-live`, ratified 2026-10-13 + amendment B), root
`snake-dqn-artifacts/redesign-m4-phase-r-nv-20261013/`, plan sha `72097708…`.

Candidate: arm **B8** = ego2s-bv8 student (ego2s-b raster + 30 v8 per-action safety features as
inputs; no veto at decision time), M4-A setup fixes (mass reward κ 1, kill 0, γ → 0.998, 25 % mixed
training worlds, v8-label margin anchor with on-policy v8 labels), 20M transitions, 5 seeds
(Mac). Selected by `m4_select.py` over {C, B8}: B8 545.0 vs incumbent 434.9 on dev probes;
arm C (no v8 features) 517.5 but failed the scripted survival guard.

| Clause | Estimate | Threshold | Pass |
|---|---|---|---|
| c1 primary mass (H5000) vs frp3-s12 + v8 | **+128.7** (LB +101.5, SE 16.4; seeds +76/+162/+143/+155/+105) | ≥ 20 | yes |
| c2 scripted mass | +151.6 | ≥ 0 | yes |
| c2 scripted survival | −0.003 | ≥ −0.03 | yes |
| c3 frozen / mixed survival | +0.011 / +0.038 | ≥ −0.03 | yes |
| c4 positive seeds | 5 / 5 | ≥ 4 | yes |
| c5 H10000 mass | +74.3 | ≥ 20 | yes |
| c6 guard mass vs champion_a5 + v8 | +136.8 | ≥ 0 | yes |
| **c6 guard scripted survival vs champion_a5 + v8** | **−0.156** | ≥ −0.05 | **no** |
| c7 feasibility | — | — | yes |

**Reading.** The standing goal's efficacy is met with a wide margin: a learned policy with no
decision-time veto gains ~+129 mass over the served frp3-s12 + v8 and is not worse at survival
than it on any mix. The one failure is survival against scripted opponents relative to the
older champion_a5 + v8 (guard, seed 0 bank). No promotion; next lever: scripted-opponent
survival (longer training with the 2.9–3.7× faster actor, and/or a stronger death cost).
