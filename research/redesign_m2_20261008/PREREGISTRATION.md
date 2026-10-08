# M2 pre-registration: ego2s student non-inferiority (redesign scope)

**Committed:** 2026-10-08, before any ego2s student, any distillation data, or any
NI-world episode exists. **Owner decision (2026-10-08):** adopt the doc section 15.3
proposal (48 worlds per mix, margin −15). Development check, not gate evidence.

The executable rule is `ni_spec.py` in this directory (pinned by `tests/test_m2_ni_spec.py`).
In words:

| Item | Pre-registered value |
|---|---|
| Hypothesis | ego2s student **without veto** is non-inferior to frp3-s12 **without veto** (sha `eec144bf…3723`) on dev H5000 mass integral |
| World | pinned deployment config (sha `4146baa3…15aa5`), profile `promotion-v2-watch-rect` (digest `d396d3ed…0e8b`), H5000, terminal hero, opponents respawn |
| Rosters | strict balanced rosters (`dev_screen._design_rows`), mixes frozen / scripted / mixed; opponent pool = `dev_screen.POOL` + scripted anchors |
| Worlds | 48 per mix from namespace `redesign-m2-ni/v1` (`ni_spec.ni_seeds()`; first three 2643297547, 2781215250, 3530758914), the same 48 seeds for all mixes and both arms: 144 paired worlds. Disjoint from the distillation namespace `redesign-m2-distill/v1` and the M1 identity seeds (tested). |
| Engine | `run_simd_eval(sim_engine="grid", vector61=True)`; frp3-s12 rowwise (bit-exact) forwards; student = ego2s numba featurizer + batched CPU forward (1 torch thread), greedy over the resolved action mask |
| Statistic | paired deltas `d = student − frp3`; pooled one-sided 90% Student-t lower bound `mean − t(0.90, n−1)·sd/√n` |
| Decision | **PASS iff LB > −15** and all 144 worlds completed; otherwise FAIL (or INCOMPLETE) |
| Student under test | the final (round-2) DAgger fit's checkpoint selected within that fit by held-out v8-action agreement on distillation worlds; its sha256 is written to the check's intent file before the first NI episode |
| Reported, not gated | per-mix deltas and LBs; student+v8 vs frp3-s12+v8 (I) on the same worlds; death causes; peak lengths; v8 activation counts on the student |
| Discipline | no tuning on NI worlds; one run; any re-run is reported alongside the first; an independent review precedes the run |

Rationale for the margin: on the 24 M1 identity worlds frp3-s12 without veto averaged
113.0 H5000 (with v8: 421.9), so the doc's earlier −30 (8% of the vetoed 399) would be a
27% margin; −15 is ≈13%. With a paired sd of about 100 the standard error is about 8.3,
roughly 70% power at a true difference of 0.
