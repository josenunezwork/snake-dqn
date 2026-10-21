# GPU sim spike — dense-search cost per decision vs the L4 CPU teacher (2026-10-20 ~22:45 PT)

Status: development spike, excluded from every registered verdict. Owner-approved brief. Builder: Sol. Operator
(Colab runs): Claude. Cap: **≤ 15 Colab units total**, ~1 day of Sol work.

## Question
Does a GPU-resident rollout engine make a **dense-search teacher decision** at least **10× cheaper (units per
decision)** than the current L4 CPU teacher, at an acceptable, measured divergence from the CPU reference sim?
If yes, a full port and A100/H100 use become worth funding (speed JUDGMENT D6 reopener). If no, stay on L4 CPU and
record why; the A100 verdict (`docs/research/advice/2026-10-20_a100/SYNTHESIS.md`) stays "no".

## Why the scope is what it is (do not shrink it)
1. **Port the whole rollout loop, not one step.** The S0 profile (`research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md`,
   L4 receipt `snake-dqn-artifacts/speed-s0-20261021/l4-a1-output/exports/s0/s0-baseline.json`) puts digest + v8 opponent
   veto at 60–73 % of worker time; B8's forward is ~8–13 %. The grid sim keeps ordered food lists, per-env Python RNG and
   event resolution in scalar Python (`src/simd_env/grid_sim.py` docstring). The spike must run on-device: sim step incl.
   food/corpse/boost-pellet events, the v8 opponent decision (`src/.../v8_nb.py`, `vector61_policy.py`), B8 forward, and the
   dense-search rollout/branch loop. Anything left on the host must be listed and its time measured; if the full loop
   cannot be ported in the time box, report the device-portion result explicitly as an **upper bound**, not end-to-end.
2. **Parity is measured, not chased to bit-identity.** Teacher labels only need to be good: every label is re-scored on
   the CPU reference and every gate runs there. Exact parity matters only for serving (Path B rescues), which this spike
   does not target. Use a counter-based RNG on device if needed; measure divergence vs the CPU reference instead of
   reproducing Python MT streams.
3. **Serving is out of scope.** The served game runs on the Mac backend (no CUDA). The payoff here is teacher/label
   throughput (e.g. dense R1 at scale), not play-time dense search.
4. **Background lane.** Never take a Colab slot from Round 0, Claim A D0, or O1; the account allows 3 concurrent
   assignments. Claude dispatches only when a slot is free. Mac work at `nice -n 10`, ≤ 2 threads; never during a Path B
   quiet window.

## Baseline (frozen before any GPU number)
L4 CPU teacher as measured: ~55 completed teacher worlds per lane-hour, ~0.036 u/world (A5 GO, CAMPAIGN_LOG), and the
S0 per-case stage times. Sol converts this into **units per dense-search decision** on the S0 fixtures (3 mixes, short +
long bodies ≥ 800) using the T3-dense search shape (`docs/research/t3_ceiling_test_2026-10-15.md`), and writes it to
`BASELINE.json` before running anything on a GPU.

## Measurements
- Units per dense-search decision on: Mac (CPU reference), L4 (12-proc CPU, existing engine), L4 GPU port, A100 GPU port
  (price A100 at its measured provider rate).
- Parity on the S0 fixtures vs the CPU reference: first-action agreement of searched decisions, per-branch value
  (mass-integral) error, and a classification of every first-action disagreement (float tie / RNG-stream difference /
  bug). Bugs must be zero.

## Kill rule (frozen before any GPU data)
CONTINUE (recommend a full port) only if **both**:
- end-to-end GPU cost per dense-search decision ≤ **1/10** of the L4 CPU baseline on the long-body mixes, **and**
- first-action agreement ≥ **99 %** of searched decisions on the S0 fixtures, with 0 unexplained (bug) disagreements.
Otherwise STOP: write `RESULT.md` with the numbers and the reason; no second attempt without a new brief.

## Deliverables
New package `research/gpu_sim_spike/` (no edits to frozen/registered source: `research/t34_round0*`,
`research/t34_pathb`, `research/t34_exit_b8*`, `research/t34_round0d`), `BASELINE.json`, a parity harness, a benchmark
entrypoint with a Mac preflight on a tiny input (Colab job preflight rule), Colab manifests via `research/colab_cli`
for one L4 and one A100 run (each ≤ 6 u; **not dispatched by Sol**), `RESULT.md` template. Results go to
`snake-dqn-artifacts/gpu-sim-spike-20261021/`.
