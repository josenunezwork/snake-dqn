# E1 Mac preflight and undispatched L4 handoff

Development experiment only, excluded from every registered verdict. No serving
changes. No Colab/provider calls were made. Mac checked on AC, lid open, no
reported thermal warning; Phase 0's existing session was left intact. Load average was 12.86 and
CPU idle 25.81% before qualification. This is contended runtime evidence, never
Path B latency evidence. Existing tmux sessions `colab-limit-test`, `git-sync`,
`phase0-selection`, and `t34-round0-c002` were observed and left untouched.

## Frozen identity

- E1 rule freeze, before E1 tests/preflight/data:
  `896310e38a97e30c7a8f9247b68a5c054a615b88`.
- Kernel/harness/generator implementation:
  `cfbee66a26d1059bae87afeea58d194e41842e3d`.
- Measured source and both manifest pins:
  `10ce6416ac046bc25c5cdcb167b5d386bc407475`.
- Rule SHA256: `a49aa11384a2636319d6eef80d6a38943bf759c7d14ea3804d39da0e5c17765a`.
- Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1/mac-entrypoint-01/output/benchmark.json`.
- Receipt SHA256: `b29c6daec5474045a388b5037da85f7c80ea026ca85d702edcccbe2da16bb7b1`.

The original cost/agreement criteria remain unchanged. E1 adds the frozen
<20k full world-steps/s on any L4 long mix => STOP engine thesis this week rule,
and a CPU-defined decisive-root audit. Zero decisive samples is NO_COVERAGE.
The historical BASELINE.json wording about both GPU receipts predates E1;
the E1 brief requests an L4 spike and optional L4 probe, with no A100 run.

## Actual real-entrypoint qualification

Ran the real `colab.sh` with uploaded-input-style links to the six immutable
local checkpoints. Tiny = 16×12 cells, six real snakes, three mixes, six fixed
branch lanes, D2, one decision per mix. Python 3.12.3 / Torch 2.9.1 CPU build.
`PATH` placed the primary venv first; no new environment was created.

```bash
nice -n 10 env \
  PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH" \
  COLAB_CLI_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1/mac-entrypoint-01 \
  COLAB_INPUT_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1/mac-inputs \
  GPU_SIM_SPIKE_COMMIT=10ce6416ac046bc25c5cdcb167b5d386bc407475 \
  GPU_SIM_SPIKE_DEVICE=cpu GPU_SIM_SPIKE_HORIZON=2 \
  GPU_SIM_SPIKE_FIXTURES=tiny GPU_SIM_SPIKE_SECONDS=240 \
  GPU_SIM_SPIKE_THREADS=1 GPU_SIM_SPIKE_CPU_WORKERS=1 \
  GPU_SIM_SPIKE_DECISIONS_PER_CASE=1 GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR=2 \
  SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  bash research/gpu_sim_spike/colab.sh
```

This command was executed through a Python subprocess wrapper that opened its
log create-only at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1/mac-entrypoint-01.log` and measured outer wall time.
The command above describes the completed run; do not reuse its immutable outputs.

| Measured stage | Seconds |
| --- | ---: |
| Outer process wall, including imports | 8.642362 |
| Benchmark elapsed | 7.461215 |
| Entrypoint run including result construction | 7.552676 |
| Candidate pipeline, setup/warmup/readback/replay included | 1.001712 |
| Synchronized rollout loops, warmup excluded | 0.258694 |
| Rollout seconds per dense decision | 0.086231 |
| Independent exhaustive CPU reference search | 6.435547 |

36 scheduled full world-steps / 0.258694 s =
**139.161 full world-steps/s**;
pipeline rate **35.938/s**.
Each world advances all six snakes; there is no extra ×6 snake-row multiplier.
Per-mix rollout rates: scripted 159.496, frozen 123.192, mixed 139.457 /s.
These are tiny CPU rates; the L4 floor status is NOT_L4_DATA.

First-action agreement **3/3**, unexplained **0**, branch mass MAE **0**.
Decisive roots **0** / NO_COVERAGE, agreement null; other roots 3/3.
This sample does not qualify the D300 parity/cost rule or the L4 throughput floor.

## Timing extrapolation and forecast limits

The measured tiny schedule costs 0.043116 seconds per dense
root-depth unit, including fixed root overhead. Linear depth/sample scaling on
this SAME tiny geometry predicts 232.824 s (3.880 min) of
rollout time for 18 D300 decisions, or 2.070 s for six D8 decisions.
This is an explicit workload extrapolation, not an L4/full-board completion
forecast: S0 has 145×83 = 12,035 cells versus 192 here (62.68× more cells), long
bodies, different flood graph diameters, GPU launch/synchronization costs, and a
12-process CPU comparator. A linear area multiplier or an assumed GPU speedup
would not be a measured estimate. Bootstrap/export/stop are also absent locally.

Manifests therefore state bounded allocation planning timeboxes, with substantial
headroom and no promise the full bank finishes. Incomplete means no CONTINUE.
At 2 u/h, spike planning 190 min = 3 h 10 min / 6.333 u charged (plus 1.5 u
reserved); wall cap 400 min = 6 h 40 min / 13.333 u charged, within 13.5 execution
cap + 1.5 reserve = 15 u envelope. Optional probe planning 60 min = 1 h / 2 u
charged (plus 0.5 u reserved); wall cap 135 min = 2 h 15 min / 4.5 u charged,
with 0.5 reserve = 5 u envelope. First wall/unit cap wins. The spike's benchmark
itself is capped at 180 min; the probe at 55 min. No automatic retry.

## Targeted verification

52 passed, 77 unrelated supervisor/runtime tests deselected, 8.34 s:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/gpu_sim_spike tests/test_colab_cli_supervisor.py \
  tests/test_colab_cli_runtime.py \
  -k 'gpu_sim_spike or existing_output_directory_is_refused_as_nonfresh or partial_local_export_without_completion_marker_is_rejected or read_state_returns_only_completed_hash_verified_exports or export_cache_hashes_once_and_rejects_later_mutation'
```

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/gpu_sim_spike
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --profile black --check-only research/gpu_sim_spike
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/gpu_sim_spike
bash -n research/gpu_sim_spike/colab.sh
git diff --check
```

All passed. The independent deque-BFS adversarial tests cover winding corridors,
160/4096 caps, delayed release with no waiting, singleton edges, repeated rows,
dead/OOB/blocked starts, and an 8×8 map that breaks an unsafe logarithmic bound.
Ego retains the proven V−1 bound with early convergence. On actual tensors the
flood convergence predicates synchronize a scalar every eight rounds; that cost
is included in measured rollout and launch CPU wall. Bulk state stays on-device.
Meta/FakeTensor checks retain static bounds and establish shapes only.

The export test asserts .complete + .sha256 and uses the actual supervisor
export discovery/checksum path. All four classes of runtime JSON use the same
create-only writer: CPU-before-device, individual decisions, benchmark and host
export. The launcher verified `benchmark.json: OK`. Receipt sidecars remain
next to every JSON. Independent read-only review found no remaining material
kernel/harness issue; real CUDA numerical parity and performance remain untested.

## Local-only manifest handoff

Artifacts root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1`.

- `l4-manifest.json`: full S0 D300 spike, 15 u including reserve.
- `probe-manifest.json`: optional S0 D8 diagnostic probe, 5 u including reserve.
- `DISPATCH.txt`: exact local validation commands and Claude's dispatch duties.
- `l4-local-validation.json`, `probe-local-validation.json`: local supervisor output.
- `l4-source-plan/`, `probe-source-plan/`: local immutable source plans/fallbacks.

Both drafts pin measured source, all six checkpoint SHA256s, the real preflight
receipt and frozen rule digest. Both were generated using:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.gpu_sim_spike.make_manifests \
  --preflight /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1/mac-entrypoint-01/output/benchmark.json --validate
```

Each was additionally passed through `python -m research.colab_cli.supervisor
--state-dir <owned lane> validate <manifest>` locally. Empty other_assignments is
a draft placeholder, never current account evidence. Claude refreshes inventory,
positive rates, qualified transport/source admission and priority before dispatch.
No Colab calls, allocation, status, usage, auth, dispatch or stop occurred here.

Only E1 package/docs were authored. Frozen/registered source, research/colab_cli,
CPU oracle, released checkpoints/pins, live jobs and the forbidden study artifact
directory were untouched. The automatically generated published integration merge
`10ce6416` lacks the requested coauthor trailer; all authored E1 commits carry it.
Published main history was preserved rather than rewritten.

## Artifact identities

| Artifact | SHA256 |
| --- | --- |
| `l4-manifest.json` | `27f6012505a6440aa96878d39685625ba5a7b2781d6a26202af04c6acab5016c` |
| `probe-manifest.json` | `1ed0e65ed690734fd5ec972ed8f0b7e0afb85d41d36f997432bf40c19b65a2b8` |
| `DISPATCH.txt` | `55ac2550e91b27b3c035040b460c76582e07843c1710463c3baf776bfa82c022` |

l4 sparse source plan: 460 files, 8,596,011 unpacked bytes;
fallback archive SHA256 `bb6936241feb74bc2bdafc194cdf2b6c17a2011575c14a4054a72ab18be63398`.

probe sparse source plan: 460 files, 8,596,011 unpacked bytes;
fallback archive SHA256 `978cd9cb524a2832db3ba2b7bce12dd0585d08153db0aa6f67a24cc5553a14e5`.
