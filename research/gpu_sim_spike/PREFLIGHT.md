# GPU simulator spike — Mac qualification and dispatch handoff

Development evidence only; no GPU cost result, CONTINUE recommendation, serving
change, or registered study verdict. No Colab/provider calls were made. Both GPU
manifests are prepared and locally validated, **not dispatched**.

## Source identity and chronology

- Frozen baseline and verbatim kill rule: `179d60446fc4819011fda555ade0191a2bcf8f92`.
- Complete device implementation: `b1b5d099`.
- Measured checkout runtime: `81ea8141c4e3e6367312fd2cf48ef090ad921bad`.
- Both manifest pins after source-closure correction: `9c50da82ebc5e2642707ccb3f2e07fba9d295288`
  (execution modules unchanged; generator adds backend/config source inputs).
- Documentation handoff commits may follow that runtime pin; execution Python
  hashes are bound in the receipt and checked by the manifest generator.

The frozen L4 conversion is explicitly approximate: 0.036 u/world divided by
60 eligible dense searches = 0.0006 u/decision. Its prospective 10x ceiling is
0.00006 u/decision. Existing S0 evidence has 1/5 CPU processes, not 12; the new
benchmark persists matched CPU12 per-case measurements before device setup.
Repeated inputs fill CPU throughput slots but do not inflate parity sample size.

## Actual launcher smoke

Primary checkout Python 3.12.3 / Torch 2.9.1 CPU build. CUDA hardware unavailable.
Ran `bash research/gpu_sim_spike/colab.sh` at `nice -n 10` with one native-library
thread, one reference worker, `GPU_SIM_SPIKE_DEVICE=cpu`, `HORIZON=2`, `FIXTURES=tiny`,
one decision per case and a 240 s wall limit. The script used uploaded-input-style
paths containing links to the six immutable local checkpoints. `COLAB_CLI_ROOT`
was the owned artifact `mac-entrypoint-01` directory; this local invocation calls
no Colab CLI. The script verified the final JSON checksum and completion marker.

Tiny means a 16x12-cell arena, six snakes, real B8 and vector/v8 opponents, three
mixes and six fixed branch lanes. This is runtime qualification, not the full S0
long-body bank or D300 parity qualification. Existing local work contended for
CPU; no Path B quiet timing window was used or interrupted.

| Measured stage | Total seconds, three decisions |
| --- | ---: |
| Benchmark elapsed | 5.121966 |
| Entrypoint run, including result construction | 5.189716 |
| Candidate pipeline, including setup/warmup/replay | 1.536570 |
| Resident rollout loop | 0.644968 |
| Host fixture creation | 0.585043 |
| Model/table/state setup and transfer | 0.017294 |
| Readback | 0.000333 |
| Selected-label CPU action replay | 0.003838 |
| Exhaustive independent CPU search, study overhead | 3.564175 |
| Final JSON serialization/hash/local write | 0.001080 |
| Process CPU during resident loop, overlapping its wall | 0.638971 |

Resident time was 0.214989 s/decision. Process CPU includes tensor computation in
CPU fallback and must not be interpreted as measured GPU launch overhead or added
to loop wall. Host stages do not sum to pipeline wall because warmup, orchestration
and per-decision persistence are also included in that wall. Bootstrap, provider
transfer, allocation and stop are absent locally and must be taken from future
supervisor receipts. All three first actions agreed; per-branch mass MAE was 0;
there were no unexplained disagreements on this tiny sample.

Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/gpu-sim-spike-20261021/mac-entrypoint-01/output/benchmark.json`

SHA256: `389c750052239978d99bf4e008230284e4e34a7bfcb238eab51d81087b9d7323`.
The same directory retains CPU-before-device evidence, individual decision
receipts, checksums/completion markers, and `host-export.json`.

## Port boundary

Device tensors own body rings and movement, food ordering and FIFO corpse eviction,
boost pellets, collisions/deaths/respawn, advisory masks, vector61 runtime carry,
v8 opponent choices, scripted anchors, actual B8 ego observations and forward,
and dense D-step six-branch continuation/ranking. The fixed tensor schedule is
launched from Python without per-tick device scalar readback. Random-safe hash
lookup tables are constructed once on the host, then indexed on device.

Host work consists of fixture/model/table creation, transfers, Python dispatch and
deadline checks, final readback, independent CPU scoring and artifact export.
The benchmark measures these stages. The optional reference-policy bridge is
explicitly diagnostic and makes a result an upper bound; both GPU manifests use
the resident default. Counter RNG differs from Python MT, rejection sampling is
bounded, and circular geometry is a known divergence risk. Any disagreement
requires evidence for float/RNG classification; otherwise it counts as a bug.
No full S0 numerical parity or GPU speed claim follows from this preflight.

## Targeted checks

36 passed, 64 unrelated supervisor tests deselected, in 4.41 s:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  ../snake-dqn/venv/bin/python -m pytest -q research/gpu_sim_spike \
  tests/test_colab_cli_supervisor.py \
  -k 'gpu_sim_spike or existing_output_directory_is_refused_as_nonfresh'
```

Black (100 columns), isort, flake8, `bash -n`, and `git diff --check` passed.
CUDA coverage includes real B8 FakeTensor CUDA forward and a complete real-policy
meta dense rollout, plus simulator/dense shape checks. These prove tensor syntax
and shapes, not numerical CUDA parity/performance. CPU oracle cases include
long rings, exact masks/features/v8, head-on ordering, boost-food events, corpse
cap/FIFO and same-frame eviction/reinsertion, and respawn metadata. Independent
final read-only review found no remaining material blocker.

## Undispatched manifests

External root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/gpu-sim-spike-20261021/`.

| File | SHA256 | Forecast / caps |
| --- | --- | --- |
| `l4-manifest.json` | `e0da796a8c71e8aa0ed40ba15d65c9f2300aa3143404180c27c2b69a33f03725` | strict L4, >=12 vCPU; 80 min benchmark deadline; 165 min supervisor wall; 5.5 u + 0.5 u reserve |
| `a100-manifest.json` | `cf7ff833cce77e445c71105317b255ddffb7568e26770e7f703dd79cb0e94ec5` | strict A100, >=12 vCPU; 23 min benchmark deadline; 48 min supervisor wall; 5.5 u + 0.5 u reserve |
| `DISPATCH.txt` | `7ca1674845f47059afdeaa73103c4c755185c278cdd2ff625c7cddba159970da` | Operator instructions; no automatic retry |

Combined ceiling is 12 units, below the 15-unit spike envelope. Planning rates are
2.0 u/h L4 and 6.8 u/h A100; substitute actual provider rate only in the final
cost report and stop if allocation would breach its bound. Estimates are timeboxes,
not measured full-bank completion times. Timeout means INCOMPLETE, no CONTINUE.
Before dispatch the operator must refresh every account assignment/rate and keep
Round 0, Claim A D0 and O1 priority. No automatic allocation retry or GPU fallback.

Both drafts passed `research.colab_cli.supervisor validate` locally and retain
`l4-local-validation.json` / `a100-local-validation.json`. The immutable source
plans are created with the existing supervisor source delivery helper, without
provider calls. Both manifests bind all six checkpoint hashes, the measured source
commit and preflight receipt. `RESULT.md` remains the operator result template.

Each offline source plan contains 457 files / 8,563,427 unpacked bytes, below the
32 MiB bound. L4 fallback archive SHA256 is
`b44bb543082280bbbaa975344d5d5ee1979fff430f7fd69cbd946fa0f83ed4de`;
A100 fallback archive SHA256 is
`12153025f1d263a82a2961a750669aab21af32f201d6d776744e647da83e8054`.
The differing gzip headers do not change the verified pinned file inventory.

The first sparse entrypoint check exposed a missing transitive CPU-reference
backend import. Its original artifacts are preserved under
`superseded-source-closure-01/`. The generator now includes `web/backend` and
`configs`; no reference source was edited. Installed the corrected L4 fallback
with `source.install_fallback` (verified inventory, no fetch/provider call), then
ran its actual launcher on all three tiny mixes. It passed checksum/completion,
3/3 first-action agreement and zero branch mass error, in 14.441184 s including
cold imports/JIT. Receipt: `mac-sparse-entrypoint-02/output/benchmark.json`; log:
`sparse-local-entrypoint-02.log`. This verifies the sparse bootstrap source closure
and manifest-pin fallback, in addition to the warm checkout preflight above.
