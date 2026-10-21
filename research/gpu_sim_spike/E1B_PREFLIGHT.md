# E1b Round 02 preflight and prepared-only handoff

Outcome: implementation and local qualification complete; L4 manifests locally
validated, **NOT DISPATCHED**. No Colab/provider calls, serving changes, or
registered verdict. CUDA compilation, numerical parity, memory use and production
throughput are unmeasured on this Mac.

## Identities and scope

- Prospective Round 02 brief: `aa37b7dc` (`ROUND02_BRIEF.md`), published before E1b data.
- Native kernel/benchmark implementation: `f9c0f552`.
- Inference-cache reset correction: `c531751a`.
- Final-unit deadline correction: `83b83488`.
- Measured runtime and both manifest pins: `7bd4ce0cc560f98599220516510fb983b59e77ea`.
- Frozen rule SHA256: `a49aa11384a2636319d6eef80d6a38943bf759c7d14ea3804d39da0e5c17765a`.
- Preflight: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1b/mac-entrypoint-03/output/benchmark.json`.
- Preflight SHA256: `f4a0a1f8dc1b8ee246aa9b9c0034894ad0db3f8a83386e492bd8f4fc8ceb794b`.

CUDA ego connectivity now uses actual 16x16 tile-local union-find, cross-tile CAS
root unions, pointer jumping, integer component counts once per world and row
start gathers. Parents decrease monotonically; bounded root/union watchdogs set
device status and an asynchronous assertion fails visibly. CUDA v8 uses one block
per world/slot/action, sparse same-distance FIFO batches, atomic deduplication,
and a global queue sized for queued cells (`min(V, 1 + 4*max_cap)`). Release checks
precede seen marking; rejected cells remain available for later eligible paths.
All phases use the caller's current CUDA stream. Expanded inputs become contiguous.
CPU and FakeTensor retain the prior implementations. Reach-time's exact function
bytes (24 synchronous tail-aware rounds), the CPU reference adapter, parity/
decisive-root audit, `BASELINE.json` and `KILL_RULE.md` are unchanged.

The native CUDA selfcheck compiles before expensive CPU12 audit work and tests
production geometry (83x145), 160/4096 caps, cross-tile disconnected components,
duplicate rows, and release rejection followed by acceptance via a longer path.
Compiler/load/selfcheck failure is a failed preparation, never a silent fallback.
No native compile or execution occurred locally; the two native tests skip only
because CUDA is unavailable.

## Actual Mac entrypoint

AC power, lid open, no reported thermal warning; host contended. Existing sessions
`colab-limit-test`, `git-sync`, `phase0-selection`, `t34-round0-c002` were observed
and left untouched. Commands ran at `nice -n 10`, one native-library/Torch thread,
one CPU reference process, using the primary checkout's existing Python 3.12 venv.
This is operational preflight, not Path B timing or a quiet-machine claim.

The exact argv/environment are retained in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1b/mac-entrypoint-03-command.json`.
The command runs the actual `colab.sh` locally with uploaded-input-style links to
all six hash-verified immutable checkpoints. Settings: mode all; tiny dense
fixtures on all three mixes, D2, one searched decision per mix; REALISTIC requests
256 worlds and explicitly clamps to two resident CPU worlds, all six body lengths
800/900/1000/1100/1300/1500, one warmup tick plus one measured tick per repeat,
three repeats per mix, 600-second combined deadline. No runtime allocation occurs.

Outer wall: **76.842020 s**. Entrypoint wall:
**75.643130 s**. Dense loops: 0.297729 s;
36 scheduled full six-snake branch-world steps, 120.915329/s.
The serial rate is diagnostic and cannot qualify the production-shape floor.
Dense first-action agreement **3/3**, unexplained bugs **0**, branch mass MAE **0**.
Decisive roots **0 / NO_COVERAGE**, agreement null; other roots 3/3. This tiny bank
cannot qualify the frozen D300 cost/agreement rule.

| CPU long-body mix | Resident worlds | Median full world-steps/s | p95 full tick, seconds |
| --- | ---: | ---: | ---: |
| long-scripted | 2 | 0.537091 | 3.737572 |
| long-frozen | 2 | 0.501219 | 4.136759 |
| long-mixed | 2 | 0.524256 | 3.859667 |

All nine warmed long-body repeats completed. Floor status:
`PRECHECK_ONLY_NOT_L4_DATA`. Every timed tick copies a resident long-body
world/runtime template, performs lifecycle preparation, evaluates the real B8 hero
and five real opponents, steps the full simulator, and writes resident length,
alive, action and Q-max collection fields. This is explicitly a conservative
fixed-root stress workload; resetting each tick prevents cheap dead-row padding.
It is not a continuous training trajectory or learner benchmark. Vector61 feature
free-space wavefronts remain unchanged and their cost remains included.

The first mechanical attempt (`mac-entrypoint-01`) exposed inference-cache tensor
mutation outside inference mode. Its failed log/process record and completed
dense receipts are preserved. The correction and a reproducing regression were
committed. Attempt02 passed; attempt03 requalified the final deadline-corrected
source. No scientific failure was retried and no thresholds changed.

## Targeted verification and source delivery

Final targeted check: **61 passed, 2 CUDA-only skipped, 77 unrelated deselected,
5.60 s**. The package contains CPU adversarial FIFO/connectivity oracles, parity,
decisive-root classification, export discovery/checksums, inference-cache reset
and final-unit deadline regressions.

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/gpu_sim_spike tests/test_colab_cli_supervisor.py \
  tests/test_colab_cli_runtime.py \
  -k 'gpu_sim_spike or existing_output_directory_is_refused_as_nonfresh or partial_local_export_without_completion_marker_is_rejected or read_state_returns_only_completed_hash_verified_exports or export_cache_hashes_once_and_rejects_later_mutation'
```

Black (100 columns, <=2 workers), isort, flake8 on the package, `bash -n colab.sh`,
and `git diff --check` passed. No broad suite was run. Both pure manifest validation
and actual `research.colab_cli.supervisor ... validate` passed locally. The recorded
transport executable inventory hash-matches the qualified `dab00823` inventory;
`transport-source-check.json` is offline identity evidence, not renewed acceptance.

Each sparse plan has **468 files / 8,668,423 unpacked bytes**, including native
sources, requirements, launcher and their transitive source closure. The L4 plan
was installed through the real hash-verifying `source.install_fallback`; its actual
launcher passed on all three tiny dense mixes in **16.701649 s**,
checksum/completion verified. `mac-sparse-entrypoint-01` is preserved. This proves
CPU import/source closure and sidecars, not native CUDA readiness.

- L4 fallback SHA256: `44ec546da3059f9f8cb31ce6eeef3a0aac4a4c26e69a78a60a652a127d3cbff7`.
- Probe fallback SHA256: `a912d7820ed5d2ddc634d11fad5d2e550363a177c46e93e27288ad3b64058959`.

## Prepared manifests and forecast

Artifacts: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1b`. Every exported payload has `.sha256` and `.complete` sidecars;
65 closed payloads were verified before writing `artifacts-index.json`, which has
its own sidecars. A completion marker authenticates a closed payload, not a success
verdict: inspect each status/returncode. No pre-existing evidence was overwritten.

| Artifact | SHA256 |
| --- | --- |
| `l4-manifest.json` | `29b33db54b89ed279264ac0661e301ba188c76890a5ed93f04f17f4c61b1f575` |
| `probe-manifest.json` | `102da36f3e36ab655c0286387c678a47a0c0342e70bdfce02256c63a6bc35967` |
| `DISPATCH.txt` | `fe0cbbba82c2a947bbec95e4445837292b01b313f2e30e59163499deccdd3c48` |

Both manifests run mode all: unchanged dense audit plus REALISTIC
256/1024/4096 actual concurrent CUDA worlds, six long bodies 800–1500, all three
mixes and five real opponents, three warmed repeats, p95 tick and peak device
memory. The main spike measures 12 ticks/repeat; probe 3, warmup 3. All nine shapes
must be present; rates at both 1024 and 4096 on all three mixes test the unchanged
<20,000/s L4 STOP rule. Serial dense rates cannot satisfy it. The original <=0.1x
cost, >=99% agreement, zero unexplained disagreements and decisive-root audit remain.

| Job | Planning timebox | Whole-session wall cap | Execution cap + reserve | Envelope |
| --- | ---: | ---: | ---: | ---: |
| L4 spike | 190 min | 400 min | 13.5 + 1.5 u | 15 u |
| Optional L4 probe | 60 min | 135 min | 4.5 + 0.5 u | 5 u |

At the prepriced 2 u/h ceiling, planning charge is 6.333 + 2.000 = **8.333 u**;
including reserved 2 u gives **10.333 u planned commitment**. Combined hard envelope
is **20 u including probe/reserves**. Whole-wall maximum debit is 17.833 u before
2 u reserve (19.833 u total), within the envelopes. Benchmark deadlines are 180/55
minutes; first applicable cap wins. These are bounded allocation timeboxes with
setup/export headroom, not measured L4 completion forecasts. No L4 speedup, cost
verdict, CONTINUE recommendation or serving qualification is claimed.

`DISPATCH.txt` contains exact local validation commands and the later dispatcher's
source/account/rate/priority checks. Empty `other_assignments` is a draft
placeholder, not observed account inventory. Neither job was dispatched. No
status/usage/auth/allocation/exec/upload/download/stop provider call was made.
No automatic retry, hardware substitution or RunPod job is authorized here.

No authored changes to protected T34 packages, `research/colab_cli`, serving pins,
released checkpoints or live jobs. The forbidden study artifact directory was
never opened. Unrelated concurrent main-line commits were merged without editing
their files; all authored commits and integration merges use the requested coauthor
trailer. The runtime pin precedes this documentation-only handoff commit.
