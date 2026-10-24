# R32 consumer rehearsal coverage

`consumer.py` owns only the R32-1 artifact-to-consumer trace. It uses the existing
F4 factory-shaped synthetic table, `validate_rows`, `tensor_cache`, residual fit,
fusion, `export_torch` sidecars and `load_ego2s_checkpoint`. It writes a fresh,
create-only output directory. Every exported file has a payload, SHA-256 sidecar,
and marker published last by `research.bl_ppo.exports`.

The runnable trace covers both a zero residual and a deterministic nonzero residual.
For each delivered fused checkpoint a new Python process reloads the bytes and selects
actions through the normal advisory-mask rule, including the empty-mask normal-action
fallback and a deterministic all-Q tie. `heldout.validate_candidate` is also invoked
against a synthetic checkpoint/control envelope and must refuse it. This proves a real
production admission boundary; it is not a JSON-only assertion.

## Deliberately blocked boundaries

The complete T34 strict request must be validated by
`src.evaluation.strict_promotion.freeze_strict_request`. That validator requires a
live-engine request and bound E0, pilot, calibration and serving artifacts, while this
rehearsal produces an ego2s residual checkpoint. No adapter or compatible T34 schema
exists in this package, so strict validation, independent audit/admission, test-local
registry exercise, rollback and product-mode checks are **BLOCKED**, not mocked passes.
The missing adapter/schema requires separately reviewed work.

Without `--mac-smoke`, the three excluded worlds and continuous-respawn Watch smoke are
**NOT_RUN_PENDING_ADMISSION**. On macOS, `--mac-smoke` (also accepted as
`--mac-world-smoke`) runs one isolated H3 exact-root world per scripted/frozen/mixed mix,
then steps an explicit-checkpoint `GameSession` Watch loop three times. It does not resolve
a registry name or constitute registry admission, access serving pins, write approvals, write
a leaderboard, or make a release decision. The flag itself refuses non-macOS hosts.

## Claude command after admission

Run only in an admitted Mac gap, at the existing nice-15 and one-thread limits, with a
fresh external output directory and the exact immutable B8 checkpoint selected by the
operator:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.r32_rehearsal.consumer \
  --b8 /ABSOLUTE/IMMUTABLE/b8_s3_t7_20261016.pth \
  --output /ABSOLUTE/FRESH/r32-consumer-rehearsal --mac-smoke
```

The optional `--mac-smoke` executes the isolated H3/Watch plumbing described above.
Use the pinned `tools/ops/dispatch.sh --mac` command in `HANDOFF.md` for operator
execution. The consumer enforces a nice floor of 15 without raising an inherited
lower priority, one Torch thread, the shared lease, and a 2700-second hard wall.
This trace has no standalone Colab job and no dispatch authority. Mac R32-1 work
is capped at 0.75 core-hour; portable fixtures share the separate cloud envelope:
forecast 2.0 h / 0.16 u, base 2.4 h / 0.192 u, owner-2x 4.8 h / 0.384 u.
No L4, A100, or H100 allocation is justified unless a separately admitted transport gap
is established.
