# E3: information value of geometry and history

Development-only offline projection experiment. Every output is excluded from
registered campaign verdicts. No serving/release authority. The immutable
[PROTOCOL.md](PROTOCOL.md) was committed before any E3 data reconstruction.

The package reconstructs saved T3 roots from the unsealed bank, accumulates real
post-transition D300 mass for every resolved first action, and fits A/B/C ranking
students. Counterfactual continuations react with the original saved B8 checkpoint
and opponent policies. The original T3 survival labels select roots; they never
stand in for mass returns. Sources, checkpoints and protocol are hash-bound.

`catalog.py` selects exactly 256 worlds and 1,536 roots from final-bank provenance,
with 192/32/32 world splits and 768 decisive/768 counterweight roots. Each world
has both classes. Split quotas fill shortages from other roots in that split,
in fixed identity order; no target outcome participates in selection.

`reconstruct.py` verifies original raster/scalar/mask/action/frame identities,
records preceding 128 decisions at the prepared selector seam, extracts six
owner-relative occupancy planes plus unclipped tail-release order, and verifies
the unchanged B8 continuation against the actual mass prefix. Historical T3
worlds and roots remain reused development data, never fresh strict evidence.

`students.py` defines parameter-matched ranking students; `fit.py` fits the frozen
recipe, selects epochs using validation only and evaluates held-out worlds once
after all arms finish. All geometry uses fixed physical scaling, without clipping.
Artifacts are final-only checkpoints, not exact training resumes.

All outputs use create-only payload + `.sha256` + `.complete`, including failed
terminal receipts. `job.sh` uses a fresh child under the supervisor-owned output
directory. `run.py` merges deterministic process-pool results in task order.
The Mac smoke invokes this same Bash/Python entrypoint at nice10, one math
thread per process, serial reconstruction plus a two-process byte-parity check.

Operator artifacts are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e3/`.
See its `DISPATCH.txt` for the measured forecast, pinned manifest and absolute
commands. `manifests.py` only calls local supervisor validation/source packaging;
it never allocates. Active endpoint prices must be reviewed by the operator at
dispatch; unknown assignments remain a broker refusal. No auto-retries.

Targeted qualification:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q research/bl_info_value
```
