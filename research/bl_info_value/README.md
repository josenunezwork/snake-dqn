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

The upload recovery uses `compact.py` to verify the original v1 archive and
replace saved observation payloads with per-root SHA-256, dtype and shape
bindings. The v2 archive contains only the seed/roster/root catalog. Replay
regenerates local/global rasters, scalars and masks and checks their exact byte
identities before any branch target. Original source indices, frames, actions,
classes, death controls, splits, checkpoint hashes and protocol hash are retained.
The original checkpoint bytes travel in a separate public GitHub input asset;
the supervisor verifies its size/hash before allocation and again on runtime,
and `compact.hydrate` verifies the exact checkpoint member set and every digest.
This GitHub data asset grants no model release or serving authority.

`manifests.py` enforces a conservative 1 MiB local-input limit for this recovery.
The supervisor has a shared 300-second initial upload deadline, not a fixed safe
byte limit and not a fresh allowance per file. Public input downloads use its
separate bounded path. Recovery job/output identifiers must be fresh; the failed
attempt is charged conservatively against the original 16-unit envelope.

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
