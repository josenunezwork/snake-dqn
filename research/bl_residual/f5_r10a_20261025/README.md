# F5 R10a metadata annex

Disposition: **HOLD_BEFORE_EXPOSURE**. This package does not adopt F5. The exact
v2 registration and independent judgment require metadata that the available
life-level ledgers do not contain. No diagnostic exposure is authorized by it.

The annex binds the 960 selected D1024 lives, the unchanged 192-life D256 prefix,
372 eligible parents, 9 original ordinary-only sibling lives, the permanent
29-parent development exclusion and all descendants, and 167 original fold-0
parents (the 29 are a subset). It includes all 684 **provisional** parent records
and a seed/parent collision audit against 3,024 known D256/D1024 allocations.
Round 0 is clear against that limited inventory. Accepted round and accepted
allocation hash remain null because complete historical audit coverage is absent.

Exact per-row and ordinary-sibling row counts, COV missing-action IDs/count/hash,
original runtime/library/RNG/config/ordered-roster/tie-classification identities,
and complete historical alias/ancestry coverage remain unresolved. The admitted
R30 inventory explicitly defers row totals. Its 960-life count is not an exact
row/action inventory. U is null, **not zero**, and the 46,080-action ceiling has
not been assessed. Do not substitute an inferred count, a prefix, a new collector,
the current Mac environment, or a namespace-only collision audit.

The tool reads only the named metadata in the SHA-pinned input manifest. It
verifies the byte hashes, SHA sidecars, and completion markers; it never follows
payload references or computes payload hashes. The source closure is checked
using `git show` at the exact semantic commit in the draft. No model, simulator,
factory, post-mortem, or scientific constructor is imported or executed.
Only the authorized post-mortem development-parent exclusion ledger and its
sidecars are read. D4096 and `t34-round0-study-01/` are never inspected. A focused
read-only source search found no lawful existing metadata that closes the gaps.

From `/Users/josenunez/Projects/ml/snake-dqn-f5annex`, reproduce into a **new**
directory (the command refuses every existing output filename):

```bash
nice -n 15 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ./venv/bin/python -m research.bl_residual.f5_r10a_annex \
  --inputs research/bl_residual/f5_r10a_20261025/R10A_INPUTS_20261025.json \
  --publish /private/tmp/f5-r10a-reproduction-NEW
```

The tool uses one Python process and at most one sequential Git subprocess.
JSON is canonical UTF-8 with a newline; the human summary binds the JSON hash.
Each file is atomically published with a no-replace hard link; the completion
record binds JSON and summary hashes and is published last. A failed or
interrupted publication without that record is incomplete. Preserve it and
choose a fresh destination. The source/input hashes make output reproducible;
rebase/commit timestamps are deliberately absent. The final files are copied
byte-for-byte beside the v2 draft using the same create-only publisher.

Run the focused synthetic tests in one worker process (32 tests). The verifier
sets and checks exact nice 15 and Torch intra/inter-op threads 1/1. It uses
`execve` to set the macOS OpenMP library path before import, preserving the PID
and preventing pytest's default parent/child relaunch:

```bash
./venv/bin/python research/bl_residual/f5_r10a_20261025/verify.py
```

Only metadata and synthetic tests are in scope. No Colab, dispatch, real-label
completion, fitting, diagnostic selector, F4 post-mortem execution, serving or
release work is performed. Resolving HOLD requires a new sibling annex with
lawfully published missing metadata and independent adoption review; preserve
this package and every existing research document unchanged.
