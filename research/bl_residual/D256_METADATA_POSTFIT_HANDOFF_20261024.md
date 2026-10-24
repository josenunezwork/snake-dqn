# Metadata-pack D256 post-fit continuation

Use `RUN_D256_POSTFIT_METADATA_20261024.sh` in this worktree. It continues only
`big-lever/exp/f4/d256-metadata-pack-02`, fit pin
`6313fe8fce990915ef1c3702ff1687169513b624`. The operator reported dispatch
`bl-f4-d256-metadata-fit-02` completed with EXIT=0. This handoff does not inspect
its ops log, receipt contents, fit payload, fit outcome or reserve outcomes.
The commands below remain operator-owned and were not run on the real pack.

Run one command at a time; continue only after success:

```bash
NEXT=/Users/josenunez/Projects/ml/snake-dqn-f4fix/research/bl_residual/RUN_D256_POSTFIT_METADATA_20261024.sh
bash "$NEXT" publish-fit
bash "$NEXT" stage-d1024
```

`publish-fit` invokes the frozen qualified source's existing
`f4_d256_package --publish-fit` without changing its ledger format. It requires
the supervisor's PASS, verified stop and no completion/cleanup error; EXIT=0
alone does not suffice. The helper binds the metadata-pack allocation and
complete immutable fit export using path/size/hash, without loading the fit.
The create-only ledger triplet retains the registered path
`research/bl_residual/ledger/F4_D256_FIT_PHASE_20261023.json`.
A separate create-only binding triplet hashes the metadata-pack main manifest
and terminal supervisor/producer receipts, the fit pin and the ledger. Both triplets and
their campaign-log hashes are committed and pushed to main before any metadata
consumption. Publication verifies all six local files equal freshly fetched
`origin/main` and rehashes the bound manifest/receipts without interpreting them.
`verify-fit-publication` can repeat that read-only check after publication.

`stage-d1024` repeats publication verification first, then calls the frozen
`--stage-d1024` helper. It consumes fit metadata only to enforce the registered
unconditional-fault stop and validate the predecessor identity. It opens no
reserve table and performs no fit. It creates a fresh staging directory:

```bash
F4_INPUTS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d1024-metadata-staging-02/inputs
F4_CONTROL="$F4_INPUTS/control.json"
```

The later D1024 collection operator must pass these variables explicitly in its
dispatch command after `--` using `/usr/bin/env`, retaining its T-EQ/W12 gates,
frozen operational pin, collection caps and resume accounting. The older
`RUN_D256_NEXT_DEVICE_FIX_20261024.sh` hardcodes different staging inputs: do not
use its `stage-d1024`, `publish-fit` or unmodified D1024 collection phases for
this continuation. This change does not prepare or launch a new collection.
The timing-arm refusal remains binding; the script never removes the arm.
Occupied immutable publication/staging paths are a HOLD, never overwritten.
If a publication push fails, preserve the locally created triplets/commit and
finish publishing those exact bytes; do not regenerate them by rerunning
`publish-fit`. Run `verify-fit-publication` before staging after recovery.

## Next registered scientific look: deferred joint R-A

There is **no D256-only analysis/look command**. Registration R5/R6 requires
D1024 collection and its immutable, fault-free fit before joint Look R-A.
`heldout.evaluate` refuses a D256-only look. `funding.main` owns the single
joint canonical claim and calls the registered held-out evaluator for both
checkpoints. `analyze.py` is the later gameplay analyzer, not a post-D256 look.

After D1024 preparation/fit, publish and verify every intervening phase ledger
before analysis. Freeze the complete natural ordinary-row archive census before
R-A. Only then use the registered command below on the Mac, outside a timing
window. The capitalized variables refer to future sealed D1024/census evidence;
they are deliberately not guessed from the D256 pack. This is a deferred CLI
template, not a command ready to run after D256 staging. Prepare its fresh
exact Mac accounting allocation/job ID through ops before invocation:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f4-metadata-qualified-02.noindex
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02
/Users/josenunez/Projects/ml/snake-dqn-f4-metadata-qualified-02.noindex/tools/ops/dispatch.sh \
  --mac --repo /Users/josenunez/Projects/ml/snake-dqn-f4fix \
  "${R_A_JOB_ID:?fresh ops-registered Mac accounting job ID}" \
  6313fe8fce990915ef1c3702ff1687169513b624 -- \
  /usr/bin/env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.funding \
  --d256-checkpoint "$PACK/d256-output-l4-01/exports/fit/fit-0000.pth" \
  --d256-table "${D256_RESERVE_TABLE:?sealed reserve plus natural ordinary archive table}" \
  --d256-allocation "$PACK/inputs/life-allocation.json" \
  --d256-control "$PACK/inputs/control.json" \
  --d1024-checkpoint "${D1024_CHECKPOINT:?published immutable D1024 final}" \
  --d1024-table "${D1024_RESERVE_TABLE:?sealed reserve plus natural ordinary archive table}" \
  --d1024-allocation "${D1024_ALLOCATION:?sealed D1024 allocation}" \
  --d1024-control "${D1024_CONTROL:?sealed D1024 fit control}" \
  --archive-receipt "${ARCHIVE_RECEIPT:?complete natural-archive v2 census}" \
  --approval "$PACK/inputs/judgment.json" --b8 "$PACK/inputs/b8.pth" \
  --look-lock "${R_A_CANONICAL_LOCK:?allocation-bound F4_R_A canonical lock path}" \
  --out "${R_A_OUTPUT:?fresh create-only joint funding output}"
```

The lock is `F4_R_A_<canonical_sha(lock)>.lock.json` under the D256 allocation's
parent directory, where `lock` is exactly the `bl-residual-r-a-lock/v1` object
constructed from both `heldout.preflight_look` identities in `funding.main`.
Prepare this identity after both fits are published; do not create the claim
file in advance. No published F2 receipt is supplied here: the registered
unpublished-F2/strikes-volume branch applies. Add `--f2-receipt` only for a
separately verified published F2 disposition. The look remains once-only; this
handoff changes no registration, recipe, decision rule, dose or serving state.
