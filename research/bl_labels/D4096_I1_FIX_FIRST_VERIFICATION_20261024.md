# D4096 I1 FIX-FIRST verification — 2026-10-24

Outcome: required findings 1–3 fixed; metadata-only L4 inventory hardening included.
No collection or provider work was started. Real-pack first-start check remains HOLD.

Tested implementation commit: `14bf6030e636fca62f73cd10e49b10479559deb6`, rebased without conflicts
onto `b77f9408` (the newer I2 fix). The following evidence/handoff commit changes documentation and
new immutable dry-run evidence only. The source of all nine changed Python files remains identical
to this tested implementation commit.

## Finding map

1. `d4096_design.py` supplies the exact prospective **nested mixed-platform (Mac + Colab L4 CPU)
   collection, tie-tolerant L4 admission** design and owner-decision path/SHA. New readiness, B2,
   attempt-start/terminal, projection, rebinding, discard and release receipts carry it, with actual
   placement separate. `collection_receipts.py` changed only for this disclosure. The decision bytes
   were hashed with `shasum -a 256` and matched its supplied `.md.sha256`:
   `003399e7ee6835d622ecf48fc04d2e862926c61991f8426243b1bf62db300b20`.
   The original handoff/note are preserved; sibling addenda provide the updated instructions.
2. Both terminal-label and terminal-receipt discard routes use `_stop_label`; the receipt route
   infers its stage from the schema and validates any explicit stage. Reading revalidates the same
   semantics. Tests refuse recoverable receipt HOLD, accept explicitly unrecoverable receipt HOLD,
   reject forged stage and reject a legacy recoverable-HOLD seal on read.
3. `_present` uses `exists() or is_symlink()` for payload and both sidecars, including create-only
   checks. Three parameterized regressions cover dangling payload/digest/completion links on the
   canonical, explicit, alternate and symlink-target-alias routes. Independent review found the
   alias edge during this round; it was fixed and the reviewer accepted the revised code.

The release writer additionally binds a separately sealed complete L4 attempt inventory plus its
accounting ledger, rehashes each listed supervisor receipt and refuses missing charge rows for
failed/setup/identity attempts. A mixed union refuses an empty inventory. A Mac-only union may still
bind failed L4 attempts, and absent accounting requires explicit `--no-l4-attempts` instead of silently
emitting an empty zero-cost snapshot. Completeness of the inventory itself remains an operator
attestation; omissions from both inventory and ledger cannot be detected by this local comparison.

## Tests

All tests ran serially, nice 15, numerical threads one, plugin autoload disabled, no xdist, at most
pytest plus one child process. DYLD was exported inside the niced shell to avoid conftest restart.

Final focused run: **159 passed in 6.24 s**. The exact command was the following integrated command
with only `test_d4096_sealed.py`, `test_d4096_driver.py`, `test_d4096_i3_bridge.py` and
`tests/test_bl_residual_collection_receipts.py` selected.

Final integrated run at the rebased implementation: **528 passed in 55.02 s**:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SNAKE_DQN_DEVICE=cpu \
  nice -n 15 /bin/zsh -c 'export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib; exec ./venv/bin/python -m pytest -q -o addopts= -p no:xdist research/bl_labels/test_d4096_admission.py research/bl_labels/test_d4096_resume.py research/bl_labels/test_d4096_sealed.py research/bl_labels/test_d4096_driver.py research/bl_labels/test_d4096_i3_bridge.py research/bl_labels/test_w12_runtime.py research/bl_labels/test_mac_driver.py research/bl_labels/test_w12_driver_guards.py research/bl_labels/test_d1024_resume.py research/bl_labels/test_d1024_final.py research/bl_labels/test_d1024_old_root.py research/bl_labels/test_mac_gates.py research/bl_labels/test_mac_integration.py research/bl_labels/test_w12_admission.py research/bl_labels/test_slowdown_state.py research/bl_labels/test_slowdown_resume.py research/bl_labels/test_l4_lane.py tests/test_bl_residual_pipeline.py tests/test_bl_residual_f4_fit.py tests/test_bl_residual_collection_receipts.py'
```

Earlier integrated run: 526 passed before the last alias/empty-inventory hardening and its two added
bridge cases. Counts overlap and are not summed. The first focused run found a stale placement
assertion in a fixture that had manually replaced only accounting; it was corrected to exercise the
actual L4-bound writer. The final runs above include that correction.

## Lint

All nine changed Python files passed Black, isort and flake8. One initial test-local import ordering
issue was fixed before the successful final checks. Reproduction from the worktree root:

```bash
files=(
  research/bl_labels/d4096_design.py
  research/bl_labels/d4096_sealed.py
  research/bl_labels/d4096_pack.py
  research/bl_labels/mac_driver.py
  research/bl_labels/test_d4096_sealed.py
  research/bl_labels/test_d4096_driver.py
  research/bl_labels/test_d4096_i3_bridge.py
  research/bl_residual/collection_receipts.py
  tests/test_bl_residual_collection_receipts.py
)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 15 ./venv/bin/python -m black --check --line-length 100 "${files[@]}"
nice -n 15 ./venv/bin/python -m isort --check-only --profile black "${files[@]}"
nice -n 15 ./venv/bin/python -m flake8 "${files[@]}"
nice -n 15 make PY=./venv/bin/python lint
git diff --check
```

Repository `make lint` also passed: Black checked 149 src/backend files; isort and flake8 passed.

## Real-pack dry-run evidence

The exact command and all before/after bindings are in the new create-only receipt
`ledger/D4096_I1_FIX_FIRST_DRY_RUN_20261024.json` and its SHA/completion sidecars.
Receipt SHA-256: `94a7fdb103ae79d2d840f3536c6cf912ff27c65050513a748b8291d6a002967c`.
It was run using `./venv/bin/python`, nice 15, numerical threads one, at the implementation commit.
The operator reproduction is the complete `--check` command in the sibling handoff.

Result: **HOLD, exit 2, factory_started=false**, reason `F0 export lacks complete sidecars`.
B1/B2 triplets were absent. Admission/state/output paths were absent before and after. All five
frozen pack payloads verified against `ledger/D4096_OLD_EQUIV02_BINDING_20261024.json` before and
after the check, including the unchanged A1 control SHA
`753cd378abcf736df64d66b4285dd741072f8633c4a06d0402c1b35317d74d96`.
No production readiness, projection, funding or release receipt was manufactured.

The sole external write is the fresh pack sibling addendum and SHA/completion sidecars:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d4096-old-equiv-02-pack-01/D4096_MIXED_PLATFORM_ADDENDUM_20261024.md`,
SHA `0c83ad06feee6411eb6690b1d47fc3338327fd8edf9a96bc0dd0000ce29471fe`.
It is byte-identical to the committed sibling note.

Independent read-only review accepted the revised fixes and sibling instructions. Actual B1 exit,
lease release, fit-job terminal facts, B2 and runtime host conditions remain future operator gates;
this verification does not establish them. Updated operator commands are in
`D4096_I1_HANDOFF_FIX_FIRST_20261024.md`.

No live D1024 factory/state/output, tmux session, `exp/mac-lease.json*`, `t34-round0-study-01/`,
released checkpoint, serving pin or existing immutable evidence was modified. No edits were made
to `research/bl_residual/funding.py`, `f4_ra_prepare.py` or `f2*.py`. No provider query or dispatch ran.
