# I3 FIX-FIRST: D4096 admission

Implements all three findings in the full independent Codex-V review:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-review-i3/snake-dqn.last-message.md`.
This is code and synthetic metadata verification, not production admission,
Claude-V acceptance, a funding look, a fit or dispatch authorization.

## Findings and changes

1. `job.py` selects zero-cost admission from validated measured charges. When
   they equal zero, the declared projection must equal exactly zero and all
   Mac/B2 checks apply. The tolerant arithmetic comparison is reachable for the
   positive branch only when measured and declared charges are both strictly
   positive. Collection charges also reject booleans and non-finite values.
   Regressions cover `1e-10` and `1e-12` declarations with measured zero,
   present/missing B2 and Mac/non-Mac placement, plus lawful tiny-positive
   measured charges and refusal of declared zero against positive charges.
2. Only a sealed, correctly bound `bl-residual-funding-look/v1` receipt with
   `verdict: PASS` admits funding. A direct F2 disposition refuses even if
   published, non-strike and labelled PASS. Lawful F2 funding still enters via
   `funding.py`'s canonical PASS; I2 retains ownership of that producer.
   Both pack preflight and workers check terminal discard and collection state
   before generic input hashing. Missing, pending and discarded dispositions
   refuse. Terminal discard wins over funding and release evidence, including
   missing funding. The full release binding is repeated by staging validation.
3. Semantic-test mutations now reseal all dependent collection, B2, projection
   and release hashes. Provenance tests assert those relationships and the
   intended semantic refusal. Raw byte/sidecar tampering is tested separately.

## Consumer contract and exact I1 integration steps

At the latest fetch before this fix, `origin/main` was `b796bf36`; it contained
no I1 B6/disposition writer. The unpublished I1 source in the separate
`snake-dqn-d4096-i1` worktree names B6 schema **`d4096-not-run-by-rule/v1`**.
The synthetic fixture `tests/fixtures/bl_residual_i3_terminal_discard.json`
matches its judgment/readiness/terminal/terminal-label field shape. This is
schema coordination evidence, not published producer compatibility proof.

`job.py` refuses any canonical `d4096-not-run-by-rule.json` in the input root,
including partial sidecars, malformed bytes or a dangling symlink. It also
refuses the presence of a `D4096_DISCARD` stage binding at any basename. These
are fail-closed presence checks: embedded Mac paths are never followed on a
portable worker. Never remove a discard candidate to obtain admission.

The mandatory release schema is **`bl-labels-d4096-collection-disposition/v1`**,
defined by `tests/fixtures/bl_residual_i3_collection_release.json`. It admits
only `disposition: RELEASED`; every other state refuses. Required bindings:

| Field | Required value |
|---|---|
| `judgment_sha256` | `ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb` |
| `series_id` | Current fit control's frozen series |
| `registration_sha256` | Exact F4 registration SHA |
| `reserve_allocation_sha256` | Current fit control's reserve SHA |
| `recipe_sha256` | Registered fit recipe SHA |
| `collection_ledger_sha256` | Current control's `factory_union_ledger_sha256`, valid SHA256 |
| `funding_receipt_sha256` | Exact sealed canonical PASS funding receipt's SHA256 |

Before any production D4096 fit pack:

1. I1 must publish a create-only disposition writer matching that fixture,
   with SHA256 and completion sidecars. Only release after canonical PASS and
   authoritative B6 refusal checks, binding the exact published final union.
2. The future D4096 fit-pack builder must recheck I1's authoritative collection
   root for B6 immediately before copying/dispatch. Copy the released receipt
   triplet into the fit input root; bind its basename as
   `stage_receipts.D4096_COLLECTION_DISPOSITION` and its digest in `input_hashes`.
   Bind the same final union digest as `factory_union_ledger_sha256` in control.
   Propagate any discard as the canonical triplet or `D4096_DISCARD` and refuse.
   A stale copied RELEASED snapshot cannot detect a later external discard.
3. Run the existing `job --preflight-main` entrypoint on that input directory;
   `run()` repeats the guards. The F3 synthetic pack builder already calls this
   preflight when passed main inputs, but still refuses main-manifest creation.
   No D4096 main-pack builder is implemented or authorized here.
4. Reconcile B2 before production: the unpublished I1 writer currently uses
   `bl-labels-d4096-mac-h-projection/v1` with terminal-based fields, while the
   existing I3 consumer requires `bl-labels-f0-d4096-mac-projection/v1` with
   ledger-based fields. I1 must publish a compatible receipt or an explicitly
   reviewed adapter; run the actual writer-to-consumer metadata test afterward.
   No permissive schema fallback was added.

No `research/bl_labels/*`, `funding.py` or `f4_ra_prepare.py` was edited by I3.

## Verification

From `/Users/josenunez/Projects/ml/snake-dqn-receipts-i3`, use this exact launcher
so macOS does not strip DYLD before Python starts. All tests use nice 15,
threads=1, serial pytest, and at most two Python processes (pytest plus the
receipt CLI child):

```bash
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
nice -n 15 /bin/zsh -c 'export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib; exec ./venv/bin/python -m pytest -q -p no:xdist -p no:cacheprovider tests/test_bl_residual_collection_receipts.py tests/test_bl_residual_pipeline.py -k "not real_synthetic_entrypoint and not fit_fault_seals and not raw_adapter and not heldout"'
```

Final selection: **130 passed, 4 deselected**. Includes real collection and
projection CLI entrypoints on synthetic metadata and pack/worker early-refusal
tests that deny every non-staging hash read and confirm no output directory.
Black, isort, flake8 and `git diff --check` passed for the changed Python files.
Independent read-only review found no remaining blocking admission issue.
No broad payload/training suite was run while the factory was live.

An initial exploratory run placed DYLD before `/usr/bin/nice`; macOS stripped
it and conftest restarted pytest, briefly reaching three Python processes in
the CLI test. The final command above sets DYLD inside the niced shell and
execs Python directly; reruns have no restart and satisfy the process limit.

## Operator preflight (not executed against production)

After all integration steps and independent acceptance, set the path to the
freshly prepared, sealed input directory; this command performs no dispatch:

```bash
: "${I3_D4096_INPUTS:?Set the prepared D4096 fit-pack input directory}"
nice -n 15 /bin/zsh -c 'export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib; exec /Users/josenunez/Projects/ml/snake-dqn-receipts-i3/venv/bin/python -m research.bl_residual.job --preflight-main --input-dir "$I3_D4096_INPUTS"'
```

Export `I3_D4096_INPUTS` and the thread variables above first. Production
receipts, actual I1 producer compatibility and Claude-V ACCEPT remain open.
Live D1024, tmux, `exp/mac-lease.json*`, `t34-round0-study-01/`, checkpoints,
serving pins and existing hash-bound evidence were untouched.
