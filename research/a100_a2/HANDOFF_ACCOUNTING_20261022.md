# A2 preallocation accounting sibling — operator handoff

Use **pack-20261022-06**. This supersedes only the dispatch paths/accounting in
`HANDOFF_TRANSFER_20261022.md`; all scientific and family-canary requirements
remain binding. Prepared only; Claude/operator dispatches.

Runtime pin remains **cb67a4b90ec2577ed398d76b59c18427d6ae764a**. No repository
runtime, training, transfer, supervisor, dependency or packaging implementation
was changed. All five input paths and SHA256 values, source paths, scientific
allocation, empty ledgers, resume contract, forecast and decision rules are
identical to pack-05. Its clean Python 3.12 import preflight is copied unchanged.
The documentation/review commit is newer than the runtime pin deliberately.

Pack-05's failed invocation was `a100-a2-canary-20261022-03`; its manifest job ID
was `a100-a2-canary-cb67a4b9`. The immutable receipt is INCOMPLETE / ACCOUNTING,
SHA256 `76cb42bc1fad54131f53e9a046f45e1e246be864097adc019a9c00245ab237f1`.
Its precise error is **accounted assignment endpoints are no longer active:
gpu-a100-s-usc1c0-780pn6afwnq5**. This was an admission refusal before `colab new`,
not an A2 runtime that was allocated and then released. Receipt calls are only
`version`, `sessions`, `usage`; endpoint/accounting/terminal are null; exports
and both transfer inventories are empty. The pinned supervisor SHA matches the
receipt; its `admit_account()` call precedes the durable allocation intent and
`new`. The failed process has `EXIT=1`; its lane assignment/CLI session records,
account pending/active records, and accounting-baseline file are absent.

`stop_verified: false` stays false. We assert no allocation was requested, not a
successful stop. Claude's no-leak observation was independently refreshed with
read-only provider calls. At preparation the only assignment is A3's L4 endpoint
`gpu-l4-s-ass1a1-1j9axtbcc2yvd`, provider rate **1.54u/hour**, one assignment.
Both new manifests account for exactly that endpoint/rate. Changed inventory or
rates require another reviewed sibling; no old pack may be edited.

The failed admission's entire **1.3290355829522014 seconds** is conservatively
charged at 10u/hour: **0.003691765508200559u**. This is a budget charge, not a
claim of measured provider spend. Together with all three prior allocation
receipts, prior time is **1,674.2544019569177 seconds** and prior ceiling debit is
**4.650706672102549u**. All four receipts and hashes are retained in the new
ancestry. Canary remains **18min/3u**, reserve 0.6u. Main becomes **242min/40.34u**,
reserve 8u, estimate 120min. Worst-case sum is **287.9042400326153min** and
**47.990706672102554u**, inside the original **288min/48u**. No cap reset or new
application of the owner's 2x rule occurs.

**Attempt accounting:** this remains a failed canary invocation in the series
ledger. A2 registers no separate numeric canary-attempt limit. Its limit of two
completion attempts applies to main complete-fit recovery; this admission
produced no runtime or fit units, so the pending main remains **completion
attempt 1 of at most 2**. A1's one kernel-stall retry rule in `PLAN_A100.md` is
inside the A1 section, not A2. This error also differs from the wrapper's sole
exact automatic accounting retry, which additionally requires verified release.
This explicitly requested, reviewed sibling is manual preparation; it does not
expand automatic retries or waive release requirements after actual allocation.

Preparation is recorded in create-only `SIBLING_ACCOUNTING.json` and
`ACCOUNT_OBSERVATION.json` in pack-06. The one-off offline preparation script is
`../prepare-accounting-sibling-20261022-01.py` relative to the pack. Its SHA is
bound by the accounting record. It accepts only the exact owner-named receipt,
checks its manifest, supervisor, ancestor and input hashes, and refuses nine
mutations suggesting an allocation or mismatched evidence. It never allocates
or stops Colab. The old pack and receipt bytes remain unchanged.

Verification: 24 A2 tests passed in 1.97s; 15 focused canary tests passed with
60 deselected in 0.57s, each using `-p no:xdist` and single native threads. Real
`launch.sh ... validate` accepted the new canary. Unadmitted main, failed pack-05
proof and old pack-03 proof were all refused. Exact results and series arithmetic
are in `OFFLINE_VERIFICATION.json`. The transport qualification remains
`553af3f692515b2bf66173356fe4d099acc96942`. Independent Codex-V evidence is
`CODEX_V_ACCOUNTING_REVIEW_20261022.json` beside this handoff and in pack-06.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-06
python3 "$REPO/tools/ops/qualification.py" check --repo "$REPO" \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$REPO/tools/ops/dispatch.sh" a100-a2-canary-20261022-04 "$PACK/canary.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-canary-20261022-04 \
  --dispatch-txt "$PACK/DISPATCH.txt"
# Wait for genuine PASS, <=150s measured transfers, <=300s staging,
# verified immutable exports, A100-40/CUDA proof and stop_verified=true.
cd "$REPO"
"$PY" -m research.a100_a2.make_pack admit --template "$PACK/main-template.json" \
  --canary-receipt "$PACK/canary-output/receipt.json" --out "$PACK/main.json"
"$REPO/tools/ops/dispatch.sh" a100-a2-main-20261022-04 "$PACK/main.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-main-20261022-04 \
  --dispatch-txt "$PACK/DISPATCH.txt"
```

Reproduce the light tests without allocating or fitting historical data:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-a2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu "$PY" -m pytest -q -p no:xdist \
  tests/test_a100_a2_data.py tests/test_a100_a2_fit.py tests/test_a100_a2_protocol.py \
  tests/test_a100_a2_aggregate.py tests/test_a100_a2_job.py tests/test_a100_a2_transfer.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu "$PY" -m pytest -q -p no:xdist \
  tests/test_colab_cli_canary.py \
  -k 'receipt_cannot_admit_an_uncovered_dispatch or probe_retains_real_input_inventory or staging_accepts_slow_source_delivery'
git diff --check
```

No remote canary, main dispatch, historical replay, scoring look, session stop,
tmux mutation, checkpoint change, serving change or release was performed.
