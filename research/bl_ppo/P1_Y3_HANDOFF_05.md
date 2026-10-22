# P1 Y3 source-latebody-05 operator handoff

**REVIEWED / VALIDATED LOCALLY / NOT DISPATCHED**, 2026-10-22.
One new family canary and its source template pin published main commit
`ca5382226ef9a3b1ee4aa3e42ab4161851007324`. The final evidence-only commit
containing this handoff does not change those executable bytes. Qualified
transport identity remains `f00fa88bf53e5ac67d893ef2ba98555c03cb2d14`.

The create-only index `P1_Y3_PREPARED_05.json` binds actual artifact hashes,
current acceptance, independent source and artifact reviews, debit reconciliation,
local validation and tests. Both independent reviews PASS with no unresolved
findings; no new CUDA path was added. This is software/admission evidence only.

| Lane | Estimate min | Timeout min | Cap u | Reserve u |
| --- | ---: | ---: | ---: | ---: |
| New canary | 15 | 30 | 2.00 | .40 |
| Y3 source | 75 | 300 | 9.62 | 1.92 |
| Y3 replacement reservation | 30 | 120 | 3.86 | .78 |

Total envelope 25.50 u; historical conservative debit 4.67 u is retained.
`tools/ops/spend.py` independently reconciles four receipts to own usage
3.399723284 u, without refunding the conservative reservation. Initial maximum
path is 20.15 u including predecessors; unallocated balance is 5.35 u.
Rate ceiling is 1.54 u/h, zero tolerance, no other assignments. Replacement is
reserved but not yet admitted. Retries remain outcome-blind, verified-failure-only
and residual-budget-limited per the frozen debit; no automatic retry is allowed.

Both manifests declare `cuda:0`, carry the dispatch resume block and upload the
same verified empty ledger. Operational snapshots are non-scoring, safe-boundary,
900-second full-state exports with latest verified recovery. Replacement must
pass both the boundary replay and operational-snapshot replay. Operational-only
failure produces `Y3_PASS_BOUNDARY_ONLY` and disallows operational recovery.
The collector fix restores verified simulator buffer aliases after serialization;
it was required for exact interrupted-refit replay.

Artifacts:

- Canary: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/family-canary/canary-manifest.json`.
- Existing source template: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/family-canary/source-template.json`.
- Future admitted source: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/family-canary/source-admitted.json`.
- Independent actual-artifact review: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/independent-artifact-review.json`.
- Local proof: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/local-validation.json`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/final-preflight.json`.
- Wrapper-compatible instructions: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022/DISPATCH.txt`.

`ROOT` below is the exact immutable attempt directory. The source-admitted file
intentionally does not exist: admission creates it only after the real canary
terminal receipt, exports and runtime release pass, and it reconciles the new
debit before checking the source/replacement reservation again.

Operator sequence (not executed during preparation):

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-wp3
ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-20261022
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-canary-20261022 "$ROOT/family-canary/canary-manifest.json" --broker "$REPO" --cwd "$REPO" --dispatch-txt "$ROOT/DISPATCH.txt"

# Continue only after the real passing canary terminal receipt, exports and release.
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_family_canary admit --root "$ROOT/family-canary" --receipt "$ROOT/family-canary/canary-output/receipt.json" --acceptance /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/telemetry-requalification-20261022/requalify-g4fekqn4/acceptance-1792700171/acceptance-result.json --review "$ROOT/independent-artifact-review.json"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-source-20261022 "$ROOT/family-canary/source-admitted.json" --broker "$REPO" --cwd "$REPO" --dispatch-txt "$ROOT/DISPATCH.txt"
```

Require fresh empty other-assignment inventory and calibrated rate before each
actual dispatch. No synthetic receipt may replace the real canary evidence.
An allocation exit code is not success; inspect telemetry, final terminal receipt,
verified exports and confirmed release before proceeding.

Verification: whole `research/bl_ppo` package **221 passed**; dispatch/spend and
qualified transport regression selections **224 passed**; final admission checks
**64 passed**, final source05 name checks **4 passed**. The full package includes
interrupted refit and actor recovery through the actual source/replacement
entrypoints, flat exported source closure, exact replay and a negative
operational-only replay. Black/isort/flake8, shell syntax and whitespace checks
pass. Actual artifacts pass local source/canary validation, both upload paths,
wrapper argument parsing and family preflight. The source template correctly
refuses without its SHA-bound family canary receipt. Detailed commands and bound
logs are in the index and registration directory.

No Colab/provider call, dispatch, training allocation or live L4 qualification was
performed. No tmux job, released checkpoint, serving pin or pre-existing hash-bound
registration/receipt was modified. The excluded study directory and
`../snake-dqn-l2mac` were not accessed. No serving/release change is authorized by
this package.
