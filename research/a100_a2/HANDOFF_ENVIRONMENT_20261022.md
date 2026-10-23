# A2 environment repair — fresh sibling pack, operator dispatch only

Supersedes the dispatch paths in `HANDOFF.md` for the mechanically failed canary.
The original handoff, registration, pack-02 and receipts remain immutable.
No scientific code, allocation, labels, fit recipe, thresholds or kill rules changed.

Root cause: the remote requirements omitted Pydantic 2. The lazy import chain
`job._run -> run -> src.model.ego2s_network -> obs_spec -> src.core.__init__ ->
config_loader` imports Pydantic. Mac smoke borrowed it from the project venv.
A2 now declares Pydantic 2 and the bootstrap Torch pin. The conservative recursive
import probe also declares psutil/cloudpickle and ships the helper source closure,
including optional runtime branches; it never calls those helpers or runs a study.

`make_pack pack` now refuses to publish a pack until a fresh managed Python 3.12
venv, installed only from the pinned job requirements, imports the pinned sparse
source closure successfully. Python runs with `-I`, no user/system site packages,
no inherited PYTHONPATH, and no project install. Imports inside functions and
transitive helper modules are checked. `tools.ops.import_preflight` is reusable by
A0/A1/A3; see `tools/ops/README.md`. Real CUDA and A100-40 family proof stays live.

The prior receipt records INCOMPLETE, CHILD_NONZERO_EXIT (missing pydantic),
zero exports and `stop_verified: true`. Its elapsed 792.4886501249857 seconds
includes cleanup. Charge 2.201357361458294 units at the conservative 10u/h ceiling.
Replacement canary remains <=18min/3u; main <=256min/42.79u with 8u reserve.
Prior + replacements <=287.20814416874976min/47.99135736145829u, inside the
original 288min/48u envelope. These are reduced remaining operational limits,
not a new budget. No fit units exist to resume; original empty ledgers are reused.

## Build and evidence

The new immutable pack is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-03`.
It reuses `inputs-20261022-01` by hash. Its `pack.json` contains the full new
published SHA and the clean-import preflight digest. Its sibling
`CODEX_V_ENVIRONMENT_REVIEW.json` is the independent review of the source and
concrete pack, with hash sidecar and completion marker. No old receipt is replaced.

Build from the clean, published repair commit:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-a2
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.a100_a2.make_pack pack \
  --inputs /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/inputs-20261022-01 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-03 \
  --commit "$(git rev-parse HEAD)" \
  --failed-canary-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-02/canary-output/receipt.json
```

The initial pack declares no other assignments. If assignments are active at
actual dispatch, prepare another fresh sibling with reviewed positive endpoint
rates via `--other-assignments`; never edit pack-03. This reruns the clean import
gate. The supervisor refuses unknown/unpriced assignments. The original remaining
series caps continue to apply to every attempt.

## Exact operator commands

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-03
python3 "$REPO/tools/ops/qualification.py" check --repo "$REPO" \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$REPO/tools/ops/dispatch.sh" a100-a2-canary-20261022-02 "$PACK/canary.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-canary-20261022-02 \
  --dispatch-txt "$PACK/DISPATCH.txt"
# Wait for genuine PASS, scanner-verified exports and confirmed runtime release.
cd "$REPO"
"$PY" -m research.a100_a2.make_pack admit --template "$PACK/main-template.json" \
  --canary-receipt "$PACK/canary-output/receipt.json" --out "$PACK/main.json"
"$REPO/tools/ops/dispatch.sh" a100-a2-main-20261022-02 "$PACK/main.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-main-20261022-02 \
  --dispatch-txt "$PACK/DISPATCH.txt"
```

Prepared only. No Colab dispatch, aggregate scoring look, tmux operation, checkpoint
or serving-pin change was performed. The qualified transport tree remains
`553af3f692515b2bf66173356fe4d099acc96942`.

## Offline verification

The final light selection passed 41 tests, 72 deselected, in 3.25s, with
`-p no:xdist` and one native thread. This includes all 20 A2 tests, five real
clean-venv guard tests, and 16 focused source/canary admission cases. Black,
isort, flake8 and Git whitespace checks passed for changed files.

The full declared-dependency probe on Python 3.12.15 passed 541 static import
statements and 179 imported local modules, with system/user site packages
excluded. The final published-pin output is in pack-03's
`clean-import-preflight.json`. Installed distributions are enumerated there;
CUDA execution remains unverified until the new real family canary.

Actual `bash research/a100_a2/colab.sh`, CPU smoke with the original hash-bound
input inventory and synthetic table, completed 8/8 units in 0.7284032499883324s.
Receipt: `../snake-dqn-artifacts/big-lever/exp/a2/mac-smoke-environment-20261022-01/output/receipt.json`.
No historical fitting, replay, scoring, or aggregate look occurred.
