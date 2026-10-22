# DENSITY-2 completion transport repin 02 — prepared, not dispatched

Mechanical repin requested by the owner on 2026-10-22. No Colab calls or dispatch
were performed. Science, budgets, seeds, ledgers, allocation and decision rules
remain unchanged; METRICS_AMENDMENT_05.md still governs. Prior evidence is immutable.

New pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-02`

Source pin: `96e9b7769dbf0d4ce428dadd0c9320988fc864ca` (published main).
Qualified transport tree: `213f25be1b6a9de5b42a02c387eb86304e3731fe`.
Acceptance: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/warmup-requalification-20261022/requalify-v9t3s_ai/acceptance-1792703228/acceptance-result.json`,
SHA256 `fd7de192c376f2245dfdd290238b7dba995d2bb27b5ac8163e5f11caaa4c583e`.
The handoff commit adds only this record; executable source remains at the pin.

## Attempt accounting and preserved contract

The previous pack's canary 01 failed only the staging-time admission rule following
the first-upload warm-up stall, and its receipt proves stop/release with no cleanup
error. It consumes mechanical canary attempt **1 of 2**. The replacement is canary
**02 of 02**, the last authorized canary. There is no automatic retry or third canary.
Production completion has not run and remains completion attempt **01**. The
prepared canary's internal job ID retains the generator's completion index `01`;
the operator dispatch name explicitly identifies canary `02`.

If a later production completion failure requires a changed ledger and fresh canary,
no additional canary is available under this pack. Preserve that gate; this repin
adds no attempt, budget or waiver.

The production ledger has 408 verified entries, byte-identical SHA256
`70b252ec52c7d61c62c419e77e4e837a8dc1c5d120c283a61439d186565f444c`.
There remain 84 registered episodes and 12 non-scoring identity replays. The synthetic
canary ledger and input-hashes file are also byte-identical. Both templates keep the
resume block, exact namespaces, N, workers, horizons, rules and base/effective owner
2x envelopes. Canary: 10-minute estimate, 30-minute timeout, 1.0-unit cap including
0.2 reserve. Completion: 35-minute estimate, 180-minute timeout, 5.0-unit cap including
1.0 reserve. Series envelope remains 16.50 units, including both canaries. No cap is
raised or redoubled. Both runtime output directories are fresh.

## Local validation

`LOCAL_VALIDATION.json` and `DELIVERY.json` in the pack bind the exact templates,
identical input digests, prior canary receipt and qualification. Recursive comparison
against pack 01 allows only commit/job/output identities, the environment commit,
qualification path/hash and local paths to byte-identical inputs. Frozen science
checks pass against `f1904777aded79366703c94afbf5dc13155bb0ee`.

Both templates pass schema, upload-readiness and `tools/ops/dispatch.py`
`require_resume_policy`. Canary offline admission passes the new source-bound
acceptance. Production admission without a recovered canary proof refuses as intended.
The real dispatch admission path ran in an isolated temporary OPS_DIR through
qualification at broker and manifest pin, immutable snapshot and actual launcher
`validate`. Its `start` function was replaced by an exception before tmux/provider
submission; no watcher or allocation ran. The false-admission completion template
also refuses at the dispatch admission gate. The offline admitted canary is validation
material only; operator admission creates a new file with the current account map.

Commands:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 \
  research/bl_density tests/test_colab_cli_canary.py \
  tests/test_colab_cli_schema_failures.py tests/test_ops_dispatch.py \
  tests/test_colab_cli_warmup.py
# 247 passed in 19.06 seconds.

bash -n /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-02/DISPATCH.sh
git diff --check
# Both pass.
```

The local real runner canary uses the prepared manifest environment, the frozen B8
checkpoint, the exact synthetic skip ledger and the actual runner's fixed 12-unit,
H8, one-worker allocation. Its output/log are under `local-real-skip-canary/` and
`local-real-skip-canary.log` in the new pack. This is local producer proof, separate
from the pending live family canary; it cannot admit production by itself.

## Exact operator dispatch commands

Review the current endpoint-to-positive-rate JSON map before **each** command.
The supplied map must reflect the account at that dispatch; offline `{}` is no
claim about live assignments. The script creates admitted manifests and invokes the
ops wrapper with fresh state/output paths.

```bash
# Set ACCOUNT_MAP to the reviewed current absolute JSON map path.
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-02/DISPATCH.sh canary "$ACCOUNT_MAP"

# Wait for terminal canary PASS, recovered exports and verified backend release.
# Refresh ACCOUNT_MAP, then run completion; this verifies/binds the real proof.
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-02/DISPATCH.sh completion "$ACCOUNT_MAP"
```

The script names are `density2-skip-canary-02-96e9b776` and
`density2-completion-01-96e9b776`. Failed admission/dispatch is not permission to rerun
an attempt. Keep remote production summaries sealed; the registered Mac merge is the
only scoring look. No excluded artifact directory, tmux job, released checkpoint,
serving pin or prior hash-bound file was modified.
