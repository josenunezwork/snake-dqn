# DENSITY-2 completion transport repin 03 — prepared, not dispatched

Mechanical repin requested by the owner on 2026-10-22. No Colab calls, allocation,
tmux submission, or dispatch occurred during preparation. Prior evidence remains
immutable. METRICS_AMENDMENT_05.md continues to govern without amendment.

Pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-03`

Executable source pin: `0e7bbd307df6cbec8d0ed1aa34c188680d47a271`.
Qualified transport tree: `e8e207d24472decb1eb586119c53ff1e5707edf0`.
Registry: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`.
Qualified commit: `168a7a6b082789b6e079d513658fb20465cb4f2b`.
Acceptance: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/warmup3-requalification-20261022/requalify-80auhwsd/acceptance-1792707441/acceptance-result.json`,
SHA256 `fc7d6a907a8fb314caf7e796d72585f09c3a0816ca60d51c577991001c440214`.
The handoff commit adds only this record and its inventory; source stays at the pin.

## Attempt accounting

Canary 01 from pack -01 consumed **1 of 2** canary attempts. Pack -02's replacement
canary failed during allocation/kernel-probe setup with `tracked session lacks
kernel probe metadata`. Its receipt records INCOMPLETE/SUPERVISOR, no transfers,
no exports, terminal null, cleanup_error null, and stop_verified true. An assignment
was created and stopped; the research workload did not start. Receipt SHA256 is
bound in the new pack's LOCAL_VALIDATION.json. No production outcomes were opened.

The owner's explicit instruction for this repin says that this pre-workload failure
consumes **no additional mechanical retry**. Therefore **1/2 canaries are consumed;
this remains canary 02/02**, not canary 03. Production completion has consumed
**0/2 attempts**; next is **completion 01/02**, with one further completion attempt
only under the existing registered failure/release/ledger/canary conditions. The
resume block retains `max_outcome_blind_retries: 2` as required by the generic ops
policy; it does not enlarge the study's tighter two-completion-attempt limit.
No third canary, budget increase, retry automation, or rule waiver is introduced.
The canary's internal generated job ID retains the production completion index 01;
the operator dispatch name identifies canary 02.

## Preserved contract and validation

Both templates differ from pack -02 only in source commit/environment commit,
job/output identities, qualification receipt path/hash, and local paths to identical
input bytes. All other fields compare recursively equal. The production ledger
retains 408 verified units and SHA256
`70b252ec52c7d61c62c419e77e4e837a8dc1c5d120c283a61439d186565f444c`.
Canary ledger and input-hashes are byte-identical. There remain 84 missing registered
episodes and 12 non-scoring identity replays. N, seeds, allocation, namespaces,
workers, horizons, science and decision rules are unchanged. Both templates retain
resume blocks. Both runtime output directories are fresh.

Owner 2x envelopes are preserved exactly: canary base 0.5 units / 0.1 reserve /
15 minutes, effective 1.0 units / 0.2 reserve / 30 minutes; completion base 2.5 units /
0.5 reserve / 90 minutes, effective 5.0 units / 1.0 reserve / 180 minutes. Estimates
remain 10 and 35 minutes. Series envelope remains 16.50 units. No redoubling.

LOCAL_VALIDATION.json binds qualification, unchanged input hashes, recursive
comparison, upload readiness, prior failure evidence, resume admission, and local
producer proof. OFFLINE_VALIDATION.py records the one-shot validation procedure;
it refuses if its create-only validation outputs already exist. It ran the actual
`tools/ops/dispatch.py` admission in an isolated temporary OPS_DIR: qualification at
broker and manifest pin, immutable snapshot, actual `launch.sh validate`, and resume
and science admission. The submission function was replaced by an exception before
tmux/provider submission; watcher startup was never reached. An empty account map
was used only for offline validation and makes no claim about live assignments.
Production admission without recovered canary proof refuses; dispatch also refuses
the non-admitted completion template. Local fixed H8 producer proof: 9 episodes
completed, 3 skipped, status COMPLETE; no production data inspected. Live family
canary proof is still pending and required before production dispatch.

Verification:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 \
  research/bl_density tests/test_colab_cli_canary.py \
  tests/test_colab_cli_schema_failures.py tests/test_ops_dispatch.py \
  tests/test_colab_cli_warmup.py tests/test_colab_cli_guard.py \
  tests/test_colab_cli_admission.py tests/test_colab_cli_dispatch_warmup.py
# 292 passed in 32.42 seconds.
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python tools/ops/qualification.py check \
  --repo /Users/josenunez/Projects/ml/snake-dqn-density \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
# e8e207d24472decb1eb586119c53ff1e5707edf0
bash -n /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-03/DISPATCH.sh
git diff --check
# PASS.
```

## Exact operator commands — not run

Supply a reviewed current absolute endpoint-to-positive-rate JSON map before each
command. The script admits fresh manifests using that map and invokes ops dispatch
with fresh state/output paths. Do not reuse an attempt following a failure without
checking its registered recovery conditions.

```bash
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-03/DISPATCH.sh canary "$ACCOUNT_MAP"

# Wait for live canary PASS, recovered exports, and verified backend release.
# Refresh ACCOUNT_MAP to the reviewed current account map before completion.
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-03/DISPATCH.sh completion "$ACCOUNT_MAP"
```

Dispatch names: `density2-skip-canary-02-0e7bbd30` and
`density2-completion-01-0e7bbd30`. Keep remote production summaries sealed; only
the registered Mac merge takes the scoring look. No Colab calls, live jobs, tmux
sessions, released checkpoints, serving pins, prior hash-bound files, or excluded
artifact directory were accessed or modified by this preparation.
