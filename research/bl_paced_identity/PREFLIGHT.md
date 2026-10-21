# X7 build verification — 2026-10-21

**BUILD_VERIFIED_NOT_QUALIFIED. No paced study or Colab call.**

Worktree `/Users/josenunez/Projects/ml/snake-dqn-bl-x7`, branch
`bl-paced-identity-20261021`. The clean baseline was `57483b2f`; upstream main
was fast-forwarded to `442397dd1185522be8c20be1cf58a0ad9e814bf4` before publication.
Only `research/bl_paced_identity/` and its five targeted test files were added.
The publication SHA is in the operator handoff; the immutable evidence below
binds exact source bytes rather than an ambiguous dirty-tree commit label.

## Retained evidence

Authoritative build artifact directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-x7/preflight-04/`.

| Export | SHA256 |
| --- | --- |
| `receipt.json` | `47340cbbce8d6a8217506425da0e6f33db7fb44a635b740914bbe902507187f0` |
| `manifest.json` | `4bff4aab9a2567353cda64a2a905d4b17d819f792fe67b4fe7151ada492bad2c` |

All JSON exports have verified `.complete` and `.sha256` sidecars, including
headers, chunks, terminal seals, semantic/replay receipts and the fixture index.
The audit reran every synthetic trace and the real-game replay, checked the
joined oracle identities, and rehashed the explicit imported local source
closure. It passed again after integrating upstream main.

The Mac preflight took **1.078488625 seconds**, one spawned worker, one math
thread per process (parent + one worker), `nice -n 10` (actual nice recorded in
the receipt), 26 synthetic interleaving fixtures and eight unpaced GridBatchSim
ticks with reacting opponents. The independent replay reproduced request and
dispatch sequences, actions, states and integer score prefixes. The semantic
fixture compared GridBatchSim with a separately constructed BatchSim reference
for six first-action branches and independent greedy continuations.

Preflight PASS and tiny semantic/replay PASS are explicitly PRECHECK_ONLY.
The joined receipt remains INCOMPLETE, L3 identity prerequisites BLOCKED, paced
oracle NOT_RUN, and manifest `dispatchable=false`. No served checkpoint was
loaded. No production 18+96 root bank was run. Phase 0 was running; these elapsed
times cannot qualify serving latency, roots/s, p95 or zero overruns.

Earlier attempts are retained: `preflight-01`/`02` are superseded functional
checks; `preflight-03` completed its checks but failed source-closure publication
on a generated Torch filename (`_classes.py`). The source inventory now includes
only existing local files. No attempt was reused or overwritten.

## Targeted verification

Final X7-only pytest: **43 passed in 1.38 s**. Black, isort, flake8 and
`git diff --check` passed. Independent read-only review found and closed
ranking bypass, delayed carry aliasing, receipt cross-binding, omitted request/
dispatch evidence, deadline recorder-cost, worker cleanup and exact inventory
issues. No broad test suite ran.

Exact test command:

```bash
nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_bl_paced_identity_controller.py tests/test_bl_paced_identity_journal.py \
  tests/test_bl_paced_identity_receipt.py tests/test_bl_paced_identity_replay.py \
  tests/test_bl_paced_identity_semantic.py
```

The preflight used the same environment and interpreter with:

```bash
-m research.bl_paced_identity.preflight --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-x7/preflight-04
```

Verification-only command (does not overwrite artifacts):

```bash
nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_paced_identity.audit \
  --preflight /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-x7/preflight-04
```

Lint commands use the same five test files and `research/bl_paced_identity`:
`python -m black --check --line-length 100`,
`python -m isort --check-only --profile black`, and `python -m flake8`.

## Manifest and forecast

The manifest describes W=1/K=1 and W=4/K=8. All intermediate integer W/K
configurations within W≤4/K≤8 are accepted by the model; real multi-worker
operation remains behind a strict boolean owner-authorization guard. A string
such as `PENDING` cannot authorize it. The manifest records that approval as
pending; no multi-worker process probe ran.

Colab units: **0**. No paced hours used. Future reservations from advisor 03:
2 unpaced Mac semantic hours, 8 unpaced replay hours, L3 cap 9 paced hours,
L4 cap 32 paced hours. These are adopted engineering reservations, not measured
production forecasts or dispatch authorization. Production semantic/replay
rates, strict N and bare-comparator hours remain null/unmeasured.

The illustrative full-horizon candidate subtotal for 640 H5000 worlds at24 fps
is 37.037 hours before overhead. Death padding can shorten it; 640 is not a sized
strict bank. A bare comparator may be unpaced only under the applicable future
registration. No A100/H100, RunPod or Colab acceleration claim is made.

No tmux session was changed or stopped. Released checkpoints, serving pins,
hash-bound campaign documents, prohibited research paths and the sealed Round 0
study directory were untouched. Serving still requires the owner's explicit OK.
