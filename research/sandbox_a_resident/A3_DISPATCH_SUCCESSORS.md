# A-3 mechanical dispatch successors — 2026-10-26

Outcome: both real `tools/ops/dispatch.sh --dry-run` calls PASS from the exact
clean detached checkouts below, with allocations=0 and live_accounting=NOT_CHECKED.
The refused predecessor dispatch identities remain consumed and untouched.
This is offline admission evidence; no A100 runtime was allocated or executed.

| Packet | Runtime commit (unchanged) | Detached host seal commit | Broker and cwd |
| --- | --- | --- | --- |
| packet-a3-eager-02 | 225e0227db7e35eaa34d7c2f06ebc93561388e27 | f3c5fa719ab0a1a5822d6388518ec7dd9ae9051b | /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex |
| packet-a3-04 | d12bf047c6c954c657d15e104e28a2a6ab7d5c32 | 8223e67868b9271b24cb6c218f422ad98e045b8e | /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex |

The selected checkout must contain the committed successor packet and the host
admission timeout fix, while every shipped runtime source byte must match the
older runtime commit. Thus its HEAD is the **host seal**, with the runtime-only
pin still in `dispatch-manifest.json` and `control.json`. A checkout literally at
the older runtime commit would not contain its later successor packet. Dispatch
from sbxA/main would again mix the compiled source into the eager packet.

The eager host seal is published at `origin/sol-a3-eager02-seal-20261026`.
Main contains both successor packets and the timeout fix. Rebase preserved the
compiled seal at `8223e678`; the eager host seal remains separately published
because its eager runtime source must stay frozen.

Only the sandbox adapter's validation window changes: 180 seconds instead of 60.
Ordinary host commands retain 60 seconds; the allocation lock's deadline remains
a stricter bound. Every schema, registry, committed-manifest, shipped-source,
qualified-transport, resume, device and focused-test check still runs. No focused
test cache or test bypass was introduced, and `research/colab_cli` is unchanged.

Both successors retain their predecessor's byte-identical controls,
registry-check and empty verified ledger, identical full seed allocation,
checkpoint, runtime pin, source closure, environment, endpoint, technical envelope
(base14400/effective28800 seconds), exports and resume policy. Registry admission
rechecks all 22,457,142 allocated seeds. Since both predecessors were refused
before allocation, no seed replacement was needed. Fresh job IDs, state/output
paths, dedicated checkout references and derived manifest hashes are the only
packet changes. All 101 shipped source files in each host match its runtime pin.

The two packet `SEAL.json` files record the preparation baseline under the name
`host_seal_parent`; this is not the immediate parent of the final host seal.
The actual eager seal parent is `7fd831fe41972a68a2231d62268eae957758aac5`;
the actual compiled seal parent is `db334038038484434dc23e54cde9c16178c6ae59`.
The literal packet files remain immutable. This sibling document is on current
main at `/Users/josenunez/Projects/ml/snake-dqn-sbxA/research/sandbox_a_resident/A3_DISPATCH_SUCCESSORS.md`;
the historical host seals predate this verification handoff.

## Recreate the sealed checkouts if absent

The named checkouts already exist and are detached and clean. These commands are
for recreating absent checkouts; never reset or overwrite an existing checkout.

```bash
git -C /Users/josenunez/Projects/ml/snake-dqn-sbxA fetch origin
git -C /Users/josenunez/Projects/ml/snake-dqn-sbxA worktree add --detach /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex f3c5fa719ab0a1a5822d6388518ec7dd9ae9051b
git -C /Users/josenunez/Projects/ml/snake-dqn-sbxA worktree add --detach /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex 8223e67868b9271b24cb6c218f422ad98e045b8e
ln -s /Users/josenunez/Projects/ml/snake-dqn/venv /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex/venv
ln -s /Users/josenunez/Projects/ml/snake-dqn/venv /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex/venv
```

## Exact operator dispatch commands

Eager successor:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex/tools/ops/dispatch.sh sandbox-a-resident-a3-eager-20261026-02 /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex/research/sandbox_a_resident/packet-a3-eager-02/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex --cwd /Users/josenunez/Projects/ml/snake-dqn-a3eager02.noindex --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/a3-eager-attempt-02/state
```

Compiled successor:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex/tools/ops/dispatch.sh sandbox-a-resident-a3-20261026-04 /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex/research/sandbox_a_resident/packet-a3-04/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex --cwd /Users/josenunez/Projects/ml/snake-dqn-a3compile04.noindex --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/a3-attempt-04/state
```

Append `--dry-run` to either exact command to repeat offline admission without
reserving the identity or allocating. Both were run with that appended flag:

```text
PASS_DRY_RUN sandbox-a-resident-a3-eager-20261026-02 allocations=0 live_accounting=NOT_CHECKED
PASS_DRY_RUN sandbox-a-resident-a3-20261026-04 allocations=0 live_accounting=NOT_CHECKED
```

Fresh state/output and dispatch reservations remained absent after both calls.
Actual dispatch still performs live account inventory/accounting and allocation
checks. No automatic completion retry is launched by this preparation.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 SNAKE_DQN_DEVICE=cpu nice -n 15 ./venv/bin/python -m pytest -q tools/sandbox/tests/test_admission.py tests/test_ops_dispatch.py
```

Result: **119 passed in 51.76 seconds**, exit 0. Black at line length100, isort
check with profile black, flake8 on the two changed Python files, and
`git diff --check` passed. Independent review also ran the three new timeout tests
in the compiled sealed host: 3 passed, 28 deselected.

The final invariant check reported:

```text
PASS_SUCCESSOR_INVARIANTS packets=2 runtime_files=202 clean_detached=2 fresh_identities=2 predecessor_immutable=PASS
```

Machine-readable commands, exact dry-run stdout/exit status, source/manifest
hashes, seal commits and absent fresh identity paths are recorded in
`VERIFICATION_A3_DISPATCH_SUCCESSORS.json`. No tmux sessions, released checkpoints,
serving pins, original hash-bound packets, runtime source or qualified transport
were modified by this repair.
