# C1 six-session RPC qualification — Claude dispatch

Authority: `big-lever/round-24/JUDGMENT_PLACEMENT_MAC.md`, section 7,
SHA256 `738a367bec2c3399ed8bd9383811d7fa643109a8aaba384a775f1c028a7285ea`.
This package supersedes the old budget/harness-blocked paragraph in
`research/colab_cli/CONTROL_RPC_REQUALIFICATION.md`; those historical bytes are preserved.

The package is prepared offline. A successful dispatch is not qualification.
No live session or scientific outcome is claimed by the Codex-V offline receipt.
The existing final-only acceptance on `a18bf6b4` must pass, with the complete
source-bound acceptance evidence and zero counted calls, before any profile dispatch.
The ops qualification registry must contain that transport tree. The qualified
`research/colab_cli` tree and `tools/ops/requalify.py` are unchanged.

The explicit engineering adapter is outside the qualified transport tree. It admits
only this authority-bound profile, verifies the offline Codex-V source hashes and
the pinned source bytes, then uses the same supervisor `Broker`, CLI, allocation,
lease, transfer and verified-stop implementation. It does not weaken the generic
family-canary admission used by other jobs. `tools/ops/dispatch.sh` routes only
the exact profile entrypoint through this adapter and retains both qualification checks.

Each generated pack has precisely six loop templates and one separate injection
template. Frozen F1 source is archived at
`2af48cd0b98a2063f46871cf4d320a5541b789af`; the logical 64-root bank hash is
`a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d`.
Every session has a new namespace, allocation, manifest, state, output and ledger.
Worker setup is outside the measured loop. W2 CPU and W12 L4 repeatedly admit
the ordered bank for 3,600 seconds, stop admission, and finish in-flight roots.
Only timing, integrity and opaque hashes leave the workers. Rates describe the
workload and cannot decide placement.

| Session | Count | Base units / minutes | Effective units / minutes |
|---|---:|---:|---:|
| L4 W12 loop | 3 | 2.5 / 80 | 5.0 / 160 |
| CPU W2 loop | 3 | 0.15 / 80 | 0.30 / 160 |
| CPU injection | 1 | 0.06 / 15 | 0.12 / 30 |

Reserves are inside the caps: effective L4 1.0 u, CPU 0.06 u, injection 0.024 u.
The series caps are 15.0 u L4 and 0.90 u CPU, plus 0.12 u injection; whole-package
ceiling 16.02 u. The owner's factor is applied once. Each of the fixed session
slots is dispatched at most once; there are no automatic completion retries.
The existing ops exact accounting-allocation retry remains separately recorded
and never turns a failed call into a clean session.

From this worktree, prepare a fresh pack after the branch commit is pushed:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-stallfix
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.rpc_qualification.manifests prepare --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-24/rpc-qualification-01
```

The generated `DISPATCH.txt` gives the three `wave` commands. Supply the actual
`a18bf6b4` final-only acceptance result path. The L4 wave admits three sessions
one allocation at a time under the account lock, leaving their workloads
concurrent. Run CPU wave after all L4 receipts exist, then injection whenever
capacity is available. Every dispatch refreshes the reviewed live assignment map;
the account cap remains three including every other lane. Registered lanes are
never stopped or displaced. A full account refuses dispatch.

All failed, exhausted, fenced or timed-out calls count even after recovery. The
checker retains raw CLI, transfer and release records. No new fault carve-out
is introduced. A platform with any counted fault fails. Injection uses three
actual F1 units; responses are dropped client-side only after successful sends,
then the same request IDs are retried after independent kernel-idle evidence.
Completed IDs are also reissued. Proof includes the actual RPC markers, unit
result, execution evidence and the raw synthetic failure indices. A real fault
in injection fails CPU; L4 also requires the injection proof to pass.

After all seven terminal receipts and verified disappearance, publish the
hash-only ledgers **before** opening the one-look verdict. The checker `publish`
command creates the immutable publication index at the plan's repository-relative
`publication_ledger`; commit and push that index to this branch only. Then
`verdict` fetches this exact branch and verifies its published bytes, plan hash,
ledger hashes and every unit before reading integrity/faults. No main push is
part of this task. Exact commands are included in the generated pack.

The result admits a clean platform only to F1-bis, earliest 2026-10-26. The frozen
25 percent and stall rules and the cross-platform audit remain binding. Two
failed attempts close a platform for October; no second attempt is supplied
here. Any mechanical second-attempt plan requires new namespaces and independent
Codex-V review at the same envelope.

Historical L4 36.738-second fault: `UNRESOLVED_NON_TIMEOUT_CONTROL_FAILURE`;
the saved records cannot establish child, provider, or tunnel attribution.
It remains counted and is never relabelled as a timeout.
