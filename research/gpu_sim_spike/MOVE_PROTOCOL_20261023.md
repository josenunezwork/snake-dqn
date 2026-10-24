# MOVE A100 sprint packet — frozen optimization candidate, CPU gate first

Authority: owner MOVE work order, round-28 JUDGMENT.md and round-35 astra_01,
astra_02 and astra_03 (all read). This packet neither restarts T0 nor changes the
72-hour integrated speed deadline, CPU regeneration obligations, serving, or
strict/product qualification. Operator dispatch remains required by round-28.

Only MOVE files are authored: `device_sim.py`, new `move_*` modules, the frozen
MOVE oracle and MOVE tests/handoff. FLOOD/V8/shared transport code is untouched.
The default remains `move_backend="reference"`; `indexed` is opt-in.

## Candidate and comparator

The independent optimization oracle is the exact simulator from
`cd26df509b124dc6ea2d1c84844ff3728436a760`, vendored at
`fixtures/move/device_sim_pre_move.py`, SHA-256
`816ebd5fbd4016bb01eda5fe31ee7739f8cb7bf021b2885684f769ea63dc58ae`.
No candidate helpers are imported by this frozen oracle.

The candidate builds occupancy once only for worlds with ready respawn slots,
parallelizes valid ring inspection, and resolves placements and collision events
in the original within-world order. It preserves all 32 counter-RNG advances,
including no-ready worlds, int64 wrapping, earlier ineligible duplicate candidates,
self offsets >=3, unconditional frozen swap predicates, stable death ranks and
corpse materialization order. No float reassociation or CPU-correctness repair is
introduced. Native CUDA compilation/execution must be proved by a current-pin
canary; the NumPy portability twin does not establish native parity.

## Measurement and decisions

Requested main matrix: E64/E256/E1024, scripted/frozen/mixed, six actual body
lengths 800/900/1000/1100/1300/1500, unchanged 10,400-segment capacity. Three
interleaved warmed reference/indexed repeats restore identical recorded state and
actions. Synchronized profiler-free full simulator-step wall is authoritative for
the MOVE stage; respawn and collision diagnostics are separate timing passes.
This stage packet excludes policy inference and cannot prove complete-engine
2,000/s or closed-loop policy identity. Full integrated qualification still needs
historical E6 coverage and continuous trajectories.

**Any unexplained mismatch is INVALID and stops this candidate.** A valid complete
packet with <2x MOVE stage speedup at E1024 on any roster is STOP_ROUTE; missing
cells are INCOMPLETE. No scientific failure is retried. A native pass is not a
release, training admission, or factory qualification.

Independent GridBatchSim checks compare recorded matched-state mechanics outputs
without numerical tolerances or dtype normalization. Device-counter-RNG identity
and GridBatchSim MT identity are separate contracts. Existing mismatches must
remain explicit; they cannot be waived by a successful optimization comparison.
No labels are produced or admitted by this packet.

## Pre-existing discrepancy discovered during preparation

A direct CPU diagnostic before native data used length and previous length 800
at a prepared no-food/no-respawn seam. GridBatchSim emits float64 reward
`-0.7999999999999972` (bytes `809999999999e9bf`); both the original and indexed
device implementations emit float32 `-0.8000030517578125` (bytes `00cd4cbf`).
This mismatch predates the MOVE change. The counter RNG also deliberately differs
from CPU MT. These findings are blockers under the owner's CPU-byte contract,
not declared repairs or exceptions. Preflight must preserve the independent
CPU results and refuse dispatch if they disagree. Stop further performance
tuning/spend when this check fails. A separately declared correctness repair
would require a new candidate identity and fresh preflight/canary within the
original clock and caps.

## Transport, caps and resume

Use the exact qualified transport tree from
`313758255ee5b9f1422a0202c14a80cc32554857`:
`research/colab_cli` tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
A new MOVE source commit is necessary to contain the candidate; the original
transport commit cannot also identify new code. Do not weaken qualification or
modify transport to manufacture admission. Freeze a sibling runtime checkout
based on this pin and publish its MOVE source identity before preflight.

A100-40 only at verified <=10 u/h, >=12 vCPU; one admitted serial A100 lane.
Forecast including setup/export: 90 minutes, 11.55-15 u (planning allowance,
not measured native time). Base: 108 minutes/18 u including 3 u reserve.
Owner's 2x effective envelope: 216 minutes/36 u including 6 u reserve.
Canary: 24 minutes/4 u; main: 192 minutes/32 u, sharing that series ceiling.
Existing round-28 Q4/global balances, liabilities, account balance and original
calendar deadlines dominate. No allocation is authorized by a template.

Namespace + frozen allocation identifies each immutable cell. Payload and SHA
sidecars are published before completion markers. The Mac ledger verifies path,
size and hashes, rejects foreign/duplicate units, and skips exactly verified
units. One original and at most two manual mechanical completion attempts share
the frozen series envelope, fresh manifest/output/receipt each, only after
verified mechanical failure, verified stop and missing units. Scientific failure
is not mechanical recovery. No automatic retry is introduced.
