# GPU simulation development spike

This isolated package implements the owner-approved [brief](../../../docs/research/specs/2026-10-21_gpu_sim_spike.md).
It never changes serving, a registered collector, or the Colab supervisor.
`BASELINE.json` and the verbatim `KILL_RULE.md` were committed first (`179d6044`).

The S0 fixture adapter imports the unchanged reference world constructor. It uses
all three opponent mixes, short bodies and long serpentines with hero length 900.
The benchmark forces dense searches on the six fixture roots and at five-decision
offsets; short roots are deliberately included even below T3's production 600-cell
trigger. Every search expands all advisory-legal actions for D300, follows bare
B8 greedy continuation, ranks alive transitions, then prefers current B8, highest
Q and lowest index. It separately records post-step mass over the full horizon.

`reference.py` rescoring owns independent CPU worlds, branches, RNG and runtime
carry. `parity.py` reports first-action agreement, per-branch mass errors, and every
first-action disagreement. It does not relabel unknown bugs as RNG differences.
`rollout.py` keeps six fixed branch lanes and never reads a device scalar to choose
a branch. Python launches a fixed schedule; its CPU time is reported and overlaps
rollout wall. The default device policy ports ring-body movement, ordered ambient/corpse/boost
food events, death/respawn, advisory masks, vector61 features and runtime carry,
v8 decisions, scripted anchors, actual B8 raster/scalar features and forward,
and six-lane dense branching. Counter RNG changes are explicit parity risks.
An optional diagnostic CPU policy bridge is named in the receipt and makes the
device speed an upper bound; it is not selected in either GPU manifest.

The host constructs fixture/model inputs and random-safe lookup tables, launches
fixed tensor schedules, checks the deadline, reads final results, replays every
selected label on the CPU reference, and writes receipts. Each stage is timed.
Independent exhaustive CPU search is separately measured parity-study overhead.
The GPU runs measure the 12-process CPU reference per S0 case before device setup,
using repeated frozen inputs for throughput only, not additional parity samples.
Timeouts persist an incomplete result with no retry. GPU timeboxes are allocation
forecasts, not measured throughput or a promise that all D300 cases finish.

Run only targeted package checks, with `nice -n 10`, one native-library thread and
the primary checkout's existing venv. Use `SNAKE_CKPT_DIR` for the uploaded or local
reference checkpoint directory. The real Mac entrypoint command and immutable
receipt are in `PREFLIGHT.md` once verified. CUDA FakeTensor checks cover tensor
syntax/shapes on a Mac; they do not prove CUDA numerical correctness or throughput.

`make_manifests.py` prepares exactly one L4 and one A100 manifest in the external
artifact directory. Both use the qualified CLI bootstrap and local hash-bound
uploads. The supervisor's `validate` subcommand is local only. No dispatch,
allocation, status, usage or other Colab calls are part of this task. The operator
must preserve higher-priority lanes and obtain a free account slot before dispatch.
`RESULT.md` is the operator result template; no GPU cost verdict is claimed here.
