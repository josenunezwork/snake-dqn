# GPU simulation development spike

This isolated package implements the owner-approved [brief](../../docs/research/specs/2026-10-21_gpu_sim_spike.md).
It never changes serving, a registered collector, or the Colab supervisor.
`BASELINE.json` and the original rule were committed first (`179d6044`). E1's
prospective additions were committed before E1 data at `896310e3`; see
[KILL_RULE.md](KILL_RULE.md). The original cost/agreement bars are unchanged.
E1 also stops the engine thesis this week below 20k full world-steps/s on any
L4 long-body mix and reports decisive CPU-root agreement separately.

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
a branch. Python launches the branch schedule; flood convergence reads a scalar every
eight rounds on real tensors. Those CUDA synchronizations are included in rollout
wall time and measured launch CPU time, which overlaps
rollout wall. The default device policy ports ring-body movement, ordered ambient/corpse/boost
food events, death/respawn, advisory masks, vector61 features and runtime carry,
v8 decisions, scripted anchors, actual B8 raster/scalar features and forward,
and six-lane dense branching. Counter RNG changes are explicit parity risks.
An optional diagnostic CPU policy bridge is named in the receipt and makes the
device speed an upper bound; it is not selected in either E1 L4 manifest.

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
receipt are in `E1_PREFLIGHT.md`. CUDA FakeTensor checks cover tensor
syntax/shapes on a Mac; they do not prove CUDA numerical correctness or throughput.

`make_manifests.py` prepares one L4 spike (15 u envelope) and one optional L4 probe
(5 u envelope) manifest in the external
artifact directory. Both use the qualified CLI bootstrap and local hash-bound
uploads. The supervisor's `validate` subcommand is local only. No dispatch,
allocation, status, usage or other Colab calls are part of this task. The operator
must preserve higher-priority lanes and obtain a free account slot before dispatch.
`RESULT.md` is the operator result template; no GPU cost verdict is claimed here.
E1 artifacts/manifests live under `snake-dqn-artifacts/big-lever/exp/bl-e1/`.
The earlier `PREFLIGHT.md` remains historical evidence for the pre-E1 implementation.

Ego region sizes use uncapped min-label component propagation with pointer
jumping, computed once per world. V8 uses newly discovered cardinal wavefronts,
capped at the CPU's 160/4096 limits and stopped when all lanes are saturated or
exhausted. Release-rejected cells are not retried after the frontier empties.
The unchanged production CPU implementations remain the oracle.

Throughput is `6 branch worlds × depth / synchronized rollout wall seconds`;
each world includes the hero and all five real opponents. Dead/illegal branch
lanes remain scheduled and are disclosed. Pipeline rates also include setup,
warmup, readback and selected-label CPU replay. Neither rate substitutes isolated
simulator throughput or multiplies by six snakes. CUDA numerical parity and L4
speed remain unmeasured until Claude dispatches.
