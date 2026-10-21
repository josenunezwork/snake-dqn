# BL PPO: E4 collector ruler followed by the E2 pilot

Development only, excluded from every registered verdict. Binding local rules are
[INTENT.md](INTENT.md), committed before gameplay in `3bedc5de`, its pre-data
[E8 amendment](E8_AMENDMENT.md), and the [E9 frozen rule](E9_RULE.md).
The design comes from the owner's `big-lever/round-01/JUDGMENT.md`, section (f).

`model.py` copies B8's ego2s-bv8 encoder and preserves a frozen calibrated B8 policy
as the base logits. A zero-initialized residual actor learns around that base.
Arm R uses GRU256; arm F uses an exactly parameter-matched feedforward core.
`learner.py` implements finite-horizon undiscounted GAE/PPO, rollout-cut
bootstrapping and a PopArt critic with a critic-only remaining-horizon input.
Actor unfreeze requires the transition-counted refit and world-disjoint late-body
qualification; tiny operational preflight cannot satisfy that scientific gate.
`collector.py` keeps GridBatchSim and the immutable opponent policies in CPU
processes while the parent batches hero inference on the learner device.
`run.py` is the real entrypoint for the ruler and training.

`exports.py` publishes immutable payloads with SHA256 sidecars and completion
markers recognized by the unchanged supervisor. Periodic full-state checkpoints
preserve optimizer, collector and RNG state. Incomplete rollouts are never promoted
to scientific endpoints. Checkpoint exports and terminal training-status receipts
are separate evidence; a weights file alone does not prove completion.

`bank.py` allocates the prospective 128/48/48 screen (legacy 96/32/32 bytes preserved) and excluded fixtures, audits the
published history without opening sealed live outputs, and binds the audit.
`screen.py` runs/analyzes paired fixed endpoints and applies only E2's frozen rule.
A development GO permits replication planning; release still requires the Mac strict
gate, serving qualification, and the owner's authorization.

`manifests.py` prepares local supervisor manifests; `colab.sh` is their pinned
runtime entrypoint. It contains no allocator. E4 must run first. E2 templates are
locally validated preparation artifacts and deliberately fail closed without a
complete E4 receipt, passing immutable E9 admission and budget forecast.
E9 currently BLOCKS scientific E2: at the largest authorized screen, simulated
joint funding at +40 is .307, below .60. S1 measured variance inputs are also
unavailable. The R/F manifests are prepared templates only; a locally valid
manifest does not confer scientific admission. No synthetic ruler result may authorize
production collection. Actual assignments and their reviewed prices remain the
dispatcher's responsibility under the account-wide supervisor admission rules.

The build handoff, exact commands, measured Mac preflight, input hashes, output
paths and forecast are in `DISPATCH.txt` under the owner's artifact directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/`.
No scientific L4 data or pilot outcomes are produced by this implementation task.

The current Mac preflight, rule/result hashes, manifest validation and forecasts
are recorded in [E8_PREFLIGHT.md](E8_PREFLIGHT.md). Historical PREFLIGHT.md and
INTENT.md remain unchanged. Exports include mandatory `.sha256` and `.complete`
sidecars, including the no-gameplay E9 report, bank/audit and prepared manifests.
