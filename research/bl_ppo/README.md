# BL PPO: E4 collector ruler followed by the E2 pilot

Development only, excluded from every registered verdict. Binding local rules are
[INTENT.md](INTENT.md), committed before gameplay in `3bedc5de`.
The design comes from the owner's `big-lever/round-01/JUDGMENT.md`, section (f).

`model.py` copies B8's ego2s-bv8 encoder and preserves a frozen calibrated B8 policy
as the base logits. A zero-initialized residual actor learns around that base.
Arm R uses GRU256; arm F uses an exactly parameter-matched feedforward core.
`learner.py` implements finite-horizon undiscounted GAE/PPO and critic refit.
`collector.py` keeps GridBatchSim and the immutable opponent policies in CPU
processes while the parent batches hero inference on the learner device.
`run.py` is the real entrypoint for the ruler and training.

`exports.py` publishes immutable payloads with SHA256 sidecars and completion
markers recognized by the unchanged supervisor. Periodic full-state checkpoints
preserve optimizer, collector and RNG state. Incomplete rollouts are never promoted
to scientific endpoints. Checkpoint exports and terminal training-status receipts
are separate evidence; a weights file alone does not prove completion.

`bank.py` allocates the fresh 96/32/32 screen and excluded fixtures, audits the
published history without opening sealed live outputs, and binds the audit.
`screen.py` runs/analyzes paired fixed endpoints and applies only E2's frozen rule.
A development GO permits scale planning; release still requires the Mac strict
gate, serving qualification, and the owner's authorization.

`manifests.py` prepares local supervisor manifests; `colab.sh` is their pinned
runtime entrypoint. It contains no allocator. E4 must run first. E2 templates are
locally validated preparation artifacts and deliberately fail closed without a
complete E4 receipt and budget forecast. No synthetic ruler result may authorize
production collection. Actual assignments and their reviewed prices remain the
dispatcher's responsibility under the account-wide supervisor admission rules.

The build handoff, exact commands, measured Mac preflight, input hashes, output
paths and forecast are in `DISPATCH.txt` under the owner's artifact directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/`.
No scientific L4 data or pilot outcomes are produced by this implementation task.
