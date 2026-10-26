# Sandbox B-2 packet-02: Argon corrections

This prospective correction supersedes the loss and gate descriptions in `B2.md`
for `b2-packet-02` only. The source-bound `b2-packet-01` remains immutable.
Authority is the same Standing A100 Development Sandbox rule. This is preparation
for simulator-D development evidence; no live dispatch or release is performed.

Argon's full review is at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/review-sbx-b2/answer.md`.
Both MAJORs and all three MINORs are addressed:

- Survival pairs use decisive rows only. Pairs are averaged within each valid
  decisive row, then rows are averaged equally.
- The supervised head uses the same detached sigmoid gate during train and eval.
  Balanced BCE alone trains its gate, with positive weight `negative / positive`
  over supervised rows when both classes exist; single-class batches use ordinary
  BCE. Gate logits are computed once per optimization step.
- Survival losses use gated corrected Q. Ordinary KL and hinge use base Q plus
  the **ungated** body, eliminating quadratic gate attenuation of body regularization.
- KL normalizes over selectable actions only and safely clears excluded log
  probabilities before subtraction, avoiding NaN gradients.
- Production and tests share a cached sampler. Life weights are computed within
  the training split. Balanced draws retain exact halves and the deterministic
  odd remainder in the complement, including batches of one.
- Smoke selection is outcome-blind round-robin across first-seen parent groups,
  taking additional rows per group until the requested count or all rows is reached.
- Checkpoint metadata records `gate_function: detached_sigmoid` for the gated
  head and null for other heads. There is no hard gate threshold. The existing
  post-hoc action-switch thresholds remain `0, 0.1, 0.2, 0.4, 0.8`.

The registry, D1024 input bytes, frozen B8, parent split salt
`a100-dev/sandbox-b/split-v1`, namespace, allocation, 48 configurations,
48,000-update live endpoint, 240 evaluation points, optimizer settings, and
60-minute base / 120-minute effective runaway guard are unchanged. Source hashes
change the recipe fingerprint; packet-01 units cannot be resumed into packet-02.
The new packet begins with a freshly generated empty verified ledger.

Tests cover non-decisive pair exclusion, unequal per-row pair counts, empty losses,
identical train/eval gates at positive and negative logits, BCE class balance and
single-class batches, no Q-loss gate gradients, unattenuated ordinary-body gradients,
masked finite KL gradients, shared sampling, checkpoint metadata, round-robin smoke
selection, and real entrypoint exports and resume. The real admitted-input CPU
smoke uses 600 rows and three updates per configuration. This outcome-blind prefix
contains decisive and safe-base rows but no ordinary rows; synthetic entrypoint and
mechanism tests exercise ordinary preservation directly.

The current operator commands and source pin are in `b2-packet-02/DISPATCH.txt`;
hash-bound verification evidence is in `b2-packet-02/VERIFICATION.json`. The sandbox
manifest has ten physical lines and is committed before preflight. Offline preflight
and dispatch dry-run do not inspect live account capacity or establish protected-study
coexistence. The qualified `research/colab_cli` transport is unchanged.
