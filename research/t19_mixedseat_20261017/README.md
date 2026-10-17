# T19 mixed-seat tooling status (2026-10-17)

This directory records prepared tooling for step 4 of
`docs/research/specs/h100_preruns_plan_2026-10-17.md`. It is not a launch
receipt, training result, promotion result, or serving change.

The `n3` suite in `research/t16_colab/run_smoke.py` contains exactly two
static-pool rows: 128 worlds, fp32 B8 recipe, raw rings, gather threads 3,
three learner seats, and either three or four actors. It uses the trainer's
source-defined frozen six-policy pool. The bundled B8 bytes are explicitly
passed with `--league-b8-checkpoint`; the dispatcher binds all pool identities
to their checkpoint SHA-256 values at startup.

Each completed N3 row must lift the learner-decision (D), replay-row (R), and
update (U) ledger; per-policy live rows; mean and p95 policy lag; and frozen
GPU-forward time. Missing league telemetry makes the sample invalid. N3 is a
measurement-only suite and has no comparison or promotion verdict.

`research/t16_colab/mac_n3_baseline.sh --check` validates and prints the
prepared MPS N=3 command. Without `--check` it is a bounded 420-second run
with 120 seconds of warm-up and 300 seconds of scored collection; it refuses
to overwrite an existing receipt. `research/t18_h100pod_20261017/commands.json`
contains the corresponding unlaunched 30-minute H100 command.

No pod, long training, or tmux job was launched while preparing these files.
