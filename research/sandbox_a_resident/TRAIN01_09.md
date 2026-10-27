# TRAIN-01-09 mechanical reseal

Prepared only; operator dispatch remains a separate action. The consumed identity
`sandbox-a-train01-20261027-08`, packet08 bytes, detached checkouts and attempt08
evidence remain preserved. This successor uses
`sandbox-a-train01-20261027-09` and runtime pin
`c8ab2e6feefc63ce866e1493bfd2f7f1bf3f64dd` unchanged.

## Refusal diagnosis and remaining risk

The attempt08 receipt at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-08/output/receipt.json`
records `INCOMPLETE`, `RPC_EXEC_POLL_TIMEOUT`,
`Refused('kernel remains busy/unknown after telemetry hiccup')`, elapsed
85.039987 seconds, `stop_verified: true`, no cleanup error, no exports and no
terminal workload record. Account assignments were zero before and after; one
A100 endpoint was allocated during staging and subsequently released. The
reported balance stayed at 1021.97 units. This is not a claim of zero allocation
or zero elapsed-rate accounting.

The last input verification finished at monotonic 817542.673023. The next control
exec began at 817542.675900 and failed after 30.014266 seconds. The pinned
supervisor's staging sequence identifies that exec as `deliver_source("fetch")`
(`research/colab_cli/supervisor.py`, lines 1654-1671), before bootstrap or the
training entrypoint. Three recovery-idle probes did not establish an idle kernel.
`upload-measurement.json` reports `staging_completed: false` and
`source_delivery_seconds: 0.0`; the latter accumulator updates only after a
successful call and does not mean source delivery was never attempted.

This need not have been a transient control hiccup: `source.fetch()` allows a
120-second GitHub fetch (`research/colab_cli/source.py`, lines 24 and 245-254),
while staging clips the RPC to 30 seconds (`supervisor.py`, lines 592-605).
The normal depth-one fetch retrieves the full pinned Git tree; the packet's
`source_paths` bounds only sparse fallback delivery. That is a legitimate
long-running staging operation and can explain a busy kernel after the poll
timeout. The receipt has no completed source-delivery record or remote Git log,
so the exact remote cause remains unproven.

No packet-specific runtime defect is established. The qualified transport
unconditionally attempts this fetch and exposes no packet setting to shorten
it or select fallback. No compliant packet-only fix was found. The learning
recipe, telemetry, runtime source pin and `research/colab_cli` are unchanged.
This mechanical reseal does not resolve the transport timeout incompatibility;
another dispatch could encounter it again.

## Packet and operator paths

- Packet: [packet-train01-09](packet-train01-09/SEAL.json).
- Broker: `/Users/josenunez/Projects/ml/snake-dqn-train01-09-broker.noindex`.
- Runtime: `/Users/josenunez/Projects/ml/snake-dqn-train01-09-runtime.noindex`.
- State: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-09/state`.
- Output: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-09/output`.
- Exact command: [DISPATCH.txt](packet-train01-09/DISPATCH.txt).

Only dispatch identity, local packet/input paths, output path, sandbox manifest
binding and reseal provenance change. Control, empty ledger, registry audit,
runtime input digests, all 186 runtime source hashes, scientific namespace,
allocations, endpoint, 480-minute estimate, 960-minute timeout and historical
129-unit runaway guard remain identical to packet08. The original regular
monitoring and nomination rules, update16 diagnostic and detached diagnostics
remain as documented in [TRAIN01_08.md](TRAIN01_08.md).

Verification is recorded in the sibling `VERIFICATION_TRAIN01_09.json` and
create-only artifacts under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-09-reseal-20261027`.
The two new detached checkouts must remain clean and pinned. State and output
remain absent until operator dispatch; dry-run does not allocate or reserve the
dispatch identity.
