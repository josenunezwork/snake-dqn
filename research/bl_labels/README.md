# F0 hindsight label factory

F0 plays untouched bare B8 lives, reads the bare H300 outcome from the natural
trajectory, and spends branch simulation only on alternative actions and the
selected bare audits. It creates development data; no gate, model fit, timing
qualification or serving change is performed. The frozen authority, allocation,
caps, retry policy and kill rules are in [INTENT.md](INTENT.md).

## Exact execution and exported evidence

`factory.py` composes B2 `WatchDriver.snapshot/restore`, the T3 gate-world
constructor, `clone_world_exact`, `full_digest`, native v8, and `exact_stop.run`.
All streams use the same prepared maintenance/selector seam. Watch6 has bare
B8 against five independent FRP3-s12+v8 opponents with continuous respawn.
S-gate preserves the L2 constructor's frozen/mixed/scripted populations,
including their disabled gate veto seats; zero eligible native-v8 rows in those
gate worlds is expected. Watch opponent rows must attest native-only execution.

The fatal action has zero-based index `e`; the exclusive death endpoint is
`f=e+1`. Doom roots are in `[f-300,f)`, including the fatal action itself.
The 60 stride roots union the last 30 roots normally gives 84 distinct roots,
under the doom cap of 90. Eight ordinary roots are separate from that cap.
They are a uniform bottom-hash sample of the same life's known-safe natural
H300 roots. Their probabilities and capped natural-prevalence weights are
recorded. Exact outcomes include fixed-H300 mass with zero mass after death.
T3 ranking retains bare among surviving ties and chooses longest delay if
every branch dies. Skipped actions remain `NOT_EVALUATED`, with null outcomes.

Snapshots remain in a ring covering 420 decisions, plus the eight ordinary
reservoir roots and their short replay prefixes. Every restored selected root
must match its live full digest. B2 restore also restores process RNG and
configuration. Label branches cannot change the continuing natural stream.
No simulator implementation or shared simulator files are changed.

Each death-life exports immediately as `lives/<parent>--life-NNNNNN.json`,
with SHA sidecar and completion marker last. The terminal session exports as
`lives/<parent>--session.json`. Session exports contain the shared natural
six-seat action tape, per-decision Q/legal/length, digest cadence, UNKNOWN
short suffixes, audit-root metadata, timing buckets and native attestation.
Raw raster/scalar inputs retain their original values. Every row binds its
namespace, parent world, life, code and checkpoint identity, and parent split.
`audit-sealed` parents remain reserved from consumer fitting until the consumer
model is frozen; F0 performs no fitting or admission of a student.

Ten percent of doom roots get independent bare audits of death and final length.
Last-30 roots and a ten-percent hash sample of stride roots get complete legal
outcomes; ordinary roots are complete. Ten percent of rescue roots get H1440
continuation. The Mac aggregator derives a tranche-wide `delay` effective label
if reversals exceed ten percent, leaving immutable raw outcomes intact.

## Prepare and dispatch the CPU canary

The current frozen source pin, verified artifact hashes, Codex-V receipt and exact
Claude commands are in [HANDOFF.md](HANDOFF.md). The transport allocation is
frozen in [FAMILY_PROBE_EXECUTION.md](FAMILY_PROBE_EXECUTION.md), with the
owner-authorized execution envelope replaced by
[SESSION_CAPS_AMENDMENT.md](SESSION_CAPS_AMENDMENT.md).

Preparation requires a clean published source pin, the qualified CLI tree,
six pinned public checkpoint identities and a clean Python 3.12 import
preflight via `tools/ops/import_preflight.py`. It creates an excluded family
probe and an unadmitted scientific template with identical input inventories.
The original scientific attempt carries an empty verified ledger.

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd /Users/josenunez/Projects/ml/snake-dqn-f0
PIN=$(git rev-parse HEAD)
"$PY" -m research.bl_labels.manifests \
  --commit "$PIN" --artifact-dir /absolute/fresh-pack \
  --output-dir /absolute/fresh-output \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --frp3 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth
```

Claude dispatches only through
`/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh`.
First run `f0-family-probe.json`: CPU, 20 minutes, 0.04 units. Its separate
H3 namespace emits no life labels and cannot contribute to scientific aggregates.
After its actual verified PASS receipt and backend release, bind that receipt:

```bash
"$PY" -m research.bl_labels.manifests bind \
  --template /absolute/pack/f0-cpu-canary-template.json \
  --receipt /absolute/probe-output/receipt.json \
  --out /absolute/pack/f0-cpu-canary.json
```

Only the receipt-bound manifest admits the full F0 canary: 12 frozen + 12 mixed
+ 8 scripted H5000 worlds and four Watch6 H14400 sessions, CPU, W=nproc,
one numerical thread per worker. It has 120 minutes / 0.20 units / 0.040 reserve;
the excluded probe has 20 minutes / 0.04 units / 0.008 reserve. The combined
base/effective replacement envelope is 70/140 minutes, 0.12/0.24 units,
0.024/0.048 reserve, doubled exactly once. Base whole-session forecasts are
10 minutes for the probe and 60 minutes for science, including setup, useful
work, export and margin. The runtime reads the qualified watchdog window and
subtracts measured setup plus the 300/540-second effective export/margin
reserve; it never starts a separate full-session work timer. It exports those
operands in `session-budget.json`, and ledger verification consumes the same
absolute collection deadline.
The measured accounting rate is 0.08 u/hour. Setup, export, failures and manual
mechanical completion attempts share these limits; first cap wins. There is
no automatic retry, purchase or cap reset. The designated broker must match
the qualified CLI registry before either dispatch.

## Verify, resume, aggregate, and audit

Use only Mac-verified payload/SHA/completion triplets. Ledger construction is
outcome-blind. Aggregation is the registered outcome look and must wait until
the frozen allocation is complete or mechanically closed.

```bash
"$PY" -m research.bl_labels.exports ledger \
  --control /absolute/pack/control.json --exports /absolute/verified-attempt-output \
  --out /absolute/fresh-ledger/verified-ledger.json
"$PY" -m research.bl_labels.exports aggregate \
  --control /absolute/pack/control.json \
  --attempt-exports /absolute/verified-attempt-output \
  --out /absolute/fresh-aggregate/verified-ledger.json
```

The aggregator refuses duplicate IDs, foreign namespaces/allocations, partial
payloads and durable INVALID markers. It materializes verified payloads beside
its ledger. Completed sessions skip their parent entirely. A partial parent
replays its deterministic natural discovery to reach missing lives, skipping
every verified life's label computation and export. This is completed-unit
resume, not a claim of full-state continuation of an interrupted trajectory.

The independent Mac audit replays all six recorded actions, never x86 policy
decisions. It classifies Mac-side policy ties at four float32 ulps, checks each
replayed simulator digest and requires 64 distinct roots before reporting PASS:

```bash
"$PY" -m research.bl_labels.replay_audit \
  --unit /absolute/verified/lives/parent--session.json \
  --unit /absolute/verified/lives/another-parent--session.json \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --frp3 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth \
  --out /absolute/fresh-audit/mac-replay.json
```

Any digest/bare-audit mismatch is INVALID. A completed zero-rescue canary is
INSUFFICIENT, with no extension. The throughput kill is fewer than eight
rescue-bearing lives per Mac-worker-hour equivalent. Colab hours cannot be
silently substituted for Mac hours; only a measured Mac ruler or a separately
measured conversion can supply `aggregate --mac-worker-hours`. Full Mac Canary
B (W2, nice 10, at most one hour, unleased quiet gap) remains an operator step;
the tiny implementation smoke does not establish throughput. Strict gates and
serving qualification remain Mac-only and separate.
