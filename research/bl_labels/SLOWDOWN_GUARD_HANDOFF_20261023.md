# D256 windowed slowdown guard repair

Binding ruling: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_SLOWDOWN_GUARD_20261023.md`,
SHA256 `e50a573e69e6aeea1de7ddde590bac36e62a28a46c3399f5bce0182ae4109551`.
This composes with the W12 throughput admission ruling at SHA256
`03a7afd1a1ab48dfbb4a2a4fd760ea00d2fe5c8bf6132bfc3660f3077b8815ef`.

The operational driver tests the median of the last `4 * admission_width`
eligible lives against `1.30 * B_w`, separately for each stream. Each attempt
starts with an empty window. Replay first lives, width-straddling lives and
stop-in-flight lives are excluded; fresh first lives and arbitrarily short
lives remain eligible. Consecutive life boundaries restart the allocation's
timer in the same attempt. A stopped timer drains queued messages safely.

Baselines collect eligible boundaries in the first 3,600 cumulative active
stream wall-seconds at each width, extending only to obtain at least 60 lives.
The original D256 W6 baseline uses exactly the 136 original first-hour samples.
Every frozen median is published create-only with a SHA256 sidecar and
completion-last marker in the existing state directory **when the operator
runs the continuation**, never by this implementation task. Frozen baselines
are verified and reused, and never extended. W12 baseline sanity uses the
same stream's B6 or D256's B6 and the 1.911 multiplier; a new stream also uses
the most recent same-pin baseline at its width with the 1.30 multiplier.
Newly frozen same-attempt stream baselines update those references immediately.
Baseline sanity failure refuses the freeze, preempts and requires escalation;
a W12 failure first requires W6. A W6 sanity failure remains HOLD for a judge.
The old `first_hour_life_seconds*` receipt fields are retained legacy metadata;
`slowdown_state_by_stream` and the immutable baseline files govern the repair.

A trip records the exact window-median telemetry, stop reason and per-attempt
`slowdown_trips` count. Two same-stream/width trips within six charged stream
wall-hours require a W12-to-W6 cooldown (the same verified 24-hour clean proof
as the thermal policy), or a logged host check after W6 trips. The cooldown
propagates across canonical dose states. Legacy simultaneous notifications
from one stopped attempt count as one episode, with epoch derived from its
sealed start and wall charge. Both original W6 episodes therefore require the
host check before a requested W12 continuation too. The explicit operator
`--host-check` option logs pmset thermal state, top CPU consumers and absence
of Sol/Codex before the resume admission. No automatic resume is introduced.
A preserved thermal reduction below six admissions uses that effective width
for the guard; the existing nominal attempt accounting is unchanged.

Original control, namespace, allocation, state path, verified union, cumulative
wall cap (57,600 s) and exact sum of nominal attempt width times wall charge
remain unchanged. W12 admission still rechecks the canonical
`w12-admission-01/receipt.json` (SHA256
`65ea988e086ea1c26fc13dba505f4aae7ba524d443095def77ba10c1b80fa4fe`).
The original calibration remains FAIL. RSS, AC/lid, disk, timing preemption,
thermal, parent-throughput sanity and sealed rescue-rate guards remain in
force. D256 scientific collection still executes in the separately validated
clean detached `61900af28932e47e22e332840fc4406559adb062` tree.

Finish all Sol/Codex activity before operator dispatch. Use the full published
implementation SHA from the final report and a fresh dispatcher name:

```bash
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d256-guard-w12-01 IMPLEMENTATION_SHA -- \
  /bin/bash research/bl_labels/RUN_D256_RESUME.sh \
  --host-check \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json
```

The host check, any newly frozen baseline, union ledger, next start/terminal
receipts and next output directory are created only by that future operator
command. All prior files stay immutable. A missing terminal, failed host check,
missing/tampered admission, cap exhaustion or registered stop remains HOLD.
No calibration, production factory, strict, serving or long job was launched
by this implementation.

Owner notice: D256 slowdown guard compared single lives (10 % exceed 1.30x by chance) and self-armed on every resume; judge replaced it with a 1.30x test on the median of the last 4xW fresh lives vs a per-width first-hour median; thresholds unchanged; attempt-1 data never exceeded 1.04x.

Independent Codex-V light verification passed all six ruling obligations: 81
tests in 21.56 s, then 6 overlapping state tests in 2.55 s after the final
no-recomputation delta. The timestamp fixture has 253/23 original attempt
events, independently matched to source receipt hashes. The 1.35x case trips
after 20 eligible boundaries (K24); 1.20x never trips. Preempt/resume opaque
life/session JSON is byte-identical to uninterrupted collection. Exact commands,
reviewed hashes and limits are in `CODEX_V_SLOWDOWN_GUARD_20261023.json`.
Black, isort, flake8, shell syntax, CLI help and read-only W12 admission
recomputation pass. No canonical artifacts were written: the pack, admission
and calibration inventory remains 2,110 files with metadata digest
`8921c11673b42e96f09996d6b9adbcccf52bf0847bd48ee3c948aa67ef9b5c23`.
