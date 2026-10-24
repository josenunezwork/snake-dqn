# W12 throughput admission implementation

Authority is the complete independent ruling at
`snake-dqn-artifacts/big-lever/exp/JUDGMENT_W12_THROUGHPUT_20261023.md`,
SHA256 `03a7afd1a1ab48dfbb4a2a4fd760ea00d2fe5c8bf6132bfc3660f3077b8815ef`.
The original calibration-01 remains FAIL and SHA256
`082304bc3e94ce48f82313ed285fc7388c099ef4b8d5c1bcc47545cb6fb3cdbc`.
No calibration, factory, timing, strict, serving or long job is run by this task.

`w12_admission` publishes only the canonical create-only
`bl-labels-f0/w12-admission-01/receipt.json` triplet. It binds the exact ruling
and original calibration, rechecks host-sample integrity, and recomputes
`throughput = 2 / w12_w6_wall_ratio >= 1.20` from the sealed measured cells.
Duration, thermal/slowdown fields, platform and original F0 pin are checked.
Every admission use recomputes the full record. An original calibration/v1 PASS
can still authorize W12 under its original bar; the original FAIL alone selects W6.

The D256 continuation keeps the original namespace, admission/control bytes,
allocation, state directory and total 57,600-second wall cap. Each attempt has
one width. RSS above 34 GiB at a five-minute sample, any W12 thermal warning,
or a failed first-three-hour parent sanity check cooperatively drains W12 at
life boundaries and seals a W6 continuation requirement. The operator dispatches
the same resume body with a fresh dispatcher name to continue at W6. No completion
retry is launched automatically. Worker-second charges remain the exact sum
of each attempt's nominal width times its charged wall seconds, including drain.
RSS/sanity fallback is sticky for that series. A thermal fallback permits W12
only after a verified continuous 24-hour clean monitored interval, checked
across canonical dose state directories; an unobserved idle gap is insufficient.

First-hour slowdown baselines restart at width changes. The parent sanity
baseline is frozen when W12 starts and is total completed parents per wall-hour
from the specified matched factory/source stream. For the original D256, its
W6 operational attempt completed 36 parents in 4,463.309045792033 seconds.
The original attempt completed only Watch6 parents, so a separate baseline for
every opponent mix does not exist. Mix counts are descriptive; no per-mix bar
is invented. At three cumulative W12 hours, aggregate throughput must reach
1.15 times the frozen W6 baseline. D1024/D4096 use D256 W6 on the same science
pin or the bound T-EQ W6 timing on a changed pin. Different parent composition
can cause a conservative operational fallback. The rate-drift comparator at
W12 is Canary-B times 1.36; rolling events retain width and do not mix rates
between widths. Per-worker-hour reports are descriptive, split by width, and
never a new F0 look or comparison with the registered F0 floor.

AC, open lid, fail-closed pmset, 30-percent slowdown, 4 GiB per-worker / 40 GiB
total hard RSS, 50 GiB free disk, the Mac lease and timing preemption remain.
The production driver also refuses concurrent Codex CLI or local research
compute and drains if it appears. The Mac is reserved for the factory during
production: finish Sol/Codex work before the operator dispatches it.

Future-dose operational wrappers execute the sealed science pin in a separate,
clean detached worktree through the isolated frozen-factory worker loader.
The current operational commit is attested separately; sealed controls and
T-EQ evidence retain their scientific pin. D1024's registered predecessor/F4
and T-EQ gates remain required. D4096 has its own stage and paths and remains
HOLD pending its separate R6 funding/projection admission; this ruling grants
width authority only.

The independent Codex-V admission review passed and the canonical record was created.
Its SHA256 is `65ea988e086ea1c26fc13dba505f4aae7ba524d443095def77ba10c1b80fa4fe`.
The creation command below is retained for provenance; rerunning it is refused:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f0
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_labels.w12_admission \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json
```

An external operator started W6 D256 from `6498f992` at 09:59:53 during
verification. It was not started or modified by this implementation. Wait for
its verified terminal receipt before using the next resume command; a live
start without a terminal is deliberately HOLD.

Operator dispatches use the final published operational implementation SHA,
not the original scientific SHA. Substitute the full SHA delivered in the
implementation report for `IMPLEMENTATION_SHA`. Every name must be fresh.

```bash
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d256-resume-w12-03 IMPLEMENTATION_SHA -- \
  /bin/bash research/bl_labels/RUN_D256_RESUME.sh

# After D256/F4 predecessor and T-EQ admission, use the actual immutable staging path.
: "${F4_INPUTS:?Set the registered immutable D1024 F4 staging directory}"
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d1024-w12-check-01 IMPLEMENTATION_SHA -- \
  /usr/bin/env F4_INPUTS="$F4_INPUTS" F4_CONTROL="$F4_INPUTS/control.json" \
  /bin/bash research/bl_labels/RUN_D1024.sh --check
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d1024-w12-01 IMPLEMENTATION_SHA -- \
  /usr/bin/env F4_INPUTS="$F4_INPUTS" F4_CONTROL="$F4_INPUTS/control.json" \
  /bin/bash research/bl_labels/RUN_D1024.sh

# This check remains HOLD until a separate registered D4096 R6 admission exists.
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d4096-w12-check-01 IMPLEMENTATION_SHA -- \
  /usr/bin/env F4_INPUTS="$F4_INPUTS" F4_CONTROL="$F4_INPUTS/control.json" \
  /bin/bash research/bl_labels/RUN_D4096.sh --check
```

Owner notice: W12 calibration measured 1.36× wall throughput with 0 thermal;
the 1.30 per-worker bar measured E-core speed, not heat, so the judge replaced
it with ≥ 1.20× throughput plus unchanged thermal guards; any thermal warning
returns the factory to W6.
