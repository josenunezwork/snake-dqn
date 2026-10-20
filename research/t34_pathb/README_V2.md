# Path B v2 — Amendment 2

The v2 implementation is an offline, opt-in assisted hero over released B8 s3.
Read [PROTOCOL_V2.md](PROTOCOL_V2.md) and the prospective
[build details](PROTOCOL_V2_BUILD_RULES.md) before using it. The historical v1 files
and evidence remain intact. The registered runner imports only the v2 adapter,
controller, engine, timing and analysis modules. Serving defaults stay OFF.

`engine_v2.py` maintains a D500 exact continuation, stops its future at hero
death, and searches still-future decisions earliest-first. Windows of at most
eight consecutive branch points and their alternatives share
a native multi-environment simulator batch, retain independent opponent/v8
carry, and use exact rowwise PyTorch CPU forwards. The worker reuses its action
and active-row scratch. Native batch assembly remains a per-window cost.
No v1 or serial LAD fallback is used. ONNX is unavailable in this implementation;
enabling it requires a later explicit backend and gameplay identity gate.

`controller_v2.py` owns one worker and at most one outstanding maintenance/search
request. Every real post-step digest is retained while the worker is busy.
Published predictions permit immediate digest rejection; action execution also
checks session, hero life, checkpoint, generation and exact target frame. Late,
stale, unconfirmed and unavailable results use bare B8. Logs distinguish the
proposal, execution, fallback, horizon rescue and all-die delay.

`adapter_v2.py` uses the T3 captured-world seam for terminal worlds. Its Watch
adapter drives the actual T15 `GameSession` with a separate exact Grid mirror
and fails on physical/RNG, carry or v8 divergence. Product opponents are five
fixed FRP3+v8 policies. Full-tick timing includes this bridge and snapshot/digest
cost; it must not be presented as an already qualified production hook. It
does not start a server, write a score or write release-monitor telemetry.

## Commands

Run from this worktree. Use the existing primary interpreter. Every command
below is local; no tmux, service or cloud allocation is involved.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-pathb2
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python

# Serialize checks that can construct a real background worker during Phase 0.
"$PY" -m pytest -q tests/test_t34_pathb_*v2.py

# Prepare a reused diagnostic roster; this constructs a plan and runs no episode.
"$PY" - <<'PY'
import json
from pathlib import Path
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from research.t3_rollout_labels_20261015.bank_engine import planned_rosters
rows = planned_rosters(gid._context(1), [991827], ["mixed"])
with Path("/tmp/t34-pathb-v2-smoke-input.json").open("x") as stream:
    json.dump(rows, stream, indent=2)
PY

# Tiny real terminal and Watch ON/OFF smoke; create a fresh output directory.
"$PY" -m research.t34_pathb.retention_v2 smoke \
  --input /tmp/t34-pathb-v2-smoke-input.json --frames 2 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-v2/smoke-01

# Tiny timing plumbing check during Phase 0; these are contended observations.
"$PY" - <<'PY'
import json
from pathlib import Path
from research.t34_pathb.adapter_v2 import open_watch
from research.t34_pathb.timing_v2 import run_probes
result = run_probes(
    lambda arm, fps: open_watch({"seed": 991827}, fps, arm),
    ticks=2, warmup=0, contended=True,
)
with Path("/tmp/t34-pathb-v2-timing-smoke.json").open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
PY
```

The following commands are **operator-run**, after implementation verification.
They are not launched by the implementation task. The T3 input must contain the
original 64 dense source trajectories: B8 s0 and B8 s3 over the same 32 scripted
seeds. The pre-check preserves each trajectory's original checkpoint and roster.
These are 64 reused trajectory cases, with 32 unique physical seeds; no
independent-world inference is made. The >=50/64 rule is unchanged. The fresh
registered study uses released B8 s3 exclusively. Every row binds its original
ceiling receipt, checkpoint, arm/env coordinate and materialized source hash.

```bash
"$PY" -m research.t34_pathb.precheck_v2 \
  --out /absolute/path/t3-dense-64-materialized.json

"$PY" -m research.t34_pathb.retention_v2 precheck \
  --input /absolute/path/t3-dense-64-materialized.json \
  --out /absolute/path/fresh-v2-precheck

# Only after >=50/64, Phase 0 has finished, and full pre-registration is committed:
"$PY" -m research.t34_pathb.retention_v2 run \
  --registration /absolute/path/registered-v2-retention.json \
  --precheck /absolute/path/fresh-v2-precheck/precheck.json \
  --out /absolute/path/fresh-v2-retention

# Exactly one registered look; preserve every immutable input/output.
"$PY" -m research.t34_pathb.retention_v2 analyze \
  --out /absolute/path/fresh-v2-retention
```

Real execution requires Mac CPU, AC, open lid and admitted thermal telemetry.
The harness refuses existing output roots. No automatic retry, resume, extension
or replacement bank is provided. Numerical pools are capped before imports.

## Registration schema and evidence

The operator prepares `schema:"t34-pathb-v2-registration/v1"`, `ratified:true`,
`source_commit` (an ancestor of execution HEAD with identical execution bytes),
`source_hashes` from `retention_v2.source_inventory()`, `arms:["bare","paced",
"unpaced"]`, `paced_fps:24`, `terminal_per_mix:32`, `product_sessions:16`,
`horizon:5000`, `workers:1`, `max_elapsed_hours:32`, `max_cpu_hours:48`,
`fresh_disjoint_banks:true`, and the complete materialized `terminal_rows`,
`product_rows`, `timing_row`, `timing_ticks` and `timing_warmup`. Freeze the load
profile and full contract bars in the committed preregistration. The unpaced
descriptive arm uses the same successor with no pacing sleep. Between simulated
ticks the harness holds the simulation clock while its sole worker finishes
maintenance and search, then uses the ordinary action boundary. This reference
has no rewind. The paced arm never waits. Settle time stays outside ON timing
and counts against the study's elapsed/CPU caps.

Set `load_profile:{"quiet_ratified":true}` and `contended:false` only after the
registered quiet-host condition actually holds. Product and timing rows declare
`mix:"mixed"`; the timing seed is also disjoint from every study seed. A nested
terminal `roster` must repeat its outer row's seed and mix exactly.

`checkpoints` is a list of `{path,sha256}` objects, including released B8 s3 and
every frozen opponent, including Watch FRP3 s12. Coverage is checked against
every terminal checkpoint spec before an episode starts. `profile`, `bank`, `seed_audit`, `preregistration`,
`trace_receipt` and `operating_characteristic` are
each `{path,sha256}` mappings. The preregistration bytes must exist in committed
HEAD. The bank JSON must reproduce the registered rows exactly. The seed audit
must pass, report no hits/collisions, and contain exactly the registered seed
union. Mixes and product sessions are disjoint; strict-gate banks stay unused.
The pre-check receipt binds raw paired outcomes and the same source inventory.
Trace and operating-characteristic receipts must bind that exact source. The
operator must produce the study-specific fixed-N survival-band operating-
characteristic check before running; implementation tests do not provide it.
The registered load profile must explicitly ratify quiet-host execution. Record
large deaths at pre-action length >=800 and late deaths at frame >=2000, with
their overlap; these are diagnostics.

The source inventory covers tracked Python in `src`, `research`, and
`web/backend`, configuration YAML, protocol/governance and serving pins. It
includes T3/T33 imports. Changing any executable dependency requires a new
prospective registration; an opaque success flag or CLI exit status is not proof.

## Acceptance table

| Requirement | Verification / deciding evidence |
|---|---|
| Exact batched v8 continuation and B8 tie/mask semantics | Independent reference traces and focused engine tests; see `VERIFICATION_V2.md` |
| One thread, one request, bounded confirmation history, stale/late fallback | Controller queue, identity, mismatch and overflow regressions |
| Real terminal + continuous Watch adapter | Tiny actual adapter smoke, physical/carry/v8 comparison |
| T3 pre-check >=50/64 at 24 fps | Implemented; full run reserved for operator |
| Fresh 32/mix H5000 +16 paired Watch6 sessions | Registered runner implemented; full run reserved for operator |
| P7 paired-world/session inference and Holm family | Analyzer rule and boundary tests |
| First-result p95 <=2 s | Engineering target; no final timing claim from implementation smoke |
| Update p95 <=11.688 ms, zero 12/24 fps overruns | Quiet-host registered probes after Phase 0; contended smoke cannot qualify |
| All contract skill/product bars | Full registered decision; no efficacy claim from smoke |
| Default serving OFF / no release | No serving-selector or pin change |

Passing this implementation's tests does not pass Claim B. A development GO
admits fresh Mac strict-gate preparation; release and wider-roster qualification
remain separate operator decisions.
