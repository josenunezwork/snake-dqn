# T34 Path B deadline-feasibility check

**V1 infrastructure only; v1 efficacy is retired.** The owner-approved
[design v2](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md) supersedes
this H40 controller with shadow trajectories and LAD search. Do not prepare or
run a v1 efficacy bank. M1–M4 measurements and the v2 rework are separate work.
Only implementation smoke checks have been run here; they yield no efficacy
estimate. Controller bounds, paced banks, timing tools and assisted identity
remain available for that later rework.
The later owner-ratified [P7 rules](../../docs/research/final_run_20261019/P7_STATISTICS.md)
also supersede v1's statistical frame: fixed-controller inference uses paired
worlds (at least 16 per mix), and product inference uses complete paired sessions.
The frozen v1 analyzer is historical tooling and cannot establish P7 compliance.

The implementation is **development tooling, default OFF**. It neither qualifies
nor releases an assisted hero. Read [PROTOCOL.md](PROTOCOL.md) and its prospective
[timing clarification](PROTOCOL_TIMING_ERRATUM.md) and
[analysis clarification](PROTOCOL_ANALYSIS_ERRATUM.md). All preceded real efficacy data.
All commands below run from this worktree. No command starts a service, uses tmux,
or changes another running job.

## Setup and tiny smoke

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-pathb
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pip install -r research/t34_pathb/requirements.txt

env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run smoke \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/owner-smoke-01 --smoke-key owner-smoke-01
```

The smoke reads [smoke.json](smoke.json): one world per terminal mix plus one
continuous Watch6 world, three arms, both 12/24 fps, H4, one warmup and four timing
ticks. It forces the trigger at every living decision with a two-transition search
so short-body plumbing actually exercises the worker. This is a different smoke
controller, never an efficacy candidate. Startup misses count; no accepted result
is required for an asynchronous smoke. Unit tests separately prove timely acceptance.
Use a new explicit smoke key for a later plumbing check; keys never alter the real bank.
Outputs must be fresh absolute paths; manifests, logs, starts and results are
create-only. A failure preserves its evidence and does not automatically retry.

## Retired v1 bank command reference — do not execute

The frozen v1 entrypoints are retained as implementation documentation, not an
execution recommendation. This example uses five independent blocks with one
world in each group. It is a small, noisy development example, with no power or
acceptance claim. The three terminal mix counts and product count would have
been fixed **before** preparing a new bank.
The production action rule stays fixed at length >=600, every fifth target frame,
horizon 40. `--workers 1` is enforced: one parent and one single-thread CPU worker.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-pathb

env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run estimate \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/sizing-only \
  --blocks 5 --scripted-worlds 1 --frozen-worlds 1 --mixed-worlds 1 --product-worlds 1 --horizon 5000

env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run bank \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/dev-01 \
  --workers 1 --blocks 5 --scripted-worlds 1 --frozen-worlds 1 --mixed-worlds 1 --product-worlds 1 \
  --horizon 5000 --max-hours 24 --warmup 200 --ticks 1500

/usr/bin/caffeinate -dims env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run efficacy \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/dev-01

/usr/bin/caffeinate -dims env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run timing \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/dev-01

env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.analyze \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/dev-01
```

Execution reads sizing, caps, timing counts and smoke identity from the frozen
manifest; supplied conflicting flags are refused. The bank must be prepared from
a clean, committed tree. Source/config/protocol bytes and every checkpoint are
hashed and rechecked before gameplay and analysis. Seeds are allocated once and
collision audited against visible source/docs/scripts/artifacts and the historical
FRP training exclusions. Any collision rejects the whole allocation. No world is
silently substituted. Seed audits disclose binary, ignored and external-history
coverage limits. Terminal rosters use the historical deployment profile and exact
ordered specs; product worlds use real mixed Watch6 with FRP3+v8 opponents.

The small example's **sleep-only floor** is 10.417 wall-hours, or **13.889 reserved
mathematical-thread core-hours** (bare 1 + async 2 + sync 1 across three arms).
Defaults of 5 blocks × (96+48+48+25) worlds cost at least **565.104 wall-hours /
753.472 reserved core-hours**. These are capacity estimates, not measured active
CPU use: setup, serialization, ticks and synchronous unlimited searches add time.
Per-arm parent CPU and elapsed time are recorded; the smoke is too short and its
body/search regime too different to price real long-body search. Any future v2 bank
needs its own prospective protocol and sizing. Cap
expiry means `INCOMPLETE`, with no resume or automatic extension.

## Historical v1 evidence mapping against SUCCESS_CONTRACT

| Contract | Produced evidence | Required acceptance |
| --- | --- | --- |
| §2 H5000 mass | Per block, equal three-mix paired gains; one-sided Student-t LB90; 12/24 separate | Mean >=+40 and LB90>0 versus bare B8 |
| §2 survival/growth | Scripted no-death delta, every registered survival guard, actual boost share | Scripted delta >=.03; all guards; boost [.005,.10] |
| §3 persistence | Continuous hero deaths per 1k actual decisions; paired 95% interval | Delta upper95<0; target >=20% fewer deaths reported |
| §3 mass/length | Per-world paired relative full-horizon mass; median/p95 length; late/large deaths | Relative mass lower95>=-.05; length/death diagnostics retained |
| §4 processing | T15 real step+snapshot timer, p95/p99, ON/OFF at 12/24 | Mixed Watch6 update p95<=11.688ms; zero processing overruns |
| §4 cadence/tails | Delivered cadence; natural L>=600 and busy-worker distributions | <=5% cadence loss; missing tails explicitly `NOT_MEASURED` |
| §4 client latency | Local full-tick p95 proxy, separate from browser/network latency | Proxy <=10% worse; true client latency `NOT_MEASURED` pending qualification |
| §4 identity/resource | One bounded request/process; frame/life/session/checkpoint/world guards; decision JSONL | No wait at action selection; all misses/fallbacks included; visible assisted label |
| Strict gate / release | No gate worlds or publication change | Fresh Mac strict gate and serving qualification versus B8 s3 under P7; owner release approval |

Analysis emits `INCOMPLETE`, `SMOKE_ONLY`, or development-bar pass/fail; none is a
strict pass. Uncertainty with fewer than two independent blocks is not estimable.
The unlimited arm only diagnoses deadline losses. Natural long-body/contention
samples can be absent in a short probe, and then cannot qualify those tails.
The harness does not measure browser/network client latency; that acceptance row
remains open even if every local development bar passes.

The opt-in product flag is `SNAKE_SERVE_PATHB=1`, eligible only for registry-verified
mixed Watch6 B8 slot zero. Assistance is labelled even if all requests fall back.
Watch8/12, Play, Train and fallback checkpoints retain their existing path. On the
next session construction, unsetting `SNAKE_SERVE_PATHB` and selecting
`SNAKE_SERVE_CHECKPOINT=b8-s3` disables assistance. No running server is restarted
by these tools. Per-decision `pathb-decision` records contain requested/applied,
stale/miss/fallback, bare/proposed/executed actions, frame and hero life; harness
JSONL logs are hash-bound to the paired records.
