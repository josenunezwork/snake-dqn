# V0 Mac ladder — Sol-K handoff

The frozen rules are in [REGISTRATION.md](REGISTRATION.md), committed/published
before V0 target-platform data. This package leaves K-1, the shared ruler and
all protected T34 packages unchanged. C1 is re-timed here; K-1 sample reuse is
deliberately avoided because windows/host state differ.

**Execution note:** smoke-01 did execute four real frozen-800 cells, but its
live-job exclusion check missed the active K-1 audit because macOS truncated
the COMM column. Its create-only INVALIDATION.json revokes admission. The
guard now also checks the executable in ARGS and checks again after each
batch. Use prepared-02 and a fresh smoke-02 after K-1/X5 exit. This is a
mechanical execution proof only; smoke-01 supplies no latency qualification.

`operator.py` subclasses ExactT3Operator, retaining the original dense selector.
The scoped rollout adapter supplies C1/all, C2/stop, C3/prefix64_oracle and
C4/prefix64_oracle_stop. Oracle decisions use each repeat's preceding C1 H300
outcomes. Actual prefix frames and skipped branches remain separate from oracle
outcomes. The fixed dummy leaf is CPU FP32 128→64→64→6, 12,806 parameters, one
legal-count call per oracle cell, with real leaf featurisation charged.

The launcher pins all numerical thread counts to one, CPU, nice 10, AC/open lid,
thermal guards, a shared OpenMP runtime, a CPU slot and caffeinate. The Python
entrypoint owns the atomic common Mac lease, rejects active K-1/X5 processes
(including audit), checks windows and logs loads. It never kills another job,
steals a stale lease or automatically retries. Receipts and sidecars are
create-only; use a new output directory for any separately authorized attempt.

## Commands for Claude

Run from the handed worktree. Prepare only after all intended source changes
are published, then check. Preparation reads the original hash-pinned nine Y1
roots; it generates no rollout/timing data. If any source bytes change, prepare
a new manifest and obtain a new matching Mac canary.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-wp2
git pull origin main
/Users/josenunez/Projects/ml/snake-dqn-wp2/research/bl_v0/run_mac.sh \
  --prepare \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/prepared-02
/Users/josenunez/Projects/ml/snake-dqn-wp2/research/bl_v0/run_mac.sh \
  --check \
  --manifest /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/prepared-02/manifest.json
```

Bounded target-platform canary: **only when K-1/X5 have exited and the common
lease is free**, outside their protected windows. Four batches, frozen-800,
one repeat, all cells, ≤180 seconds. PRECHECK_ONLY, never timing qualification.
The run acquires/releases the lease itself; do not wrap it in a second lease.

```bash
/Users/josenunez/Projects/ml/snake-dqn-wp2/research/bl_v0/run_mac.sh \
  --smoke --operator Claude \
  --manifest /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/prepared-02/manifest.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/smoke-02
```

Real timing: Claude only, **Fri Oct 23 22:00–23:00 PT or Sat Oct 24
08:00–10:00 PT**, ≤3,600 leased seconds, first window/budget cap wins. The K-1
paths below identify the current attempt-03; use that attempt only after its
measured timing receipt exists. A different K-1 attempt must supply its matching
manifest, full identity/cases and raw timing batches. No K-1 baseline value is
required beyond a valid measured receipt; C1 is measured again here.

```bash
/Users/josenunez/Projects/ml/snake-dqn-wp2/research/bl_v0/run_mac.sh \
  --operator Claude \
  --manifest /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/prepared-02/manifest.json \
  --canary /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/smoke-02/smoke.json \
  --k1 /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-k1/attempt-03/timing.json \
  --k1-manifest /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-k1/prepared-03/manifest.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/attempt-01
```

Read `timing.json` or `stopped.json` and verify `.sha256` and `.complete`.
Per-batch exports contain lease id, root identity, chosen action, evaluated and
NOT_EVALUATED branches, actual frames, branch native counters, leaf wall and
exclusive stage timers. `metrics` retains each cold value, all30 and warm29
distributions; `operator_wall_seconds` is the complete cap metric, including
residual Python work. PASS admits only V1 registration, never serving/build.

## Prospective ruling needed before a scientific PASS

The two conflicts are frozen and disclosed in REGISTRATION.md. Literal §5(c)
uses C3 median <2×C2 median for NO_CONSUMER, despite the summary's opposite
speedup reading. Scripted fixtures have no eligible opponent-v8 rows, so their
truthful zero counters make the blanket positive-row rule INVALID. No scripted
exemption or alternate inequality is silently applied. Obtain a prospective
sibling ruling/amendment before affected timing data if either rule is to change.
Running the literal full ladder retains all 1,080 records but cannot yield PASS
with zero-eligible scripted counters.

## Forecasts and evidence boundary

Round-13 projection: **35–50 minutes**, **hard cap 60 minutes**, 1,080 batches,
0 Colab units and 0 upload bytes. This is NOT_READY(V0), not a measured speed
forecast; the four-batch smoke cannot qualify it. Builder forecast 0.5–1 day;
independent review 0.25 day. L2′ value remains NOT_READY(L2′).

Sol-K does not run the real ladder. Active K-1/X5 work takes priority over the
short canary too. Targeted unit tests cover semantics and fail-closed accounting;
the Mac canary must have an actual `smoke.json` with `canary_ok: true` before
the real timing entrypoint will admit a run.
