# PLAN_v3 C1 operator handoff

Implementation only; no C1 sessions, timing cells, production work, or serving changes have run.
Authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/PLAN_v3.md`
SHA256 `d23690617519262bd1b1a3e6c5d8b414ef952003b5b92c86842c9f5fa4210201`.

Claude operates all runs. Use this worktree's launcher, which supplies CPU, all five
one-thread environment limits, checkpoint directory and nice 10. `--run` acquires the
shared primary checkout's `exp/mac-lease.json` process-live lease, creates the idle
receipt through `research.bl_mac_latency.host.snapshot`, refuses a busy Mac, starts
caffeinate, monitors AC/lid/thermal/foreign load, and starts exactly two one-thread
processes. Each process executes both arms of one pair on the same CPU. This is
unpaced development evidence, never latency, strict, serving, or release evidence.

## Exact commands

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-c1
bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --check

# First attempt only: freezes source/allocation identity and freshness proof,
# then publishes the required empty verified ledger. No games start here.
bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --make-ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/ledger-01.json

bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --check --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/ledger-01.json

# Claude dispatches in the reserved exclusive Mac window. Keep its terminal open,
# or place this exact invocation in a dedicated Claude-owned tmux session.
bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --run --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/ledger-01.json
```

A completed unit is `attempt-01/session-NN.json` plus `.sha256` and `.complete`,
followed by `session-NN.ops.json` and its own sidecars. The outcome-free operations
record is published last and binds namespace, allocation, source identity, and
payload path/size/hash. Resume verifies that record before skipping a pair.
It contains BOTH full 14,400-frame arms, actual decision rows, death/life/mass
chronology, Q/masks/delivered actions, dense branch outcomes and native counters.
Only completed pairs enter the ledger. Incomplete pair work remains charged and
is never stitched, substituted, or analyzed. The coordinator reports only pair
completion counts. All 24 hashes and allocations must verify before it opens raw
outcomes and validates the full evidence. It then performs the sole registered
look and publishes `decision.json`; `look-started.json` prevents a repeated look.
Never run a standalone reducer on partial exports.

One manual outcome-blind mechanical completion attempt is prepared. Only use it
after `attempt-01/receipt.json` verifies `INCOMPLETE`, `verified_stop: true`, and
`mechanical_failure: true`, with missing units and remaining cumulative budget:

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --make-ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/ledger-02.json --previous-attempt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/attempt-01/receipt.json
bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --check --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/ledger-02.json --previous-attempt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/attempt-01/receipt.json
bash /Users/josenunez/Projects/ml/snake-dqn-c1/research/bl_served_search/run_c1.sh --run --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/ledger-02.json --previous-attempt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c1/attempt-01/receipt.json
```

Recovery skips exactly the receipt's path/size/hash-verified pairs. Foreign
namespaces, changed sources/allocations, duplicate units, corrupt exports and a
third attempt are refused. Hard coordinator death without a verified stop receipt
needs operator/judge reconciliation; it is never auto-retried. A second mechanical
failure goes to the judge. No extension is created by this package.

## Frozen design and raw endpoint interpretation

The real Watch6 and its independent Grid mirror run continuously with respawns.
Hero is bare B8 s3 versus unchanged dense B8: length 600, every fifth decision
anchored at the first eligible decision of each life, exact-stop full H300,
all legal first actions, unchanged `_choose`, bare first then Q/index order.
Reacting opponents are five exact FRP3 s12 + v8 policies. Each branch clones its
own simulator RNG and selection/trend/v8 carry. Native mode is kernel, with
positive kernel rows and zero stock/fallback/noncandidate branch rows per batch;
root live opponents use their unchanged stock wrappers. Every real Watch tick
must agree with its mirror on physical state, RNG, carry and v8 counters.

Each session's death rate uses its own actual decision count in each arm.
Relative mass is dense full-horizon mass / bare full-horizon mass minus one.
The paired samples are 24 session effects with one-sided Student t 90% bounds,
23 degrees of freedom. Distinct witness lives are deduplicated at their first
local H300 rescue. Same-life re-death lags are measured from that first witness;
raw frame/life/death rows disclose right censoring at the session horizon.
The raw evidence also discloses local negative H300 death witnesses and candidate
only death frames while the paired bare arm is alive. The latter are descriptive
paired-frame events: after trajectories diverge, they do not establish causation.
The saved seed and complete delivered-action chronology support deterministic
reconstruction before a later B3 witness replay; this package does not run B3.

STOP is checked for a mass UB90 below -5%, or a nonnegative death-rate LB90 with
at least eight baseline deaths. Fewer than eight baseline deaths otherwise gives
INSUFFICIENT. GO requires a negative death-rate UB90, mass LB90 at least -5%, and
witnesses in at least four sessions. All other valid complete banks are
INCONCLUSIVE. GO admits only the development chain; STOP closes its product
premise; INCONCLUSIVE continues the chain but withdraws the Nov 7 claim until L3.
No label authorizes a serving change.

## Budget (already doubled; never double again)

Forecast/base: **8 unpaced Mac wall-hours**. Effective cumulative cap:
**16 hours = 57,600 seconds**, across the original and the one mechanical completion
attempt, including setup, failed/incomplete work and teardown. W2 is two processes,
not two independent 16-hour allowances. Neither N=24, horizon=14,400, thread counts,
measured rates, nor the retry count are doubled. First cap wins; no mid-series cap
increase or inside-run extension. All outputs are create-only. Never overwrite a
study, ledger, attempt, unit, decision or receipt to make a command run again.
