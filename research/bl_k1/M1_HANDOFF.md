# PLAN_v3 M1 wiring — Sol-K → Codex-V → Claude

Status: implementation and untimed checks only. Claude runs the timed cells.
Authority: `snake-dqn-artifacts/big-lever/PLAN_v3.md` §1 M1 / §3 and
`big-lever/round-15/JUDGMENT.md`, both read in full. Shared writes are confined
to `bl_mac_latency/l1.py` and `bl_k1/*`. No Colab or serving change.

`l1.DeviceOperator` now constructs `create_stop_operator()`. Worker payloads
retain full semantic/stage/native-row evidence and actual numerical thread
settings. The old diagnostic compares chosen actions rather than comparing the
all-branch and exact-stop payloads in their entirety.

The M1 entrypoint is `research.bl_k1.m1`, through `run_m1.sh`. It is separate
from the older L1′ diagnostic CLI: M1 admits construction only, never production
L1 qualification. Frozen cells, in order: CPU W1/K1, W2/K2, W4/K4, W4/K8.
Each cell starts a fresh spawned process pool under the same absolute Watch
clock, reports a separate cold K-submission wave, then 30 interleaved rounds
through all nine original Y1 fixtures (270 warm samples). First encounter of a
fixture in a worker can include root reconstruction; no hidden warm-up occurs.
No fixture is selected using its outcome. Queue depth K is actually filled,
including eight concurrent outstanding submissions in W4/K8. Worker PID counts
and pending saturation are checked.

The co-resident game is the real product `GameSession` mixed Watch6 constructor
from T15, B8 s3 against five FRP3 s12 + v8, all continuously respawning. Optional
Path B is explicitly disabled and action transforms refused. Every real
`step()`/`snapshot()` runs on T(n)=t0+floor(n·1e9/24); update time and full-loop
deadline overruns are separate. No tick is omitted after a late iteration.
Cold pool submission, IPC decoding, and queue refill are included in full-loop
overrun accounting. Host AC/lid/thermal/process snapshots run on a separate
telemetry thread every two seconds. Failed guard stops admission and drains
owned workers cooperatively. No daemon, foreign process, or tmux session is killed.

One declared 30-second, single-process, 100%-of-one-core contention block starts
at the warm phase of W4/K4; its actual process CPU seconds and monotonic interval
are retained and at least 50% occupancy is required. It is joined cooperatively.
Ticks continue until every result is visible and this block has finished.

The historical Y1 fixture file and paired Mac reference are accepted only at
SHA-256 `bbe6975642512091ba279d9ff7c870000d68faa18e30063807f47af1352e8574`
and `b97f73d2384a95455beb3c0d243caba513f5f307edc85d0a8da4b81f5e1ac791`.
Later Mac-L2 work added source-inventory entries and the `full_depth` seam;
M1 does not loosen or edit those shared validators. It validates all 540 saved
reference batches and uses their chosen actions as its immutable oracle. No
historical latency or PASS transfers. Every new batch must independently match
the all-branch chosen action and attest exact-stop frames/branches, one numerical
thread, `kernel_rows > 0`, and zero stock/fallback/noncandidate branch rows on
frozen/mixed. Scripted rows have an explicit zero-native-row exemption.

Frozen bars: zero overruns in every cell; update nearest-rank p95 ≤ 0.011688 s;
W4 doom fixture `frozen-800` S p99 ≤ 5 s. Dense caps .830/1.250/1.458 s are
report-only. S is admission to first decoded coordinator observation, reported
by cell and fixture with all raw samples, inclusive nearest-rank p50/p95/p99.
Thirty warm samples means p99 is their maximum. Cold samples are separate.
The lead formula consumer uses `cpu-w4-k4`'s frozen-800 warm p99; the saturated
W4/K8 cell is also subject to the W4 kill rule. Missing/malformed counters or
action mismatch ⇒ INVALID; mechanical cap/incomplete counts ⇒ INCOMPLETE;
complete bar failure ⇒ BLOCKED. PASS admits construction only.

Budget: base forecast 7200 s; already-doubled effective cumulative cap 14400 s
(PLAN_v3 says cap 4h). Setup, game/pool startup, failed work and draining are
charged. No second doubling. One outcome-blind completion attempt, within the
same cumulative cap, only after a verified mechanical stop; a judged scientific
remedy is separate and is not dispatched here. No automatic retry.

Resume unit = a complete co-resident cell, with namespace binding the frozen
allocation and full source/input hashes. Create-only payload, SHA sidecar,
completion marker last. A first run requires an explicitly empty verified ledger.
An interrupted cell is preserved but rerun whole; samples from different clock
epochs are never spliced. A recovered attempt must supply the previous terminal
INCOMPLETE receipt with verified clean stop, and a ledger skipping **exactly**
all its completed cells. Ledger creation reads path/size/hash only; the reducer
then opens each unit and recomputes allocation, counters, actions, timestamps,
clock/overruns, distributions and bars. Foreign/duplicate units are refused.
A hard process death without verified terminal stop requires operator/judge
reconciliation; it cannot authorize a blind retry.

## Claude commands (no timed cells were run by Sol-K)

Run from the published primary checkout once its clean `main` has fast-forwarded
to the handoff commit. Keep it unchanged while timing. Checkpoint files must be
available in its existing `saved_snakes/`; do not run in an unprovisioned worktree.
The launcher pins CPU/math threads=1, nice=10, caffeinate, disables Path B and
takes the shared `research.mac_lease` lease at primary `exp/mac-lease.json`.
First ensure Claude's other lease holders are idle; do not kill them.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn
research/bl_k1/run_m1.sh --check \
  --fixtures /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json \
  --reference /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/y1-native-mac/ruler-receipt.json
research/bl_k1/run_m1.sh \
  --fixtures /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json \
  --reference /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/y1-native-mac/ruler-receipt.json \
  --make-ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-m1-ledger-01.json
research/bl_k1/run_m1.sh --run \
  --fixtures /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json \
  --reference /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/y1-native-mac/ruler-receipt.json \
  --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-m1-ledger-01.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-m1-attempt-01
```

All outputs are create-only; use a fresh numbered path if already present. The
first two commands do not start a game, worker, lease, or timed cell. For an
admitted mechanical recovery, create a new ledger with one `--unit /absolute/
prior/cpu-wN-kN.json` per **completed** cell listed in receipt.json, then run
with a fresh output and `--previous-attempt /absolute/prior/receipt.json` plus
that ledger. Omit incomplete cells from the ledger. Recovery validation refuses
a missing, extra, changed, or foreign completed unit and deducts prior debit.

Tests in `bl_k1/test_m1.py` mock the game, processes, host monitor and clock;
they produce no timing evidence. The independent Codex-V receipt binds the
reviewed source hashes and exact targeted verification command. Real M1
measurements, host behavior and all production/release gates remain unmeasured.
