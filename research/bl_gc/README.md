# Round-25 GC qualification

Authority: `big-lever/round-25/JUDGMENT.md`, SHA-256
`df3c55403da147195dd0508c0f5185e955f20d939ae4f7853e1d2ba2f14d20e7`.
This branch keeps `SNAKE_SERVE_GC_POLICY=default`. Rollback is that selector plus
a process restart. No serving default, cadence, model, observation, or game rule
changes here. Workers never apply the serving lifespan's policy.

**Qualification is blocked at d.3.** The untimed 600-tick identity control passes;
the exact frozen-object-count invariant fails under `freeze_startup`, despite
cleared gameplay weakrefs. The assertion stays unchanged. Full d.1 (three seeds,
5,000 ticks, three fresh processes per seed) has not run. A1 and Sunday have not
run. Consult `GC_CODEX_V_20261023.json` for final source/evidence hashes. Do not
advance the qualification chain on a failed predecessor or release this change.

`identity.py` exports the per-tick SHA-256 chain and final hash. Its `--check`
verifies immutable exports, the default/freeze/default order, source identity,
and equal seed/tick allocation. The permanent test uses 600 ticks and includes a
third default self-consistency process. `lifecycle.py` implements the 20-cycle
reset/rebuild/close/weakref test and the deliberately frozen negative control;
failures retain a JSON receipt and return nonzero. All exports are create-only
JSON + `.sha256` + `.complete` (completion marker last).

The commands below are for Claude, after the blocking d.3 finding has been
resolved through the governing process. Preserve order: d.1, d.3, A1, owner
report/approval, Sunday. All commands use this worktree, the primary venv, CPU
and one native thread. Outputs go outside Git. Never reuse an output name; the
A1 `.started` fence survives a mechanical interruption. Mechanical recovery is
manual, fresh, outcome-blind, at most once per cell after verified stop, under
the ruling's attempt-07 rule. There are no automatic retries.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-gcfix
export SNAKE_SERVE_PATHB=0 SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
GC_OUT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-gc-a1-claude
mkdir -p "$GC_OUT"
set -e
for S in 2026102300 2026102301 2026102302; do
  for ARM in default freeze_startup default-control; do
    P=$ARM; [ "$ARM" != default-control ] || P=default
    ./venv/bin/python -m research.bl_gc.identity --policy "$P" --seed "$S" \
      --ticks 5000 --out "$GC_OUT/identity-$ARM-$S.json"
  done
  ./venv/bin/python -m research.bl_gc.identity --check \
    "$GC_OUT/identity-default-$S.json" "$GC_OUT/identity-freeze_startup-$S.json" \
    "$GC_OUT/identity-default-control-$S.json" --out "$GC_OUT/identity-check-$S.json"
done
./venv/bin/python -m pytest -q tests/test_gc_policy.py tests/test_gc_lifecycle.py \
  tests/test_gc_identity.py tests/test_b8_release_monitoring.py
```

A1 uses the actual M1 `new_watch`, T15 `_timed_tick`, and M1's exact absolute
clock: `epoch + i * 1_000_000_000 // 24`. It excludes only the 200 warm-up ticks,
never rebases or skips a scored tick, and records final-sleep GC. It records per
collection histograms from fixed integer callback counters, per-tick attribution,
block trends, 1 Hz RSS, and tracked-object census in sleep slack. Baseline and
final memory samples are outside the clock; both are retained. The separate
checker requires exactly six immutable cells, identical source/build bindings,
paired seeds, all original bars, and a complete last-five-minute RSS window.
Default-policy failures remain baseline diagnostics; only the three frozen
cells decide the candidate's A1 bars. Missing evidence is INCOMPLETE.

`run_timing_waiter.sh` waits for AC/open lid and no Codex or local Colab supervisor
processes, then atomically arms `snake-dqn-artifacts/ops/mac-timing-armed`. The
Python driver independently checks admission, acquires the registered Mac-wide
exclusive lease, requires nice 0, and monitors power/thermal/process evidence.
It stops on foreign load or its mechanical wall cap and exports INCOMPLETE with
nonzero exit. The wrapper never stops another process. Run serially, outside
Codex, with no concurrent Mac job:

```bash
for P in default freeze_startup; do
  for S in 2026102300 2026102303; do
    research/bl_gc/run_timing_waiter.sh -- caffeinate -dims \
      ./venv/bin/python -m research.bl_gc.probe --policy "$P" --seed "$S" \
      --ticks 2400 --out "$GC_OUT/w0-$P-$S.json"
  done
done
for P in default freeze_startup; do
  research/bl_gc/run_timing_waiter.sh -- caffeinate -dims \
    ./venv/bin/python -m research.bl_gc.probe --policy "$P" --seed 2026102304 \
    --ticks 14400 --out "$GC_OUT/soak-$P.json"
done
# One look, after publication of all six immutable exports; no identity files in glob.
./venv/bin/python -m research.bl_gc.probe --check \
  "$GC_OUT"/w0-*.json "$GC_OUT"/soak-*.json --out "$GC_OUT/A1_VERDICT.json"
```

After the required owner report and OK, Sunday's set runs real `web/serve.py`
through its unchanged Uvicorn import-string/lifespan and uses real loopback
WebSocket clients. `sunday_server.py` applies only T15's qualification-local
Watch count override and fixed seed; it records the actual process/roster
attestation. The matrix covers Watch6/8/12 at ceilings 24/12/12, real
pause/resume/reset/reconnect acknowledgements, 600-second frame-gap soaks under
each policy, and champion checkpoint restarts under both GC selectors (including
`default` restart rollback). It preserves stderr and only stops its owned server
PIDs. Compact frame records retain delivery clocks and game-frame identities.
The separate checker matches GC using the *preceding-drain interval*, since
snapshot/broadcast collections are reported in the next `watch_tick`, and compares
candidate gaps against the paired default frame. Ordinary relative-sleep gaps
alone do not prove a GC stall. Raw collection never takes a verdict look.

```bash
research/bl_gc/run_timing_waiter.sh -- caffeinate -dims \
  ./venv/bin/python -m research.bl_gc.sunday_regression --run \
  --out "$GC_OUT/sunday"
./venv/bin/python -m research.bl_gc.sunday_regression --check \
  "$GC_OUT/sunday/SUNDAY_RAW.json" --out "$GC_OUT/sunday/SUNDAY_CHECK.json"
```
