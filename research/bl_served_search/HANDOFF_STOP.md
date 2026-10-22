# Sol-S Friday exact-stop handoff — prospective, no measurements

Rules were committed and pushed to main FIRST at
6cc423703a5d8c2a9d0be4a9f0f3722be5d4c473.

Authority: AMENDMENT_STOP.md and round-13 JUDGMENT sections 1.6, 6, 9.
Implementation adds no learning, dispatch, timing or serving result.

## What is ready

`ExactT3Operator(native_mode="kernel", search_mode="exact_stop")` evaluates
bare first, then alternatives by Q descending / index ascending. First H300
survivor stops the batch. All-die evaluates all and uses unchanged `_choose`.
Every skipped branch is exactly `{"status":"NOT_EVALUATED"}`. Actual branch
counts, order, frames, native/fallback row counters, observed CPU thread counts,
stage times and per-worker wall times are exported. A survivor must execute
all 300 frames. Full vs stop chosen action must agree on every paired batch.

`native_ruler` alternates homogeneous worker slabs between full and stop,
retains immutable per-cell batch records, and measures each cell's aggregate
rate from its own wall intervals. No hook-speed multiplication or rate clipping.
Mac uses one worker; L4 uses twelve. Thirty repeats per each of nine original
fixtures per cell; cold and warm-29 median/nearest-rank p99 are reported.
The one-repeat real canary is PRECHECK evidence only, never a full forecast.
Each full run requires the matching source/fixture/platform canary, K1_PASS,
explicit CPU/thread limits and (Mac) a valid matching lease. The Mac initial
host must be idle; periodic power/lid/thermal/load/process evidence is retained.
Lease/cap failure refuses before submission and stops only this pool's children.

The original frozen fixture bytes stay unchanged. The native-only compatibility
allowlist accepts only the exact historical ceiling/operator/chunks hashes,
K-1's previously changed bank_engine branch-routing hash, and omission of the
new exact_stop.py dependency. Every other source byte must match. Current
native routing, kernel, guard, scripts and amendment are source-bound separately.
The legacy stock ruler/pricing/readiness path remains separate; these receipts
must go through placement_stop, not legacy pricing_v2.

## Claude commands — Friday October 23, 12:00–20:00 PT, D2 PASS only

Run from this published worktree; no need to update or reprepare K-1's source.
The launcher checks AC/lid/thermal/idle through the actual ruler, uses nice and
caffeinate, acquires the canonical lease, runs a <=600-second real canary, runs
the full paired ruler within the remaining two-hour envelope, and releases only
its matching lease on exit. Existing lease/output files refuse; never auto-retry.

```sh
cd /Users/josenunez/Projects/ml/snake-dqn-wp1
research/bl_served_search/run_stop_mac.sh --check
research/bl_served_search/run_stop_mac.sh --run
```

Inputs are the original
`big-lever/exp/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json`, the pinned
B8 file under primary saved_snakes, and future `big-lever/exp/bl-k1/timing.json`.
The canonical lease is `big-lever/exp/mac-lease.json` (owner
`claude-sol-s-friday`). Output: `exp/bl-l2/y1-native-mac/ruler-receipt.json`;
canary: `exp/bl-l2/y1-native-mac-canary-01/ruler-receipt.json`.

**K-1 source caution:** its prepared manifest binds the old source closure.
Run K-1 timing from the exact source checkout admitted by its current prepared
manifest and owner timing amendment, under its Mac lease. Running K-1 from amended main will correctly
refuse source drift. Do not silently reprepare or rerun its identity audit.
This task does not modify K-1 code, manifest, identity receipt or window.
Concurrent upstream commits 82737c0c/7e307a2f changed its timing to a two-hour
window from launch; they were merged intact at daba3c86. Honor that updated
authority instead of the historical fixed Thursday window.

## L4 payload — Claude via qualified supervisor only

No Colab command was called, no paid manifest was prepared and no target-platform
canary exists yet. `run_native_ruler.sh` is the real CPU-search entrypoint for
both canary and full native rulers. It does not call any provider. The original
run_ruler.sh and old prepared manifests remain unchanged and do not opt in.

Prepare a NEW source-pinned family using script
`research/bl_served_search/run_native_ruler.sh` on L4, 12 vCPUs, Python3.12/cu126.
The supervisor uploads these flat inputs including `.sha256` and `.complete`
sidecars for every JSON export (do not use the legacy fixture-bundle unpacker):

- `ruler-fixtures.json`: the original nine-fixture aggregate; roots are embedded.
- `k1-timing.json`: verified D2 PASS receipt.
- `public-inputs.json`: the existing hash-bound public checkpoint catalog.
- Full attempt additionally `native-canary.json`: same-source L4 native canary.

For canary set string env `BL_NATIVE_CANARY=1`, `BL_NATIVE_MAX_SECONDS=600`.
For full set `BL_NATIVE_CANARY=0`, `BL_NATIVE_MAX_SECONDS` to the registered
remaining wall allowance (<=5760 seconds including the canary, minus setup and
exports). The wrapper exports CPU and all five one-thread limits. Both use
fresh outputs; export is `native-ruler/ruler-receipt.json` and its sidecars plus
bounded batch/host evidence. Preserve the original fixture export sidecars.

The shared supervisor's source/family-bound real-transport/CUDA/export/release
canary and requalification must also PASS before science dispatch. A local
canary string is not that gate. Register actual staging bytes AND file count,
<=150 seconds of the 300-second cap, <=4 L4 units TOTAL including setup, both
attempts, export, failures and 20% reserve; do not reuse the old 48-minute
estimate for a doubled workload. First cap wins, no allocation/job auto-retry.
This is NOT_READY(native L4 canary/ruler), not a dispatch-ready manifest.

## Saturday placement forecast — no scientific look

After the Mac receipt, make a Mac-only prospective judge input:

```sh
cd /Users/josenunez/Projects/ml/snake-dqn-wp1
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.placement_stop \
 --mac-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/y1-native-mac/ruler-receipt.json \
 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/placement-stop-mac.json
```

After Claude verifies L4 exports and terminal runtime release, copy the verified
payload+sidecars to `exp/bl-l2/y1-native-l4/ruler-receipt.json` and run:

```sh
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.placement_stop \
 --mac-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/y1-native-mac/ruler-receipt.json \
 --l4-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/y1-native-l4/ruler-receipt.json \
 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/placement-stop-paired.json
```

Both exports retain their input digests. Full and stop forecasts are separate;
only decision-identical stop eligibility drives the recommendation. Mac/L4
semantic mismatch in either cell refuses forecasting. L4 uses its measured
aggregate R>=2.18 plus eight 8/8/4-world chunks on 12 lanes, 2x estimate<=720min,
liability+40-u reserve<=200u. Mac uses two lanes across 64/64/32 worlds, 578
batches/world, conservative 30min setup per each of eight chunks, 20% reserve,
<=60wall-h; no paced displacement is a judge-confirmed condition. Mac timing
never prices L4. Status is recommendation only, never dispatch authority.

Current forecast: **NOT_READY(native measurements)**. ~4x common-root branch
reduction is a hypothesis, not measured whole-workload speed. No new numeric
rate, date promise, bank reduction or funding claim is made. P1/K-1/X5/KEEP_B8
and V0/V1 scopes and dates are unchanged; B8 serving is unchanged.

## Verification and exclusions

Targeted synthetic suite (exact_stop, native_ruler, mac_lease, placement_stop,
existing ceiling and ruler): 48 passed. The exhaustive identity test covers
15,552 synthetic outcome/Q/bare cases against full unchanged `_choose`.
Independent reviewer examined control flow, actual execution boundaries,
source compatibility, kill rules and placement math; its targeted slice passed.
Black100/isort/flake8 on all changed Python; bash syntax on both new scripts;
Mac launcher --check and all three CLI --help paths passed. Original nine
fixture metadata validate without reconstruction/gameplay/timing. The target
platform real canaries and scheduled measurements remain Claude's work.

No Colab/provider calls, no Mac timing, no external lease acquisition, no
scientific selection bank/gameplay, no tmux changes, no released checkpoints,
serving pins, historical hash-bound docs/receipts or protected research packages
were edited. The prohibited t34-round0-study-01 artifact directory was not opened.

Publication note: implementation commit bfdac37a, integration merge daba3c86.
The automatic merge message lacked the requested co-author trailer; published
history was preserved. All authored rule/implementation/handoff commits include
the exact requested trailer.
