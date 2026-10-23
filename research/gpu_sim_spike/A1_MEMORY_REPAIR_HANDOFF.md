# A1 mechanical memory repair — 2026-10-22

Prepared only. Claude dispatches; no GPU, account allocation, live worker,
released checkpoint, serving pin, or original receipt was changed by this repair.
The old A0_A1_HANDOFF.md and PLAN_A100.md remain immutable historical authority.

See MEMORY_DIAGNOSIS.md and the CPU-only `memory_sizing` entrypoint for the
allocation arithmetic. The 17.14 GiB request identifies vector61's E=1024
per-action body-coordinate subtraction. The original exception has no stack;
the exact location is inferred from the unique matching shape and execution
order. The earlier E=6/E=256 cells were held in a list comprehension and never
published when E=1024 failed. This is not evidence that those cells failed or
that E=256 leaked into E=1024. The new runner reports its active unit, traceback,
allocated/reserved/free bytes at failure, and each repeat's before/peak/after
memory. A `finally` block drops per-repeat simulator/policy/templates/collection,
collects garbage, then empties the CUDA allocator cache.

Main science is unchanged: all nine E=6/256/1024 × three mixes, three warmed
repeats × eight complete ticks, same six long bodies, same real policies, fp32,
reset/collection work, separate trace pass, unrepaired E6 control, independent
CPU tape classification, and every original kill threshold. Only independent
row-local feature scratch is chunked; worlds and network forwards retain their
full batch shape. The free-space flood runs once per feature invocation.

Each cell's measurement and trace form one immutable completed unit; control
and parity are separate units (11 total). Cell-local trace now immediately
follows its measurement so a later failure cannot erase earlier complete cells.
The new canary tests E=6 and E=1024 for all three mixes (one tick plus trace,
no scientific conclusion; eight units including control/parity). Canary and
main have distinct namespaces and never pool measurements. Namespace includes
the full source pin and hardware family. A100-80/H100-80 have separate packs;
they are optional hardware probes and cannot replace or pool with A100-40 data.

Payloads are create-only, atomically published, with SHA sidecars and completion
markers last. A failed attempt's complete units remain supervisor-discoverable.
The Mac ledger builder reads paths/sizes/hashes/markers only, copies verified
bytes into a fresh portable recovery bundle, ignores unfinished units, and
refuses complete corrupt units. The runtime verifies namespace, allocation,
IDs, size/hash and wrappers, skips exactly those units, and re-exports their
identical bytes into the fresh output. Remote summaries are sealed diagnostics;
`a1_analyze` refuses a partial allocation before loading outcomes and applies
the existing `classify` function once, on the Mac, after all 11 units exist.

## Preparation and dispatch

After publishing the reviewed source commit, from this worktree:

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-a0
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-a1-memfix-a10040
cd "$REPO"
PIN=$(git rev-parse HEAD)
"$PY" -m research.gpu_sim_spike.make_a1_pack --commit "$PIN" \
  --artifact-dir "$ART" --attempt 2 --remaining-minutes 84 --remaining-units 14 \
  --validate --emit-80gb-variants
```

This mechanical completion is attempt 2. Its new main cap is 84 minutes/14u
(base 42 minutes/7u, effective 2× once), with the unchanged 4u reserve and
18-minute/3u canary. Old canary and failed main receipts have verified stop;
they consumed 449.478090708 + 385.103944375 = 834.582035083 seconds.
At the conservative 10u/h ceiling that is 2.318283431u. Including the old
canary, the failed main, the entire new canary and main caps gives
115.909700585 minutes and 19.318283431u, below the original 120min/20u.
This accounting uses elapsed wall and priced ceilings, never observed results.
A third attempt must fit what remains after these receipts; no replenishment.

The generated `$ART/DISPATCH.txt` contains the exact new dispatch names,
commands, same-pin canary receipt binding, and Mac aggregation command.
Populate each manifest's `budget.other_assignments` from fresh reviewed positive
endpoint/rate inventory before dispatch. The 10u/h ceiling is a conservative
admission ceiling; it is not a current provider quote. An observed higher rate
is a refusal, including for optional 80GB packs. Main base/effective envelopes
remain 1h/10u → 2h/20u, reserve 2u → 4u; canary remains 9min/1.5u → 18min/3u,
reserve .3u → .6u. First cap wins, including setup and export.

Completion is manual, outcome-blind, after verified mechanical failure,
verified stop and missing units. At most two completion attempts; retain the
remaining original series wall/unit envelope (including failed attempts),
never increase it or automatically retry. This declaration does not authorize
restarting a completed scientific failure. Every completion needs a fresh pack,
output directory, dispatch name/state and terminal receipt. Build the ledger
from a failed attempt's verified exports without opening cell outcomes:

```sh
"$PY" -m research.gpu_sim_spike.a1_units --commit "$PIN" --hardware A100-40GB \
  --exports "$ART/main-output/exports" \
  --output "$ART/recovery-01/verified-ledger.json"
"$PY" -m research.gpu_sim_spike.make_a1_pack --commit "$PIN" \
  --hardware A100-40GB --artifact-dir "$ART/completion-01" --attempt 3 \
  --remaining-minutes REMAINING_MINUTES --remaining-units REMAINING_UNITS \
  --ledger "$ART/recovery-01/verified-ledger.json" --validate
# Run this new pack's canary, then bind its own receipt using its DISPATCH.txt.
```

Use unique completion dispatch/state names; the ops wrapper refuses used names.
Both canary and main upload byte-identical inventories containing both the
full and smoke ledgers at different destinations. A changed recovery ledger
or added resume payload changes the input inventory and therefore needs a
new canary; an earlier receipt cannot cover those new bytes.
Do not change a published manifest or old receipt to obtain a retry. Fresh
attempt caps must fit the remaining series envelope before dispatch. This
repair's offline preparation does not allocate, dispatch, or prove a GPU peak;
the new canary must record actual CUDA memory and native execution first.
