# PLAN_v3 M0 — Sol-S handoff, 2026-10-22

Scope: M0 only. Part A is published; Part B is prepared for Claude and has not
been run on any production rescue. No Colab. All new code belongs to Sol-S's
`bl_served_search` slice. Existing replay, simulator, lease, and native modules
are read-only dependencies. No frozen/registered package was edited.

## Published Part A

`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/part-a.json`

SHA-256: `f3e366453f07c018bda28a70cfadb3b4a4458f28fd2b3ea57f787b1967729dd4`.
Both `.sha256` and `.complete` verify. Produced from `871bd6c6` before any Part B
file was built, under `nice -n 10`, 0 units, approximately 0.4 s including process
startup. The payload retains the precise extraction elapsed time.

All eight completed shards and the unchanged bank are verified: 160 worlds,
28 productive rescues (13 frozen, 5 mixed, 10 scripted). Of the 18 frozen/mixed
rescues, 15 have first-ranked surviving alternatives, one has second-ranked,
and two are all-die at f. All precursor counts are zero. The two all-die rows
do not exceed the registered one-third cost threshold. This is development
bookkeeping; a, a−f, F(L), and the D1 decision remain pending Part B/M1.

## Part B behavior and limits

`lead_census_b.py` reconstructs the exact T3 bare world from the saved plan and
seed. It uses stock mode (kernel off), no opponent veto, exact-stop, full H300,
and f+5k probes through the fatal decision. Searches run in independent prepared
clones; proposed rescue/delay actions are never applied. It verifies each bare
mass/alive frame, exact terminal death/cause/length, f's saved dense batch, and
each bare branch's terminal coordinate. A first all-die batch does not end the
scan. a is the last probed decision with a surviving alternative; no such batch
is represented by null and counts as unreachable.

One serial spawned worker, one numerical thread; `nice 10`, CPU, AC/open lid,
thermal and idle receipt; canonical `research.mac_lease.held` holds the same
primary-checkout `exp/mac-lease.json` process-live fence as other timing lanes.
The run acquires/releases its own lease and never steals a live holder.
The **entire series** shares the already-doubled PLAN_v3 cap of 3,600 seconds
(base forecast 1,800); 60 seconds inside that cap are reserved for host probes
and owned-child shutdown. No worker/process outside this run is stopped.

Units are deterministic `units/world-00.json` through `world-27.json` in the
lexical Part A allocation. Each exports atomically, SHA sidecar next, completion
marker last. The initial empty ledger binds source/Part A/allocation. A manual
completion ledger includes only verified path/size/hash units; interrupted
unmarked exports stay preserved and are not skipped. Marked-complete corruption
is refused. Foreign namespaces, source changes, allocation IDs, duplicates,
expired budgets and second retries are refused. No automatic retry.

The separate aggregator requires all 28 worlds and recomputes cadence, bare
controls, full-H300 survivor witnesses, all-die ranking and LAD from the batches.
It checks stock counters, worker thread limits and original opponent identity.
It publishes the six registered F(L) points plus every possible formula-derived
lead (72, multiples of 5 through 430, 432). Equality is reachable, with the
registered +2-frame margin. It prints `COMPLETE_D1_PENDING_M1`, never GO/CLOSED
without M1's frozen doom-W4 p99. Fewer than 12 reconstructed frozen/mixed worlds
is INCOMPLETE; this harness additionally refuses every partial-bank look.

## Exact commands for Claude

These commands use the handed-off worktree. No replay command below was run by
Sol-S. The prepared empty ledger is `inputs/verified-ledger-01.json`, SHA-256
`329e2c1edc9bd442c00ad40050c1e37a0855974ea95cbe6645383bab8b53cfe8`.
Do not regenerate/overwrite it.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-m0
export PYTHONDONTWRITEBYTECODE=1 SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash /Users/josenunez/Projects/ml/snake-dqn-m0/research/bl_served_search/run_lead_census_b.sh --check

# Under Claude's scheduled exclusive Mac window; refuses foreign/live lease.
# Prefer the normal operator tmux session and a fresh noclobber log.
bash /Users/josenunez/Projects/ml/snake-dqn-m0/research/bl_served_search/run_lead_census_b.sh --run \
  --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/inputs/verified-ledger-01.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/part-b-attempt-01

# Sole scientific look, only after COMPLETE_UNAGGREGATED and verified stop.
nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.lead_census_b aggregate \
  --attempt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/part-b-attempt-01 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/part-b.json
```

If the first attempt mechanically fails, stops, and has missing units **with
budget remaining**, prepare exactly one completion attempt manually:

```bash
nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.lead_census_b ledger \
  --previous /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/part-b-attempt-01 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/inputs/verified-ledger-02.json
bash /Users/josenunez/Projects/ml/snake-dqn-m0/research/bl_served_search/run_lead_census_b.sh --run \
  --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/inputs/verified-ledger-02.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-c15/lead-census/part-b-attempt-02
# Aggregate attempt-02 with the same --out part-b.json, once complete.
```

Scientific/identity mismatches are INVALID and do not authorize retry. Cap
exhaustion does not increase the envelope. No Part B timing/throughput claim has
been measured. A lease/guard failure before start produces no attempt directory;
it has not executed a world. This preparation does not qualify C15 or change B8.
