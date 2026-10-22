# Density measurement — round 14 (b)

The active contract is the immutable METRICS.md plus amendments 01/02 and prospective
[METRICS_AMENDMENT_03.md](METRICS_AMENDMENT_03.md). Read amendment 03 for the full
allocation, minimum-map rule, witness endpoint, decision precedence and caps.

A has 12 natural episodes per mix. B70m/B70p have six per mix on 121x69 maps with
five/seven opponents. C-P1 has six per mix using P1's exact long-body construction:
hero 900/1200/1500, opponents 600/400/300/150/150. B50 is dropped. Total 90 episodes.
Primary endpoints are witnessed-avoidable doom roots and distinct doom-bearing
episodes per total CPU-second. No witness means unproven avoidability. AUC >=.80
rejects; lower AUC never certifies realism. Every source must meet the map floor.

`runner.py` records all lengths/geometry, causes, full-state carried-world branches,
all legal alternatives on first four confirmed doom roots, stage CPU and immutable
exports. `analysis.py` charges every attempted unit plus overhead before decisions;
`statistics.py` retains the frozen 35-feature classifier and separate geometry KS.
`forecast.py` prices the 1,638,000-decision worst case; tiny Mac timing is a host
sensitivity, never a measured L4 rate.

`manifests.py --phase canary` prepares the separate 12-episode tiny L4 canary
(<=20min, <=1u, .2u reserve). `--phase production` prepares the 90-episode production
manifest (<=120min priced wall, below the registered 150min outer cap; <=5u including
1u reserve; six single-thread CPU workers). `budget.other_assignments` is `{}`.
Every preparation/validation call is offline. `upload_readiness.py` uses the real
qualified supervisor sparse-source bytes and Z-9 archive sizing without Colab calls.
Manifest payloads and real run exports have `.sha256` and `.complete` sidecars.

Production is runtime-blocked without `canary-proof.json` and its real sidecars.
After the L4 canary completes and stops, `canary.py --recovered-output ...` verifies
the supervisor receipt, terminal generation, exact inventory, twelve episode exports,
summary hashes and stage counters. It creates a proof; rebuild a fresh production
manifest with `--canary-proof .../canary-proof.json` and requalify those upload bytes.
A Mac probe cannot create an accepted L4 proof. Nothing automatically dispatches.

Use `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`. Local work is `nice -n 10`,
CPU with one numerical/Numba thread and at most two workers; production stays on L4.
The current manifests, validation/transport receipts, preflight, forecast and operator
DISPATCH.txt live under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/round14/`.
The independent diff review is recorded in REVIEW_AMENDMENT_03.md. No scientific
result, +40 claim, P1 change, release or serving authorization follows from this work.
