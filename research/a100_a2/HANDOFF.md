# A2 dispatch handoff — preparation only

Authority: owner opt-in + PLAN_A100 §3 A2, round-20 judgment. The registration was
published as `595bb758` (merged `829950aa`) before A2 opened label payloads.
`SEMANTICS_ERRATUM.md` clarifies historical event-time encoding before any fit.
No Colab allocation, study look, serving change or release was performed.

## Built

480 fixed endpoint fold fits: separate T5/T8 and excluded-A5 tables; C0 frozen
linear, S random, S B8-init and M random (6,680,134 parameters with D300 head);
D300 and censored 30/60/100/200/300 hazards; two seeds; nested 25/50/100% parent
world doses; five original folds. Fixed 20 epochs, AdamW, no checkpoint selection.
A5 raw rasters are reconstructed through verified replay on Colab, with portable
frozen opponent inputs. No new labels or sealed/gate outputs are consumed.

`protocol.py` builds outcome-blind verified ledgers. Complete fold predictions
are atomic create-only exports with completion markers last. `aggregate.py` is
the only scoring entrypoint and refuses missing units, duplicates, foreign
allocations, shrunken row sets, cross-fold ancestry, or inconsistent paired labels.
Remote attempt receipts contain only operational counts and input lineage.

## Concrete pack

Prepared inputs:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/inputs-20261022-01`

Generated manifests:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-01`

The input archive is 91,491,883 bytes, SHA256
`6572e096f1e6c08b1b5dab1cc9d829b6931dfe3287491ae3d91c81e32360d68c`.
B8 SHA is `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
The published code pin, complete input inventory and allocation digest live in
`pack.json`; source/test hashes live in the independent Codex-V review receipt.

## Claude's commands, in order

The generated initial manifests declare no other assignments. If other runtimes
are active, prepare a fresh sibling pack using `make_pack pack --other-assignments`
and the reviewed endpoint-to-positive-rate JSON map; do not edit immutable packs.
The supervisor refuses unknown assignments or an unpriced rate. Priority remains
P1 Y3 > A1 > A2 > A3; at most three account assignments.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-01
python3 "$REPO/tools/ops/qualification.py" check --repo "$REPO" \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$REPO/tools/ops/dispatch.sh" a100-a2-canary-20261022-01 "$PACK/canary.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-canary-20261022-01 \
  --dispatch-txt "$PACK/DISPATCH.txt"
# Wait for genuine PASS, scanner-verified exports and confirmed runtime release.
cd "$REPO"
"$PY" -m research.a100_a2.make_pack admit --template "$PACK/main-template.json" \
  --canary-receipt "$PACK/canary-output/receipt.json" --out "$PACK/main.json"
"$REPO/tools/ops/dispatch.sh" a100-a2-main-20261022-01 "$PACK/main.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-main-20261022-01 \
  --dispatch-txt "$PACK/DISPATCH.txt"
```

The generic qualified CLI tree stays
`553af3f692515b2bf66173356fe4d099acc96942`. No full transport requalification is
required. The real same-pin A100 family canary remains mandatory: <=18min/3u,
full production upload inventory, all eight tiny encoder/head combinations, real
CUDA action selection, A100-40 hardware evidence, verified final exports and stop.
A100-80 is refused by the package before label access.

After all terminal fits have been exported and stop is verified, build the ledger
without opening predictions, then take the registered look exactly once:

```bash
ALLOCATION=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["allocation"])' "$PACK/pack.json")
"$PY" -m research.a100_a2.protocol --exports "$PACK/main-output/exports" \
  --allocation "$ALLOCATION" --out "$PACK/verified-union.json"
"$PY" -m research.a100_a2.aggregate --ledger "$PACK/verified-union.json" \
  --allocation "$ALLOCATION" --out "$PACK/decision.json"
```

A completion after mechanical failure is never automatic. Require verified
mechanical failure, verified stop and missing units; preserve every attempt;
allow at most two completion attempts, within the ORIGINAL remaining 4.8h/48u
series envelope. Prepare fresh immutable manifests/outputs/receipts, bind the
Mac-verified ledger as an input and update its control input hash. Changing a
ledger changes upload identity and therefore requires a fresh same-pin family
canary and fresh admission; the initial proof cannot cover different ledger bytes.
Include all attempt roots in the final hash-only ledger. Never retry a scientific
kill, reread an incomplete look, extend N, or recompute verified units.

## Pricing and decisions

Forecast: main 2 A100-40 hours (~10.8u at historical 5.4u/h), plus a short canary;
this is a planning estimate, not measured A100 throughput. Price admission at
10u/h until a reviewed live rate is available. Effective overall cap 4.8h/48u,
2x applied once as explicitly ruled. Canary 18min/3u; main 258min/45u;
reserve 8u (base4u), inside the series cap. First applicable cap wins. Actual
allocated elapsed time includes setup/export/failed attempts.

Primary: M-random minus S-random D300 AUC on full-dose T5, seed probabilities
averaged before scoring. Close scale/volume on this label type iff delta <0.02
AND paired world-bootstrap lower95% bound <=0 AND M 50->100% slope <0.01.
Closure permits WP4 only on WP2 counterfactual labels. Hazard improvement <10%
relative D300 Brier retains single-horizon in WP3; equality retains hazards.
Other arms/A5 are disclosed sensitivities. Historical exploratory evidence is
never a served-gain or gate claim.

## Verification

- 19 focused A2 tests passed in 2.19s, single process, `-p no:xdist`.
- Four focused existing transport tests passed, 149 deselected (21.91s).
- Real `bash research/a100_a2/colab.sh`, actual pinned B8 + tiny excluded synthetic
  table, CPU one thread: 8/8 fit units, 0.7205s job /1.36s end-to-end.
- Tests exercise source-bound label endpoints, illegal sentinels, censored masks,
  fold grouping/nested doses, parameter count, fit prediction schema, full 480-unit
  synthetic reduction, no-N-shrink guard, kill boundaries, atomic publication,
  foreign ledgers, actual entrypoint skip behavior and main archive plumbing.
- Black/isort/flake8 and shell syntax checks cover changed package/test files.

Live A100 throughput, full historical replay on Colab, transfer timing, real
family canary and final scoring remain operational gates for Claude. Initial
main-template is intentionally inadmissible until genuine canary evidence is bound.
No daemons, tmux sessions, released checkpoints, serving pins or existing
hash-bound evidence were modified.
