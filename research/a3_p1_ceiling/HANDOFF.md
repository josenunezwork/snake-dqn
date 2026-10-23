# A3 prepared dispatch handoff

Source pin: `19a9d1e76f49b0f8dbd09ac2aa2eefd37c311909`, published on branch
`a100-a3-20261022` and origin/main. Implementation commit `c3853dad` includes the
requested Claude Opus 5.5 co-author trailer. The merge imports only the concurrent
M1 carry reducer. No P1/transport executable changed in A3; the amendment branch
`p1-snapbound-20261022` was not checked out, edited or merged.

Pack root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/dispatch-20261022-02`.
`PREPARED.json` binds the immutable preparation receipt; `CODEX_V_REVIEW.md` is the
independent source review. Separate target canaries remain Claude's next step.
Main templates intentionally refuse supervisor validation until genuine target
canary receipts are bound. No Colab job was dispatched by Codex.

## Verification

All Python commands used `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`.
For pytest and real tiny work, inline environment was `OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu`. There was no xdist or Mac worker pool.

```bash
python -m pytest -q -p no:xdist research/a3_p1_ceiling/test_benchmark.py research/a3_p1_ceiling/test_runtime.py
# 12 passed, 0.47 s
python -m pytest -q -p no:xdist tests/test_colab_cli_canary.py tests/test_colab_cli_schema_failures.py
# 87 passed, 0.83 s
python -m black --check --line-length 100 --workers 1 research/a3_p1_ceiling
python -m isort --check-only --profile black research/a3_p1_ceiling
python -m flake8 research/a3_p1_ceiling
bash -n research/a3_p1_ceiling/colab.sh
git diff --check
```

Real tiny entrypoints, with the same thread environment:

```bash
python -m research.a3_p1_ceiling.benchmark --checkpoint /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/mac-tiny-final --unit-id mac-tiny --mode comparison --variant optimized --device cpu --tiny --repeats 3
python -m research.a3_p1_ceiling.runtime run --control /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/mac-wrapper-01/control.json --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/mac-wrapper-01/ledger.json --checkpoint /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/mac-wrapper-01/output --device cpu --tiny
```

Both completed. All seven independent wrapper cells exported; the verified-ledger
skip entrypoint subsequently skipped all seven without launching the benchmark.
The final optimized comparison executed two Adam epochs per repeat; max gradient
difference `3.0640512704849243e-7`, within frozen fp32 tolerance. Tiny numbers are
operational proof only, never the 128×128 ceiling result. First tiny attempt exposed
the all-held-out world-zero mask; tiny world one fixes that without changing the
production collector. An initial preparation with literal `HEAD` was refused for
not being a full SHA; that unused partial directory is retained separately.

Both actual pinned canaries passed `python -m research.colab_cli.supervisor
--state-dir <lane>/validate-state validate <lane>/canary.json` without allocation.
Both pinned upload/source qualifications passed: normal staging forecast ~69.47 s;
fallback + fetch bound ~253.51 s, below the 300 s shared limit. Main-template
validation refuses on the expected missing target-canary proof.

## Claude dispatch sequence

Use this checkout as the broker, whose Colab subtree hash matches the current
qualified registry `553af3f692515b2bf66173356fe4d099acc96942`. No new source
requalification is required. A100 and L4 each require their own real CUDA canary.
Prepared manifests assume zero other assignments. Reconcile live account endpoints
and reviewed positive rates first; if assignments remain, regenerate into a fresh
pack with `prepare --other-assignments /absolute/reviewed-rates.json` and retain
the new input hashes. Keep priority for P1 Y3/source, A1 and A2. Three account
assignments maximum. Do not run these two lanes simultaneously with the supplied
empty assignment maps; finish/release the first before allocating the second.

```bash
R=/Users/josenunez/Projects/ml/snake-dqn-a3
P=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/dispatch-20261022-02
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python

"$R/tools/ops/dispatch.sh" a3-a100-canary-20261022-02 "$P/a100/canary.json" --broker "$R" --cwd "$R" --state-dir "$P/a100/canary-state" --dispatch-txt "$P/DISPATCH.txt"
```

After the A100 canary terminal receipt reports PASS, immutable exports verify and
backend release is verified, bind and dispatch its main assay:

```bash
cd "$R"
"$PY" -m research.a3_p1_ceiling.prepare bind --template "$P/a100/main-template.json" --receipt "$P/a100/canary-output/receipt.json" --out "$P/a100/main.json"
"$R/tools/ops/dispatch.sh" a3-a100-main-20261022-02 "$P/a100/main.json" --broker "$R" --cwd "$R" --state-dir "$P/a100/main-state" --dispatch-txt "$P/DISPATCH.txt"
```

After A100 finishes and release is verified, dispatch the matched L4 canary:

```bash
"$R/tools/ops/dispatch.sh" a3-l4-canary-20261022-02 "$P/l4/canary.json" --broker "$R" --cwd "$R" --state-dir "$P/l4/canary-state" --dispatch-txt "$P/DISPATCH.txt"
```

After its PASS, verified exports and release:

```bash
"$PY" -m research.a3_p1_ceiling.prepare bind --template "$P/l4/main-template.json" --receipt "$P/l4/canary-output/receipt.json" --out "$P/l4/main.json"
"$R/tools/ops/dispatch.sh" a3-l4-main-20261022-02 "$P/l4/main.json" --broker "$R" --cwd "$R" --state-dir "$P/l4/main-state" --dispatch-txt "$P/DISPATCH.txt"
```

The binding entrypoint re-verifies the real broker canary receipt, same source and
transport, device, inputs, scanner exports and backend release, then validates the
main manifest. No invented canary receipt or direct CLI allocation is needed.

Once all units from both targets are verified, build ledgers without opening
outcomes and take the one Mac-side analysis look:

```bash
"$PY" -m research.a3_p1_ceiling.runtime ledger --control "$P/a100/control.json" --output "$P/a100/completed-ledger.json" "$P"/a100/main-output/exports/units/*.json
"$PY" -m research.a3_p1_ceiling.runtime ledger --control "$P/l4/control.json" --output "$P/l4/completed-ledger.json" "$P"/l4/main-output/exports/units/*.json
"$PY" -m research.a3_p1_ceiling.analyze --pack "$P" --a100-ledger "$P/a100/completed-ledger.json" --l4-ledger "$P/l4/completed-ledger.json" --out "$P/analysis.json"
```

## Numbers and remaining proof

Expected: A100 ~1 allocated hour at forecast 7 u/h (~7 u), L4 ~1 hour at 2 u/h
(~2 u), including setup/canary/export; no deliberate idle time. Ceiling rates are
10 u/h A100 and 2 u/h L4. Effective A100 canary/main: 16/104 min, 3/29 u; L4:
20/100 min, 1/7 u. Combined 4 h/40 u including 8 u reserve, already doubled once.
Mechanical attempts and the dispatch wrapper accounting retry consume that same
series envelope; refer to PROTOCOL.md before preparing any completion attempt.

Kill: **baseline cached actor learning median >1.6384 s/16,384-transition rollout**
pauses WP5 for a learner fix. Equality does not kill. Unexplained gradient changes
reject the optimization (`atol=2e-5`, `rtol=2e-4`). KL-skipped epochs/OOM/missing units
remain INCOMPLETE. CUDA timing, CUDA gradients, production memory and target
canaries are unmeasured. Serving and Mac strict/product gates are unchanged.

No tmux sessions, live jobs, released checkpoints, serving pins, legacy queue or
hash-bound prior documents were modified by A3.
