# F3 operator handoff — 2026-10-23

Worktree `/Users/josenunez/Projects/ml/snake-dqn-f3`, branch
`data-f3-20261023`. Final executable source/pack pin:
`560e6526d884de605af45f4c3696b5a0a127b72d`, published on branch and origin/main.
The subsequent evidence-only commit adds this handoff and sibling receipts.

**F3 built; F4 unratified. No factory labels read, Colab dispatch, spending,
timing assay, serving change or release.** The scientific analyzer, fixed Watch6
product-session N, full P7 trajectory-probe tripwires and later collection
envelopes await Astra/independent judgment of F4. The package can only emit
synthetic proof or `NOT_A_DECISION` gameplay observations in its current state.

## Delivered

- `model.py`, `fit.py`: zero-init residual, frozen cached 256-D B8 features,
  exact-only cost-sensitive preferences, ordinary growth/KL retention,
  fixed-update recipe, final B8-compatible fused checkpoint/full fit state.
- `data.py`, `heldout.py`: append-only nested life support and whole-parent
  five-fold separation, fixed 64-life/16-Watch6 holdout, actual fused rowwise
  predictions, complete ordinary archive requirement, conservative unknown
  overrides, regret/retention/harm/ESS/reliability metrics.
- `gameplay.py`: existing exact terminal-hero evaluator, clone/digest identity,
  collision-bound fresh paired 128/mix H5000 hook; production remains guarded.
- `protocol.py`, `job.py`, `colab.sh`, `make_pack.py`: immutable complete-fit
  units, hash-only resume ledgers, staged new-dose jobs, qualified CLI pin and
  clean-venv source import gate; blocked future L4 canary template.
- `F4_DRAFT.md`: gameplay-judged 256/1,024/4,096 dose proposal for Astra.
- `FREEZE_V2.json`, `CODEX_V_REVIEW.json`, `CODEX_V_PACK_ADDENDUM.json`,
  `LOCAL_VERIFICATION.json`, `PACK_VERIFICATION.json`: pre-data freeze and proof.
  Earlier freeze and refused pack bytes are preserved.

## Verification commands and outcomes

Run from `/Users/josenunez/Projects/ml/snake-dqn-f3`:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/bl_residual tests/test_bl_residual_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --profile black --check-only research/bl_residual tests/test_bl_residual_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_residual tests/test_bl_residual_*.py
bash -n research/bl_residual/colab.sh
git diff --check
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist tests/test_bl_residual_*.py tests/test_ops_qualification.py tests/test_colab_cli_schema_failures.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist tests/test_bl_residual_pipeline.py tests/test_colab_cli_canary.py
```

All style/syntax checks pass. Initial complete selection **46 passed, 4.16s**;
after the packaging correction, pipeline+qualified-canary selection **83
passed, 1.34s**. These selections overlap. Independent Codex-V: **26 passed**
for the initial harness; addendum **1 passed / 7 deselected**, accepting the
corrected launcher/pack/parser test. No full-suite or timing qualification claim.

Actual Mac fit entrypoint (already run; paths occupied):

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.make_pack --local-smoke-inputs-only --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/mac-smoke-20261023-01/inputs --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.job --input-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/mac-smoke-20261023-01/inputs --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/mac-smoke-20261023-01/output --mode smoke --device cpu
```

**3/3 synthetic fit units, 0.398560917s**, 3 updates/unit rather than the real
2,000-update recipe. Outputs are immutable and ordinary B8-loadable. No
held-out outcomes opened. This is plumbing proof, not a training-rate forecast.

Actual gameplay entrypoint (already run; H3, synthetic candidates):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.gameplay --fixture --fixture-seed 3403 --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth --candidate D256=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/mac-smoke-20261023-01/output/fit-0000.pth --candidate D1024=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/mac-smoke-20261023-01/output/fit-0001.pth --candidate D4096=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/mac-smoke-20261023-01/output/fit-0002.pth
```

All four arms completed, alive fraction1, no death, `NOT_A_DECISION`. Exact
clone/full-digest check passed. No timing is armed.

Final pack/preflight and supervisor validation (already run):

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.make_pack --commit 560e6526d884de605af45f4c3696b5a0a127b72d --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/pack-20261023-03 --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.colab_cli.supervisor --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/validate-20261023-03 --colab /Users/josenunez/.local/bin/colab validate /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/pack-20261023-03/canary-template.json
```

Both pass. Managed clean Python3.12.15, 64 import statements, 23 local modules,
no system/user site packages. The actual supervisor validation exits0. Pack03
manifest SHA256 `20403367b99764d6d7775903860da5eb381661c3c17ce7b2e377354a1a816579`.
Qualified registry path
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`,
SHA256 `967cc5802637a95827a4fd846ddfe97df3cfceed53ace7429088ddbeac2b39a0`,
tree `553af3f692515b2bf66173356fe4d099acc96942`. Transport unchanged.

Pack01/02 are preserved, refused during offline validation for device
declaration/pinned shell-variable syntax. Pack03 supersedes both; none allocated.

## Claude dispatch command — intentionally blocked now

The following is the exact wrapper command for inspecting the current prepared
template. It **must refuse** HOLD/`scientific_admission:false`. It has not been
executed. F4 judgment and F0 labels must precede a new admitted manifest; retain
the current immutable pack. No admissible main dispatch command exists yet.

```bash
/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh \
  bl-f3-canary-560e6526 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/pack-20261023-03/canary-template.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-f3 \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/colab-state-560e6526 \
  --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f3/pack-20261023-03/DISPATCH.txt
```

## Caps and kill numbers

- F3 now: **0u** until F0 labels and judged F4. Future fit series base10 /
  effective20u, 2x once, all attempts/setup/canary/export included, first cap wins.
- Blocked future L4 canary: base.5/effective1u; reserve.1/.2u;
  timeout10/20min; estimate5min; rate1.54u/h; one worker/math thread;
  lease180s; per-export32MiB; <=3 account-wide assignments. First attempt carries
  an empty ledger; at most two manually admitted mechanical completion attempts.
- F0 tranche1: base40/effective80u; no F3 authority to increase collection caps.
- Support doses256/1,024/4,096, fixed heldout64 including Watch6>=16;
  fit rescue lives192/960/4,032. At cap without floor: INSUFFICIENT, no extension.
- Offline harm ceiling **0.01/1,000** at natural prevalence, conservatively
  including unknown changed actions; actual frozen-export predictions required.
- Gameplay fixed128/mix H5000: D4096-D1024 **>=+20/LB90>0**;
  D4096-B8 **>=+40/LB90>0**; no mix UB90<0; survival A pooled margin-.05 /
  per-mix-.075; survival B scripted no-death gain>=+.03; boost[.5%,10%];
  Watch6 product deaths/1k paired95% favorable and mass relative LB95>=-5%; Holm.
- Dose-benefit UB90<=0 closes the lane. F2 negative => buy4096 only after
  D1024 beatsD256 in this harness. F4 must judge the exact funding statistic.
- Later serving assay remains separate: mixed Watch6 update p95<=11.688ms.

No tmux sessions/daemons, released checkpoints, serving pins, old hash-bound
documents/receipts or other worktrees were modified. Origin/main contributions
were merged and preserved. No merge conflicts remain.
