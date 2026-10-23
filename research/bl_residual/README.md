# F3 frozen-B8 residual harness

F3 is implemented and frozen for **synthetic verification only**. No factory
labels were opened. [F4_DRAFT.md](F4_DRAFT.md) is for Astra review and is
**unratified**. Factory fits require its independently approved immutable
successor. Gameplay observations always say `NOT_A_DECISION`; the scientific
analyzer, P7 trajectory probes, product-session design and serving assay must be
fixed by that review before scientific admission. Nothing changes serving.

`model.py` freezes B8, adds a zero-initialized `Linear(256, 6)` head to the
`a100_a2.model.b8_features` cache, and folds the final residual into a standard
`Ego2sNet` value/advantage checkpoint. Zero initialization reproduces B8.
Trained fusion is algebraically equivalent with float32 reassociation; offline
action metrics run the **reloaded fused checkpoint rowwise** so near-tie actions
are measured on the delivered controller.

`FitRecipe` fixes 2,000 Adam fp32 updates, batch 256, lr .001, seed 3403,
preference/growth/KL-anchor coefficients 1 and residual L2 .0001. Small doses
repeat deterministic sampled rows for the same update count. Exact survivors
beat exact dying actions on decisive roots; ordinary complete branches supply
mass preferences and B8 retention. Combined sampling and positive-mass-gap
preference weights are capped at 5. Unknown/skipped actions never become
negatives. Non-finite input/Q/loss/gradient/parameters refuse the fit. The final
unit includes optimizer, head, generator RNG and update count; no intermediate
checkpoint selection is provided.

## Input and split contract

The F0 adapter consumes pickle-free NPZ partitions with:

- `local[N,7,31,31]`, `global[N,4,37,37]` uint8; `scalars[N,60]` float32;
- `q[N,6]`, H300 `mass[N,6]`, natural-archive p/q `sampling_weight[N]` float32;
- physical `legal[N,6]`, served advisory `action_mask[N,6]`, `died[N,6]`,
  `ordinary[N]`, `natural_stream[N]` boolean; recorded integer `bare[N]`;
- `status[N,6]` strings `EXACT`, `NOT_EVALUATED`, `UNKNOWN`;
- nonempty string `parent_world`, `life_id`, unique `row_id` arrays.

Scalars are float32, as in the actual B8 observation contract. `bare` must match
the served advisory mask with normal-only fallback. Native F0 exports retain
their complete provenance and audit fields; this adapter is an explicit fit
projection, not a replacement label format or simulator.

`data.allocate_lives` consumes metadata only: `life_id`, ultimate `parent_world`,
`stream` (`S-Watch6`/`S-gate`), `rescue_bearing`, unique frozen
`allocation_index`. Parent hashes assign five folds; fold zero is held out.
The fixed reserve is 64 rescue lives including 16 Watch6 lives. Selected dose
support is 256/1,024/4,096 including the reserve, hence 192/960/4,032 training
rescue lives. Train prefixes follow append-only indices; passing `prior` refuses
changed metadata or insertions into the earlier prefix. Restores, audits,
extensions and future DAgger rows inherit the ultimate parent. The pool may need
more collected lives than selected support. A missing reserve or dose is
`INSUFFICIENT`, never a smaller N or automatic cap increase.

The separate ordinary archive includes **non-rescue lives**, all on the same
held-out parent side. It supplies the natural-prevalence denominator, weighted
with p/q cap 5 and reported ESS. An enriched doom-only sample cannot establish
natural admission. Unknown changed actions count in the conservative harm upper
rate. The threshold is <=0.01 harmful overrides per 1,000 natural decisions.
Admission is unavailable without both an immutable complete archive census and
actual exported-checkpoint predictions. Reliability bins/ECE/Brier describe
uncalibrated scores; no calibrator or threshold is fitted on holdout outcomes.

`heldout.py` verifies the complete final checkpoint, its control/allocation
identity and real/non-synthetic 2,000-update metadata before opening outcomes.
The verified `life-allocation.json` is input-hash-bound to that candidate;
unselected rescue roots and foreign parents are refused. Training partitions
alone are uploaded to Colab; held-out outcome files stay on the Mac until final
fit units are immutable. An archive census uses schema
`bl-residual-natural-archive/v1`, exact table hash, `complete: true`, sorted
`natural_row_ids`, `parent_worlds`, `eligible_life_ids` and
`includes_non_rescue_lives: true`, backed by F0's verified life inventory.

## Gameplay hook and existing exact code

`gameplay.py` composes the existing H5000 terminal-hero evaluator
`bl_served_search.ceiling.run_arm(dense=False)`, `bank_engine.build_world`,
`clone_world_exact` and `full_digest`. It regenerates standard plans, checks
128 worlds per mix, disjoint seed/parent identities, explicit training/holdout
exclusions and a separately hash-bound collision receipt. Production is
Mac-only, exact H5000, and requires actual hash-bound reviewed registration
files. The <=3-step scripted fixture is excluded from inference. Boost share is
explicitly unavailable in this evaluator; product/boost qualification and the
scientific decision analyzer remain F4 work.

The collector retains `WatchDriver.snapshot/restore`, `clone_world_exact`,
`full_digest`, `exact_stop.run`, and native v8 unchanged. F3 does not implement a
second snapshot, simulator, branch search, opponent tape, or label generator.

## Caps and dispatch boundary

F3 is **0 Colab units until labels exist and F4 is judged**. Under the owner's
2x rule, the ruling's <=10-unit fit envelope is base <=10 / effective <=20,
including setup/export/canaries/recovery. F0 tranche 1 remains base40/effective80
units. The prepared future canary template is base .5/effective1 unit, reserve
.1/.2, timeout10/20 minutes, estimate5 minutes at1.54u/h, all within the fit
series. It is deliberately `scientific_admission: false` with `DISPATCH.txt`
HOLD. It grants no runtime and must not be edited in place to admit a job.

Qualified CLI registry:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`;
qualified commit `f21a92edb86550df5263dbed4be7e2ff0bfa4c17`, tree
`553af3f692515b2bf66173356fe4d099acc96942`. `make_pack` verifies it, requires a
published enclosing source commit, and invokes `tools/ops/import_preflight.py`
on a sparse archive in a fresh declared-dependencies-only managed Python3.12
venv before creating the pack. This checks the fit runtime, not the Mac gameplay
hook's dependency closure. A genuine same-pin L4 family canary is still needed
before any future production manifest is admitted.

`protocol.build_ledger` reads immutable allocation metadata, filenames, sizes,
hashes and completion markers without opening payload outcomes. Runtime skips
exact verified units and rejects foreign allocations. Each completion attempt
has fresh output/manifest/receipt paths; one original plus at most two manually
admitted mechanical completions, never automatic scientific retries. At a later
dose stage, dispatch **only the new dose** and retain lower-dose immutable
checkpoint references; do not refit earlier doses under a new allocation.

## Reproduce the authorized local smoke

Use fresh paths on a repeat. The recorded smoke paths are already occupied.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f3
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.make_pack \
  --local-smoke-inputs-only \
  --out /absolute/fresh-f3-smoke/inputs \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.job \
  --input-dir /absolute/fresh-f3-smoke/inputs --output /absolute/fresh-f3-smoke/output \
  --mode smoke --device cpu
```

Prepare a pinned, still-blocked Colab template after publication:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f3
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.make_pack \
  --commit HEAD --out /absolute/fresh-f3-pack \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
```

The **only** dispatch route is the operator-owned wrapper below. This exact
template command is expected to **refuse HOLD/false admission**; it is recorded
for review, not execution now. Claude prepares a new admitted manifest after F4,
labels, measured pricing and same-pin canary proof; earlier template bytes stay
immutable. Never invoke the Colab allocator directly.

```bash
/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh \
  bl-f3-canary-REPLACE_WITH_ADMITTED_ID /absolute/fresh-admitted-f3-pack/canary.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-f3 \
  --state-dir /absolute/fresh-admitted-f3-state \
  --dispatch-txt /absolute/fresh-admitted-f3-pack/DISPATCH.txt
```

Future held-out one-look command (only a real reviewed final is accepted):

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_residual.heldout \
  --checkpoint /absolute/verified/fit-0000.pth --control /absolute/verified/control.json \
  --allocation /absolute/verified/life-allocation.json \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --table /absolute/heldout.npz --archive-receipt /absolute/natural-archive.json \
  --out /absolute/fresh-heldout-look.json
```

The F4 draft retains gameplay kills exactly: D4096-D1024 >=+20/LB90>0;
D4096-B8 >=+40/LB90>0; no mix UB90<0; all T34 survival/product/boost bars and
Holm. Dose-benefit UB90<=0 closes the lane. Failure to clear a positive bar is
reported inconclusive under the draft, not promoted. F2's negative result
conditions purchase of D4096 on D1024 beating D256 in this harness.
