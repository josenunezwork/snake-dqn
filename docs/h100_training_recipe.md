# H100 training recipe — continue the Apex vector winner

**Scope:** this recipe covers the **Apex + 61-D vector stack** only. It is no longer the whole
codebase: the redesign branch adds a raster/PQN stack (`PQNTrainer`, `RasterDuelingNetwork`,
`BatchSim`) on its own track — see
[docs/ml_redesign_blueprint_2026-07.md](ml_redesign_blueprint_2026-07.md).

**Winner (post-consolidation, within this stack):** feedforward Dueling DQN (`ApexNetwork`) +
**61-D free-space** state.

**Historical record — what actually beat what.** GRU/DRQN *was* trained and lost paired,
frozen-opponent benchmarks to the feedforward model; that elimination is real (though it was
judged under the old, since-repaired gate). The **CNN was never trained** — it was written and
deleted during consolidation with zero checkpoints ever produced, so it lost nothing. The
**CNN question is open**, and the raster/PQN stack is re-testing it cleanly. Earlier drafts of
this file claimed both variants "lost every benchmark"; that claim is refuted by this repo's own
verification appendix (blueprint Appendix B, claims 1 and 10).

- **Winner checkpoint:** `saved_snakes/champion_a5_freespace_20260621.pth` — **warm-start from
  this.** Its historical headline (mean_mass 139.5 vs frozen 41.4 on 8 seeds; 131.6 on the
  16-seed reverify) was measured with the **retired alive-frames-only metric** and against a
  single-checkpoint opponent — do not compare a new candidate's mass integral to those numbers.
  Score the champion as the `--baseline` in the same paired run instead (Step 3).
- **Config:** [configs/free_space_v2.yaml](../configs/free_space_v2.yaml) — input 61, hidden 512,
  `use_free_space: true`, gamma 0.99, n_step 3. Its `rewards:` section is set to **exactly
  champion_a5's contract** (death −11, food_base 3.0, survival 0.01, death_length_scale 0,
  boost_segment 3.0) so a training-mode `--resume` passes the checkpoint contract. **Verified:
  champion_a5 resumes in training mode under this config and continues learning.**
- **Eval arena:** [configs/eval_free_space.yaml](../configs/eval_free_space.yaml) (widens frozen
  opponents to 61-D).
- **Frozen eval opponents (kept):** `best_apex_stage1_fs.pth` (arena opponent),
  `best_apex_fs.pth` / `best_apex_pre_fs.pth` (comparators).

## Provenance note — read before tuning the reward
The overnight report attributed A5's win to a "hold-mass" reward (food_base 1.5). That was a
**mislabel.** `champion_a5`'s checkpoint records the **high-food reward** (food_base 3.0,
death_length_scale 0) at both metadata locations, and its training log shows the high-flow
signature (avg reward ~320, food ~1200/episode). **The winning ingredient is the 61-D
free-space STATE FEATURE, not the reward** — free-space stops the snake self-trapping when long,
so it holds mass even while eating greedily. The reward sweep (A1–A7 tags like `fb15`/`fb10`)
likely did not actually vary the live reward in the campaign; don't trust those labels.
`configs/free_space_v2.yaml` above uses champion_a5's real reward so resume works. (Note: the
`seed_A5_dls3_fb15_s04.pth` seed carries the food_base 1.5 contract — that's the *intended*
seed config, not what the champion trained with. Don't pair it with this config.)

## Why the H100 actually helps here
The winner is **feedforward**, so it runs on the fast distributed Apex path: 1 GPU learner +
many CPU actors. (The deleted DRQN could not fan out actors — part of why it lost.) More actors
→ more environment steps in wall-clock, which is the binding constraint for this data-limited
task.

## Step 1 — set up the H100 box
```bash
git clone <this repo> && cd snake-dqn
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt -r requirements-gpu.txt
# copy the winner checkpoint up:  saved_snakes/champion_a5_freespace_20260621.pth
```
`DeviceManager` auto-selects CUDA on the H100 (there is no `--device` flag on apex_train; force
with `SNAKE_DQN_DEVICE=cuda` if needed).

## Step 2 — train (warm-started from the champion, distributed)
```bash
python src/scripts/apex_train.py \
  --config configs/free_space_v2.yaml \
  --resume saved_snakes/champion_a5_freespace_20260621.pth \
  --num-actors 32 \
  --total-steps 4000000 \
  --batch-size 512 \
  --buffer-size 500000 \
  --save-interval 5000 \
  --checkpoint-dir saved_snakes/h100_run1
```
- The config reward MUST stay matched to champion_a5 (death −11, food_base 3.0) or the resume
  contract aborts — that's by design; don't "fix" the reward here.
- Checkpoints roll to `*_latest_*.pth` / numbered files. **Resume from those, not from a
  `best_*` file** (best-save can freeze).
- **Run 2–3 independent seeds** of this command and gate each — outcomes on this task are
  high-variance run-to-run; don't assume one run reproduces the champion's mass.

## Step 3 — gate on the Mac (the promotion contract)
Pull a candidate back and eval on CPU (MPS is ~5× slower for this tiny model).

**The metric is the MASS INTEGRAL** — mean per-frame mass over the **total** episode horizon,
with dead frames contributing **0** ([`eval_stats.mass_integral`](../src/scripts/eval_stats.py)).
The old `mean_mass` averaged over *alive frames only*, under which a candidate that boosted to
mass 60 and died at frame 200 outranked one that held 35 for all 3000 frames — it could promote
regressions. `mean_mass_alive` is still reported, but it is a **legacy diagnostic**: never gate
on it. There is likewise no survival-threshold clause any more; survival is a column in the
report, not a gate (the integral already prices dying in).

Gate with **one paired invocation** — the candidate and the baseline play the same seeds, and
the rule is machine-enforced. Do not run two independent evals and eyeball the CIs; that throws
away the pairing that the variance reduction depends on.

```bash
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
  saved_snakes/<candidate>.pth --gate \
  --baseline saved_snakes/champion_a5_freespace_20260621.pth \
  --config configs/eval_free_space.yaml \
  --mixes frozen,scripted,mixed \
  --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --json-output logs/gate_h100_candidate.json
```

**The rule the code enforces** ([`promotion_decision`](../src/scripts/tournament_eval.py)):
promote iff the paired mass-integral delta is **> 0 at 95% CI on ≥ 2 distinct opponent mixes**
AND there is **no regression** (CI entirely below 0) **vs the scripted anchor mix**. The
`scripted` mix must be in `--mixes` or the anchor clause is unverifiable and the candidate is
rejected. Mixes are deduplicated, so one mix listed twice cannot satisfy the "≥ 2" clause.

**The exit code is the verdict**: `--gate` exits **0 = PROMOTE**, **1 = REJECT** (with reasons
printed). Don't re-interpret the table by hand — script off the exit code.

Opponent diversity is the point of the mixes: `frozen` cycles the `--opponents` pool,
`scripted` fills every slot with the ungameable `greedy_food` anchor, `mixed` alternates pool
and `random_safe`. Gating against 5 clones of one checkpoint (the old `--opponent` flag, still
accepted as a legacy alias) is single-opponent Goodharting.

Keep `--engine live` here: `simd` is ~100× cheaper but featurizes **raster** checkpoints only and
will reject the 61-D `vector61` champions.

Sizing the seed count is a power analysis, not a guess — `--pilot` runs the baseline alone and
recommends seeds for a 3% minimum detectable effect:
```bash
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py --pilot \
  --config configs/eval_free_space.yaml --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9
```
If it passes, copy the candidate byte-for-byte to a new `champion_*.pth` and freeze the old
champion as a comparator in the `--opponents` pool.

## Step 4 — watch it here with the winner
Once promoted, eyeball the new champion in the **web app** — that is the interactive UI.
`src/main.py` is headless-only and rejects `--load` without `--headless`/`--health-smoke`.
```bash
cd web/frontend && npm install && npm run build   # once
./venv/bin/python web/serve.py                    # -> http://localhost:8000
```
Every `.pth` under `saved_snakes/` is listed by `/api/checkpoints`; pick the new champion in the
Controls panel to load it live. **Don't edit `configs/default.yaml`** — `GameSession` reads the
checkpoint's `input_size` and selects the matching config itself (61-D → the mechanics-v2 arena).
The startup default is `DEFAULT_CHECKPOINT` in [web/backend/session.py](../web/backend/session.py),
still `champion_a5_freespace_20260621.pth`; repoint it only if you want a new boot default.

For a non-interactive check that the checkpoint loads and still learns (loss, target-action
validity, update count), use the health smoke:
```bash
./venv/bin/python src/main.py --health-smoke --config configs/free_space_v2.yaml \
  --load saved_snakes/<new_champion>.pth
```

## Don't expect kills
Across the whole lineage kills are ~0 — trained opponents avoid danger, so the hero wins by
**survival + growth**, not aggression. Don't set kill targets.
