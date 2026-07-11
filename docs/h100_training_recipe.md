# H100 training recipe — continue the one winner

**Winner (post-consolidation):** feedforward Dueling DQN (`ApexNetwork`) + **61-D free-space**
state. The only architecture in the codebase now; GRU/DRQN and CNN were removed after losing
every paired, frozen-opponent benchmark.

- **Winner checkpoint:** `saved_snakes/champion_a5_freespace_20260621.pth` — mean_mass 139.5 vs
  frozen 41.4 on the 8-seed benchmark (131.6 on the 16-seed reverify). **Warm-start from this.**
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
Pull a candidate back, eval on CPU (MPS is ~5× slower for this tiny model). Use the **full 16
seeds** — the 8-seed headline hides a tail (seed 10 survival 0.40):
```bash
./venv/bin/python src/scripts/tournament_eval.py saved_snakes/<candidate>.pth \
  --opponent saved_snakes/best_apex_stage1_fs.pth \
  --config configs/eval_free_space.yaml \
  --frames 1500 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --json-output logs/eval_h100_candidate.json

# score the current champion the same way for a paired comparison
./venv/bin/python src/scripts/tournament_eval.py saved_snakes/champion_a5_freespace_20260621.pth \
  --opponent saved_snakes/best_apex_stage1_fs.pth \
  --config configs/eval_free_space.yaml \
  --frames 1500 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --json-output logs/eval_h100_champion.json
```
**Promote only if** the candidate beats `champion_a5_freespace_20260621.pth` on mean_mass with
**non-overlapping 95% CI** AND holds **survival ≥ 0.94** on the full 16 seeds (watch seed 10).
If it wins, copy it byte-for-byte to a new `champion_*.pth` and freeze the old champion as a
comparator.

## Step 4 — run it here with the winner
Once promoted, point local inference at the new 61-D champion:
```bash
python src/main.py --config configs/eval_free_space.yaml --load saved_snakes/<new_champion>.pth
```
(Or set `configs/default.yaml` to `input_size: 61` + `use_free_space: true` and the load path so
the GUI launches the winner by default.)

## Don't expect kills
Across the whole lineage kills are ~0 — trained opponents avoid danger, so the hero wins by
**survival + growth**, not aggression. Don't set kill targets.
