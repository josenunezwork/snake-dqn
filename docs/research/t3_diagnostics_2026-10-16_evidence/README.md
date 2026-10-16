# T3 diagnostic evidence

Development diagnostics only. The registered T3 decision is unchanged.

- `bank-separation.json`: fixed 2,000-row sample identity, shard/checkpoint/status/margin hashes,
  per-final and pooled raw-Q separation and surviving-set agreement.
- `replay-summary.json`: all 20 replay receipts, exact mass accounting, fixed H5000 survivor
  cohorts, death distributions, length/Q-decile disagreements, and per-seed summaries.
- `validation.json`: code revision, commands, checks, and evidence hashes recorded at closeout.
- `margin-training-receipts.json`: hashes of all 15 original training summaries, each recording
  `label_bank_margin_scale = 1.0`.

Full post-transition mass/alive arrays and CTRL decision observations are retained locally at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-diagnostics-20261016/series/`.
The summary binds each NPZ with SHA-256. The input models and original probe rows are external
research artifacts; their absolute paths and hashes are retained in the JSON.

## Reproduce

Run from the repository root with the project virtual environment. These commands use one
CPU process at a time; parallel execution must stay within four single-thread numerical
processes in total. They do not launch training, touch tmux, or manage other processes.

```sh
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
t3_artifacts=/Users/josenunez/Projects/ml/snake-dqn-artifacts
t3_diagnostic_out="$t3_artifacts/t3-diagnostics-20261016"

./venv/bin/python -m research.t3_diagnostics_20261016.bank \
  --bank "$t3_artifacts/t3-rollout-labels-20261015/bank" \
  --ctrl-root "$t3_artifacts/redesign-m6-mac-20261014" \
  --t3-root "$t3_artifacts/redesign-t3-mac-20261015" \
  --status "$t3_artifacts/redesign-t3-mac-20261015/status.txt" \
  --margin-json "$t3_artifacts/redesign-m2b-20261008/data/margin.json" \
  --margin-scale 1.0 --sample-seed 20261016 --sample-count 2000 \
  --out "$t3_diagnostic_out/bank-separation.json"

for t3_arm in CTRL LABELS LABELS_KEEPV8 LABELS_KEEPV8_LO; do
  for t3_seed in 10 11 12 13 14; do
    ./venv/bin/python -m research.t3_diagnostics_20261016.replay \
      --name "${t3_arm}_s${t3_seed}" --out "$t3_diagnostic_out"
  done
done
./venv/bin/python -m research.t3_diagnostics_20261016.replay \
  --summarize --out "$t3_diagnostic_out"
./venv/bin/python -m research.t3_diagnostics_20261016.report \
  --bank "$t3_diagnostic_out/bank-separation.json" \
  --replay "$t3_diagnostic_out/summary.json" \
  --out /tmp/t3-diagnostics-tables.md
```

The renderer reproduces the tables and methodological text; the report's short reading and
validation narrative are written after inspecting the completed results. The bank evaluator
uses minibatches of 256 for raw-Q fit diagnostics; gameplay replay and shadow action selection
use the original probe's CPU rowwise inference and advisory masks.

Death distributions contain observed deaths only. Common-survivor length tables use the same
paired H5000-surviving cohort at all four timepoints. Mass decomposition is checked in integer
mass-frames before conversion to the per-frame mass integral. Pooled disagreement rates are
decision-weighted, and global risk deciles are computed once from the CTRL trajectories.
