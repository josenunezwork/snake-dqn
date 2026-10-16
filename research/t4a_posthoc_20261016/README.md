# T4a post-hoc diagnostics

Diagnostic only. These tools do not change the registered gameplay decision or
authorize training or promotion. The binding request is
`docs/research/specs/t4a_posthoc_spec_2026-10-16.md`.

Use the repository virtual environment, CPU only, with at most three concurrent
single-thread numerical processes. Do not use MPS. The tools set native thread
limits before numerical imports. No tool controls training sessions or processes.

## Part 1

The boost collector passively observes the final student's greedy decisions on
its own or matching-seed CTRL round-62 scripted trajectory. It validates the
replayed trajectory against the original probe's terminal records before writing
its decision arrays and receipt. It records raw, resolved-mask, and advisory-mask
greed separately; training telemetry uses resolved-mask greed before exploration
or shield overrides, while the probe executes advisory-mask greed.

Example (run from the repository root):

```sh
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
posthoc_out=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4a-posthoc-20261016
./venv/bin/python -m research.t4a_posthoc_20261016.boost capture \
  --trajectory LABELS_s10 --student LABELS_s10 --out "$posthoc_out"
./venv/bin/python -m research.t4a_posthoc_20261016.boost capture \
  --trajectory CTRL_s10 --student LABELS_s10 --out "$posthoc_out"
```

Repeat for seeds 10–14 and arms CTRL, LABELS, LABELS_KEEPV8. CTRL-on-CTRL
is needed only once per seed. This produces 25 distinct captures. Summarize only
after all captures and their input hashes validate:

```sh
./venv/bin/python -m research.t4a_posthoc_20261016.boost summarize \
  --out "$posthoc_out" --json "$posthoc_out/boost-summary.json"
```

The optional `--report` appends generated Part 1 tables to the existing T3
diagnostic document, refusing to append the section twice. Interpretation is
written after inspecting the completed evidence.

## Part 2 execution gate

Do not begin Part 2 numerical capture until the two artifact lane receipts record
successful exits and all five `TWO_SIDED_s10..14/final.pth` files exist. Read only
the artifact receipts; do not inspect or control tmux or enter the protected
`snake-dqn-t4a` worktree. Round-62 diagnostic comparisons are separate from the
registered round-65 gameplay evaluation.
