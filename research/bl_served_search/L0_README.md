# L0 saved-timeline replay

`l0.py` is the 0-unit development-only replay specified by Round 5 judgment
section (f). It reads only the smoke's retained `lad-*.json` controller records
for the counterfactual timeline. It may inspect paired `n1-*.json` records to
establish the fixed 34 rescue-capable / 41 delay-only T3 denominator and to
describe why they cannot replace the LAD timeline.

The frozen grid is `k={1,2,3,5,10,20}`. LAD-first prioritizes candidate frames
that are 16–21, then 88–100, then 248–254 decisions before doom. The replay is
valid only if every LAD record supplies every evaluation's completion seconds
and candidate frame. Missing either field publishes `INCOMPLETE`; it must never
be guessed from a first-improvement summary or a linked N=1 row.

Run only under low priority with one numeric thread per process:

```bash
nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.l0
```

The runner uses no more than four processes and writes create-only exports at
`snake-dqn-artifacts/big-lever/exp/bl-l2/`: `l0/input_audit.json`,
`RESULT_L0.md`, and `l0_receipt.json`, each with `.sha256` and `.complete`
sidecars. This result is development-only and excluded from registered verdicts.
