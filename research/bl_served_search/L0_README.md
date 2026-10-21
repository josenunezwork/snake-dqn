# L0′ saved-smoke bookkeeping

`l0.py` implements the Round 6 L0′ / Round 7 X8 bookkeeping replacement. It
reads only the authorized immutable `t34-pathb-smoke-20261021` source and makes
no game replay, controller change, dispatch, or serving claim.

The receipt publishes four separate components:

- The own-deadline table compares each retained first fully-evaluated improving
  frame with its detection frame and measured completion time. A row without
  every field is explicitly invalid, never estimated.
- LAD reachability reports the exact-H500 LAD for all 75 T3 records and the
  D300 LAD for the fixed 34 rescue-capable records, including a LAD exactly at
  `detection + S*24` for `S={1,2,3,5,8,12,18}` seconds.
- The per-doom table reports each rescue-capable half-open window
  `[live + ceil(S*24), D300 LAD]`. Its
  `max(0, floor((D300_LAD - start) / 5) + 1)` count is a cadence-phase upper
  bound because the historical first periodic-root phase was not retained. It
  is not an observed dense-demand count or an admission bar.
- The dense-demand census requires a retained `bare_dies_within_h300` flag and
  `surviving_alternative` for every batch. Aggregate controller summaries and
  truncated event lists are insufficient. Missing batch fields produce an
  `INCOMPLETE` receipt while the available bookkeeping remains visible.

Run only after the governing amendment has been committed and published, under
low priority with one numerical thread. The output directory must be new.

```bash
nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ./venv/bin/python -m research.bl_served_search.l0 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/amendment08-l0
```

`RESULT_L0_PRIME.md` and `l0_prime_receipt.json` each receive the repository's
create-only `.sha256` and `.complete` sidecars. The output is development-only,
excluded from registered verdicts, and cannot authorize a latency bar, L2′
dispatch, a paced study, serving, or release.
