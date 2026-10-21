# E3 Mac preflight and dispatch preparation

Outcome: **SMOKE_COMPLETE**, not a scientific GO/KILL. Development-only;
excluded from every registered verdict. No Colab calls or dispatch.

Protocol frozen/published first: `d85d75af` (PROTOCOL.md). Original T3 manifest:
`0b216da882b634e79ac640a9970a58157828da96e93ec4dbe68e6924c8141391`.

Real entrypoint: `nice -n 10 env ... bash research/bl_info_value/job.sh`.
One native math thread per process, serial collection followed by a two-process
byte-parity pass. The inherited process priority was nice15 (parent already
nice5), satisfying the requested nice10 reduction. Phase0 remained running.

| Measurement | Value |
| --- | ---: |
| Source worlds / roots / branch horizon | 3 / 6 / D300 |
| Serial reconstruction | 59.604876 s |
| Two-process parity qualification | 26.819192 s |
| Tiny fits plus repeated-train batch timing | 2.326124 s |
| Total real entrypoint | 89.576391 s |
| Replayed decisions by source world | 4337 / 3401 / 4304 |
| Exactly matched trainable parameters A/B/C | 270522 each |
| Actual exports found by supervisor inventory | 14 |

Saved B8 observation/mask/scalar/action/frame bytes, preceding histories,
unchanged B8 first-step and D300 mass-prefix controls, terminal deaths and
serial/parallel array bytes passed. Every actual export has both supervisor
sidecars. Smoke fits use only two epochs and cannot generate a scientific result.

Per-arm median Mac batch32 updates: A 0.010710 s, B 0.055812 s,
C 0.071033 s. Batch192 inference: A 0.027409 s, B 0.179067 s,
C 0.184555 s. Timings use throwaway copies and repeated TRAIN roots only.
Production remains the frozen60epochs, with validation epoch selection and
one held-out test look after all three arms finish.

Production inputs: 256 unique seeds/worlds,1536 unique roots,768 per class;
192/32/32 worlds and576/96/96 roots per class in train/validation/test.
Mixes:210scripted/30mixed/16frozen. Hash-bound archive:57,831,386bytes.

Projected six-process L4 allocation: **2.96535h /5.93070u** at2u/h:
2.54286h reconstruction +0.08916h fit CPU upper projection +20min
setup/coldJIT/export. The reconstruction extrapolates the actual two-process
qualification linearly to six workers and assumes4x slower L4 host cores;
fit projection credits no CUDA speedup. These are planning assumptions, not
measured L4 performance. Scaling the cold serial timing instead gives
3.24606h /6.49212u, retained as sensitivity. Fixed study size, depth, histories,
fit recipe and decision rule are unchanged.

Manifest uses356min timeout (at least2x projected whole-session estimate),
12.8u session cap,2u/h rate ceiling and3.2u untouched experiment reserve within
the overall16u envelope. First cap wins; no automatic retry. Operator must
review current endpoint prices/account assignment count before dispatch.

Artifacts and final absolute commands:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e3/DISPATCH.txt`.
Actual timing and terminal receipts are in sibling `preflight-01/`; source
bundles are `smoke-input-01/` and `full-input-01/`. Production has not run.

Qualification: Black/isort/flake8 on this package and `bash -n job.sh` passed.
Targeted tests:22passed,64deselected in23.71s with:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/bl_info_value tests/test_colab_cli_supervisor.py \
  -k 'bl_info_value or validate'
```

Independent read-only review found no remaining blocking defect. A future KILL
tests these specific ranking architectures; it cannot establish a general
ceiling for every full-resolution or recurrent policy.
