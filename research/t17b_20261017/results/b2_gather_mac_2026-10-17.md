# T17b B2 gather-only Mac measurement (2026-10-17)

This is host-only synthetic `_sample_fast` evidence, not a learner or promotion benchmark.
It uses two rings and the smoke-03-sized batch in
`research/redesign_m3c_20261008/bench_gather.py`.  The script includes a warm-up before
timing and creates no actor processes or tmux jobs.

Host: macOS 27.0 (26A5416b), Apple M5 Pro; Python 3.12.3.  Measured from baseline commit
`c4ac741159a6fcb162eb9a1388f468929adbe445` while the T17b merge was in progress.

| layout | gather threads | median ms/update | mean ms/update | samples |
| --- | ---: | ---: | ---: | ---: |
| raw | 0 | 3.222 | 3.243 | 300 |
| raw | 3 | 1.682 | 1.698 | 300 |
| raw | 4 | 1.744 | 1.789 | 300 |
| packed | 0 | 3.024 | 3.047 | 300 |
| packed | 3 | 1.363 | 1.369 | 300 |
| packed | 4 | 1.214 | 1.225 | 300 |

The omitted `g2` rows in the script's standard output were raw 1.980 / 1.991 ms and packed
1.774 / 1.777 ms (median / mean).  The required B2 rows are the `g0`, `g3`, and `g4` values
above.  Results are noisy development evidence; they are not interchangeable with B5's full
learner host/device measurement.

Reproduce:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./venv/bin/python research/redesign_m3c_20261008/bench_gather.py \
  --actors 2 --warmup 30 --updates 300
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./venv/bin/python research/redesign_m3c_20261008/bench_gather.py \
  --packed --actors 2 --warmup 30 --updates 300
```
