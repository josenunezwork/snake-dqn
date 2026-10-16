# T4a two-sided-bank screen

This package watches the registered `TWO_SIDED` final checkpoints and evaluates
them on fresh development worlds under
`docs/research/specs/t4a_prereg_2026-10-16.md`.  It is development evidence
only and cannot promote or serve a checkpoint.

The watcher owns only its own CPU `nice` probe and descriptive re-roll children.
It never reads tmux and always permits at most two single-thread children, so it
does not address the training lanes or unrelated processes.  It probes all CTRL
finals and the incumbent before candidate finals.  A failed child stops the
watcher; it is never retried automatically.

```bash
PYTHONPATH=$PWD OMP_NUM_THREADS=1 ./venv/bin/python -m pytest -q \
  tests/test_t4a_screen_watcher.py tests/test_t4a_screen_banks.py
PYTHONPATH=$PWD ./venv/bin/python -m flake8 research/t4a_screen_20261016 \
  tests/test_t4a_screen_watcher.py tests/test_t4a_screen_banks.py
PYTHONPATH=$PWD ./venv/bin/python -m research.t4a_screen_20261016.watch_t4a
```

Before probes start, the watcher saves a fresh `seed_bank_audit.json` that
reserves rounds 65 and 66.  Round 66 is added only by the one registered
extension decision.  Outputs are kept under
`redesign-t4a-probes-20261016/`; the result document is
`docs/research/t4a_screen_result_2026-10-16.md`.
