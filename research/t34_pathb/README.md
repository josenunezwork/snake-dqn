# T34 Path B deadline-feasibility harness

This Mac-only package is development evidence, never a release or strict gate.
It does not start a web service, use tmux, allocate cloud hardware, or exceed one
assistance worker plus its parent. `PROTOCOL.md` is the frozen action rule.

Run each command from the repository root with the project venv. Outputs must be
fresh, absolute directories; all manifests and records are create-only.

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run smoke \
  --out /absolute/path/to/t34-pathb-smoke
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run bank \
  --out /absolute/path/to/t34-pathb-dev --workers 1 --blocks 5 \
  --scripted-worlds 96 --frozen-worlds 48 --mixed-worlds 48 --product-worlds 25 \
  --horizon 5000 --max-hours 24
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run estimate \
  --out /absolute/path/to/unused --workers 1 --blocks 5 \
  --scripted-worlds 96 --frozen-worlds 48 --mixed-worlds 48 --product-worlds 25 \
  --horizon 5000
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run efficacy \
  --out /absolute/path/to/t34-pathb-dev --workers 1 --horizon 5000 --warmup 200 --max-hours 24
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run timing \
  --out /absolute/path/to/t34-pathb-dev --workers 1 --horizon 1500 --warmup 200
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.analyze --out /absolute/path/to/t34-pathb-dev
```

The default registered bank has 5 blocks × 217 worlds × 3 arms × H5000 at both
cadences. Its paced lower bound is **565.1 wall-hours**:
`5*217*3*5000*(1/12+1/24)/3600`. The corresponding reserved mathematical-thread
occupancy is **753.5 core-hours** (bare 1 + async 2 + sync 1 threads across the
three arms). Actual CPU use is unknown until the Mac timing measurement; neither
number is an estimate of the other. The owner should size a smaller registered
bank before starting a real run.

Terminal worlds use the canonical `tournament_eval`
live construction with each stored historical scripted/frozen/mixed roster;
they never replace those rosters with product FRP3+v8 opponents.

| Contract area | Harness evidence | Acceptance behavior |
| --- | --- | --- |
| §2 terminal H5000 | Canonical live historical rosters, completed paired bank | `NOT_ESTIMABLE` until all clauses and intervals exist |
| §3 continuous Watch6 | Paired bare/async/sync real sessions at 12 and 24 fps | `NOT_ESTIMABLE` until complete product bank |
| §4 timing | T15 `step` + `snapshot`, 200 warmup then paced probes | Browser/client latency remains `NOT_MEASURED` |
| resource/default | `--workers 1`; default-OFF Path B installation | Any collision, cap, or missing record is `INCOMPLETE` |
