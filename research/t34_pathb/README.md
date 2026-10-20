# T34 Path B deadline-feasibility harness

This Mac-only package is development evidence, never a release or strict gate.
It does not start a web service, use tmux, allocate cloud hardware, or exceed one
assistance worker plus its parent. `PROTOCOL.md` is the frozen action rule.

Run each command from the repository root with the project venv. Outputs must be
fresh, absolute directories; all manifests and records are create-only.

```bash
./venv/bin/python -m research.t34_pathb.run smoke \
  --out /absolute/path/to/t34-pathb-smoke
./venv/bin/python -m research.t34_pathb.run bank \
  --out /absolute/path/to/t34-pathb-dev --workers 1 --blocks 5 \
  --scripted-worlds 96 --frozen-worlds 48 --mixed-worlds 48 --product-worlds 25 \
  --horizon 5000 --max-hours 24
./venv/bin/python -m research.t34_pathb.run estimate \
  --out /absolute/path/to/unused --workers 1 --blocks 5 \
  --scripted-worlds 96 --frozen-worlds 48 --mixed-worlds 48 --product-worlds 25 \
  --horizon 5000
./venv/bin/python -m research.t34_pathb.run efficacy \
  --out /absolute/path/to/t34-pathb-dev --workers 1 --horizon 5000 --max-hours 24
./venv/bin/python -m research.t34_pathb.run timing \
  --out /absolute/path/to/t34-pathb-dev --workers 1 --horizon 1500
./venv/bin/python -m research.t34_pathb.analyze --out /absolute/path/to/t34-pathb-dev
```

`estimate` is only a conservative paced lower bound. Timing samples on this Mac
must size the actual run. The current runner refuses terminal efficacy until the
historical scripted/frozen/mixed live adapter exists; this protects the protocol
from silently replacing those rosters with product FRP3+v8 opponents.

| Contract area | Harness evidence | Acceptance behavior |
| --- | --- | --- |
| §2 terminal H5000 | Planned exact historical rosters and fixed analysis | Refuses to claim until the live adapter is present |
| §3 continuous Watch6 | Paired bare/async/sync real sessions at 12 and 24 fps | Development report only |
| §4 timing | T15 `step` + `snapshot` timer, ON/OFF and both cadences | Client latency is `NOT_MEASURED` |
| resource/default | `--workers` must be 1; Path B installation explicit in harness | Any missing/incomplete record is `INCOMPLETE` |
