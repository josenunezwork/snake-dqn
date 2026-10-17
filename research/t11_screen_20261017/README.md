# T11 LOCAL30 development screen

Implements `docs/research/specs/t11_probe_spec_2026-10-17.md` and its binding
preregistration. Round 74 reserves 96 scripted / 32 frozen / 32 mixed H5000
worlds; round 75 is the single conditional scripted extension. This is not
promotion or serving qualification.

`watch.py` reads existing terminal training receipts before launching gameplay.
Any training failure establishes KILL. It never substitutes an aborted checkpoint
or reduces the registered five-seed population. The watcher uses at most three
single-thread CPU probe children plus its own process, and never controls tmux
or any process it did not start. `probe.py` reuses T2's rowwise evaluation and
records post-frame survivor masses at frames 1000 through 5000.

`audit.py` refreshes the immutable LOCAL30 seed audit against registered and
observed histories, including the complete T7 namespace. It writes a sibling
receipt without altering the bound preregistration receipt or source.

For this execution, seed 13 halted at 300160 transitions, exited 1 and has no
final checkpoint. The halt fixes KILL before gameplay. The other four final
checkpoints and all available telemetry are retained as descriptive evidence;
missing registered metrics remain null.

Run with CPU and numerical thread environment variables set to 1:

```sh
./venv/bin/python -m research.t11_screen_20261017.watch
```

Raw receipts/logs are in `snake-dqn-artifacts/redesign-t11-probes-20261017`.
Portable analysis evidence is under `results/`, and the report is
`docs/research/t11_screen_result_2026-10-17.md`.

Git workflow: after each task commit, fetch origin, merge origin/main, push
HEAD:main, then push t4p-20261016. Pre-existing untracked T4e files are unrelated.
