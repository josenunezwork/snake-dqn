# Ops tooling (2026-10-22) — make the safe path the only path

- `dispatch.sh NAME MANIFEST [--dispatch-txt F] [--cwd DIR] [--broker DIR] [--state-dir DIR]` — the ONLY way to launch a
  Colab supervisor job. Refuses BLOCK/HOLD/"do not run" lines in DISPATCH.txt, `scientific_admission=false`, list-typed
  `other_assignments`, reused names, and failed `validate`; launches in tmux `job-NAME` with an `EXIT=` marker; registers
  the log with the watcher; starts the watcher if needed. Enforced by `~/.claude/hooks/guard_dispatch.py`.
- `watchd.sh` — persistent daemon (tmux `watchd`) writing `snake-dqn-artifacts/ops/events.log`: JOB_ENDED, SOL_DONE (PIDs
  in `ops/watch-pids.txt`), COLAB_UNDERUSED (< 3 assignments for 10 min), MAC_IDLE (no python worker > 50 % CPU for 15 min).
- `wait-event.sh` — Claude keeps one background call of this running; it returns on the next event.
Register Sol jobs: `echo "$PID name" >> snake-dqn-artifacts/ops/watch-pids.txt`.
