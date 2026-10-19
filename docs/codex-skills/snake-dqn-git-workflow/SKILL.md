---
name: snake-dqn-git-workflow
description: |
  Use at the start and end of EVERY implementation job in /Users/josenunez/Projects/ml/snake-dqn (or any
  ../snake-dqn-* worktree): where to work (worktrees, branch names), how to commit and get the work onto
  origin/main (the owner wants GitHub to hold the full state at all times), what the background git-sync tmux
  job does, what must never be touched (live tmux jobs, released checkpoints, hash-bound docs/receipts),
  which tests and lint to run, and the exact shape of the final handoff message to the operator.
---

# Snake DQN: git workflow for the implementer (Sol)

Verified against the repo on 2026-10-18. Commands assume the project venv `./venv/bin/python`
(system python3 has no deps). Read `AGENTS.md` first; `docs/STATUS.md` wins on conflicts.

## Where you work

- **Primary checkout:** `/Users/josenunez/Projects/ml/snake-dqn`, branch `main`. Do NOT commit here:
  its `main` must stay clean so the git-sync job can fast-forward it (below). Treat it as read-only
  reference plus the home of `venv/` and `saved_snakes/`.
- **Your worktree:** a sibling directory `../snake-dqn-<tag>` (examples that exist:
  `snake-dqn-t9`, `snake-dqn-t17`, `snake-dqn-adv-league`, `snake-dqn-pactor`). Usually the planner
  (Claude) creates it and the handoff names it (`codex-handoff.sh -C <worktree>`). If you must create one:
  ```bash
  cd /Users/josenunez/Projects/ml/snake-dqn && git fetch -q origin
  git worktree add -b t<N>-<topic>-<YYYYMMDD> ../snake-dqn-<tag> origin/main
  ```
- **Branch naming:** `t<N>[x]-<topic>-<YYYYMMDD>`, e.g. `t30-keepalive-20261018`,
  `t20-selfdelete-key-20261017`, `t9-20261017` (topic optional). Lowercase, hyphens, date last.
- The real venv lives in the primary checkout. Some worktrees have a `venv` symlink to it
  (`snake-dqn-t9/venv -> /Users/josenunez/Projects/ml/snake-dqn/venv`), others none: check `ls -ld venv`
  and otherwise use the absolute interpreter `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`
  (`make PY=<that> test`). Creating the symlink is fine; never create a second venv.
- Other worktrees (`git worktree list` shows ~60) may host LIVE runs. Never edit, rebase, reset or delete a
  worktree you were not handed: strict runs re-hash their source closure mid-run.

## Commit and publish (standing owner instruction, 2026-10-15)

Commit often (every coherent step, not only at the end). After EACH commit:

```bash
git fetch origin
git merge origin/main            # the recorded standing instruction (branch is pushed/shared)
# or, if none of your branch commits have been pushed yet: git rebase origin/main
git push origin HEAD:main        # fast-forward only; never --force to main
git push origin <your-branch>
```

- Conflict rule: resolve only conflicts inside files YOU changed in this job. Any other conflict:
  stop, leave the tree in a resolvable state (`git merge --abort`), and report it in the handoff.
- Pre-commit hooks (isort, black@100, flake8, trailing-whitespace, end-of-file, check-yaml,
  large-file 2 MB cap) run on commit if installed (`make install-dev`). Fix, do not `--no-verify`.
- Do not commit checkpoints, result dumps or anything > 2 MB; artifacts belong in
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/<study>/` (not a git repo), referenced by path + sha256.
- Commit messages: imperative subject, <= 72 chars, body says what changed and why; mention the study/tag.

### The git-sync job (do not fight it)

tmux session `git-sync` runs `/Users/josenunez/Projects/ml/snake-dqn-artifacts/git-sync/sync.sh`:
PULL-ONLY. Every 10 min it `git fetch`es in the primary checkout and `merge --ff-only origin/main` into
local `main`; it skips when main is dirty ("main dirty, skip") or has unpushed commits. Log:
`snake-dqn-artifacts/git-sync/sync.log`. It no longer merges `t*` branches: YOU push to main yourself.
Never `tmux kill-session -t git-sync`; never run `tmux kill-server`.

## Never touch

- **Running jobs:** anything in `tmux ls` you did not start; processes matched by `pgrep -f`. A user-level
  hook (`~/.claude/hooks/guard_kill.py`) blocks tmux/caffeinate kills and `pkill -f`; the 2026-10-03
  incident killed the tmux SERVER (its argv looks like `tmux new-session -d -s ...`) and a pod kept
  billing. To stop one run you own: `tmux kill-session -t <name>` or the job's own shutdown path.
- **Released checkpoints** in `saved_snakes/`: `b8_s3_t7_20261016.pth` (served default, sha
  `3e649783...`), `frp3_m3_s12_u60000_20261005.pth` (`eec144bf...`), `champion_a5_freespace_20260621.pth`,
  `best_apex.pth`. Never overwrite, rename, re-save or "clean up". New checkpoints go to a run dir you own.
- **Serving pins:** `web/backend/served_checkpoint_pins.json` is filled only by its sibling tools
  (e.g. `research/t12_b8_serving_20261017/fill_strict_pin.py`) after a strict gate; never hand-edit.
- **Hash-bound bytes:** governance/research docs bound by sha256 (intents, receipts, `prepare` outputs,
  amendments listed as hash-bound in `docs/research/INDEX.md`), `research/*/PREREGISTRATION.md`,
  `OWNER_ADMISSION_*.json`, approved queue manifests under `research/*/jobs_approved/`, and any
  `receipt.json`/`decision.json`/`audit.json` under `snake-dqn-artifacts`. Add a sibling
  `*_erratum_<date>.md` / `*_status_<date>.md` instead (AGENTS.md rule).
- `QUEUE_REF` in `notebooks/colab_queue_worker*.ipynb` (pins the queue implementation): change only when
  the task is explicitly to upgrade the queue code, to a full published SHA.
- New Colab job transport is `research/colab_cli`; keep its supervisor and runtime changes separate from legacy Q1
  queue/notebook files. A CLI manifest pin must be a full published commit before allocation.
- `AGENTS.md`, `docs/STATUS.md`, `CLAUDE.md`, `~/.codex/*`: only when the task says so.

## Tests and lint before you declare done

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
# lint (what CI runs on src/ and web/backend; pre-commit applies the same tools to every staged .py)
make PY=$PY lint                               # black --check --line-length 100, isort --profile black --check-only, flake8
make PY=$PY format                             # auto-fix src/ web/backend
$PY -m black --line-length 100 research/<dir> tests/<file>.py   # research/ and tests/ are NOT in make's SRC: format them yourself
$PY -m isort --profile black research/<dir> tests/<file>.py
$PY -m flake8 research/<dir> tests/<file>.py   # .flake8: max-line-length 100, ignores E203/W503
# tests
make PY=$PY test-fast                          # == pytest -q -n auto -m "not slow"
$PY -m pytest -q -m "not slow"                 # the CI gate (.github/workflows/ci.yml); must stay green
$PY -m pytest tests/test_<area>.py -q          # targeted; -k <keyword> also fine
make test-web                                  # only if you touched web/frontend (npm ci && npm test)
```

- `pytest.ini`: `strict_markers = true` -- a new marker must be registered there or collection fails.
  Markers: `slow`, `integration`, `unit`, `requires_gpu`.
- Run tests from the repo/worktree root (`testpaths = tests`). Use `SNAKE_DQN_DEVICE=cpu` for
  determinism; never leave a test depending on MPS/CUDA without `requires_gpu`.
- No mypy in this project; type hints are reviewed, not checked.
- If you changed a launcher or job script, also run its real entrypoint once on a tiny real input
  (see `snake-dqn-colab-jobs`, `snake-dqn-queue-jobs`) -- unit tests alone do not count.

## Final handoff message (the operator reads only this)

Put ALL of the following, in this order, in your final message; no prose beyond it:

1. **Where:** worktree path, branch, `git rev-parse HEAD` (full SHA), and whether `origin/main` contains it
   (`git branch -r --contains HEAD | grep origin/main`). If not pushed, say why.
2. **What changed:** files added/modified (paths), one line each; behaviour change in one sentence.
3. **Verification run:** every exact command you ran (lint, pytest selections, entrypoint smoke) with
   result counts (`N passed, M deselected`) and measured numbers (per-unit time, rates) when relevant.
4. **Operator commands:** copy-paste-ready, absolute paths, env vars inline, in execution order
   (e.g. the `q enqueue ...` lines, the `tmux new-session -d -s <name> ...` line, the `--dry-run` first).
5. **NOT done / deviations / risks:** anything in the spec you skipped or changed, open questions,
   things that need the owner (spend approval, Drive clicks, Keychain items), and merge conflicts left.
6. **Nothing touched:** confirm you did not kill or modify tmux sessions, released checkpoints, pins or
   hash-bound files (or list exactly what you did touch and why).

Numbers and SHAs over adjectives; "should work" is not a verification.
