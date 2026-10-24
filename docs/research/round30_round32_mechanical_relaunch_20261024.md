# R30 lifecycle repair and R32 transport retry preparation — 2026-10-24

Both canaries are prepared and dry-run only. No Colab allocation, stop, dispatch,
real-label fit, validation look, gameplay, serving change or release was performed.
Previous receipts, frozen checkouts and hash-bound evidence were preserved.

## R30 mechanical repair

`research/bl_round30/job.py` now accepts an absent or pre-created empty output
directory. It refuses symlinks (including dangling links), regular files, and any
nonempty output directory before publishing allocation or payload evidence. This
matches the R32 lifecycle fix at `c1ab2750`.
`tests/test_bl_round30_job.py` reproduces the supervisor layout, verifies complete
canary exports, and proves repeat attempts and existing output collisions refuse
without changing their bytes.

Code commit on the integration branch: `cbff80ce`. The frozen runtime pin is
`c1bf44c89e116bc29f27293895bbb9da326c00ff`, based on the earlier qualified R30 pin
`72b6fcf0583c216f3185b9c5c31ae9abcb861a3f`. The two changed files are identical in
both commits. An ancestry-only merge (`91f4e4e2`) publishes the frozen pin to main
without rolling main's newer transport back. Frozen checkout:
`/Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024`.

The final source-bound Mac preflight passed: **78 tests in 3.18 seconds**, nice15,
one numerical thread, no xdist. At execution the D1024 driver was no longer live
and its Mac lease was absent, so the existing quiet-window launcher held its
shared serial lock and exclusive Mac lease. No factory process or lease was altered.

Exact preflight command, run from the frozen checkout (caller already nice15):

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_round30.local_check \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/sol-verification/lifecycle-c1bf44c8-20261024-01
```

Black (`--check --workers 1 --line-length 100`), isort (`--check-only --profile
black`), and flake8 (`--jobs 1`) passed for both changed Python files. Bash syntax
checks passed for the two generated dispatch scripts; `git diff --check` passed.
The reviewed preflight includes the job tests and focused supervisor/canary
admission tests. No full repository test suite or native CUDA run was performed.

R30 canary caps are unchanged: L4, estimate 5 minutes, timeout 20 minutes,
2/3 unit, 1.54 units/hour, reserve 1/9 unit, max export 128 MiB, default lease 180 s.
Its bounded short-canary exemption remains; scientific mains retain resume rules
and all admission gates. A new synthetic main template remains blocked.

## R32 same-pin preparation

Runtime pin remains `c1ab27508ef097108260b89061564f81aaae26fc`.
The new package changes only `job_id`, `output_dir`, and input source paths.
Input bytes, frozen allocation namespaces, empty verified ledgers, source paths,
scientific rules and compute envelopes are identical to the old package.
No R32 code change or fresh scientific allocation was made.

The previous receipt is `INCOMPLETE`, `stop_verified:true`, with zero exports.
The actual log records `TimeoutExpired` during `colab exec` (51.5968 seconds),
a transport/control failure. The old checkout's dispatch wrapper lacks
`--dry-run`; its argument parser refused that flag without operational effects.
The final commands therefore use the R30 frozen broker for both jobs, with
qualified transport tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
R32's runtime source is still extracted and validated at `c1ab2750`.

R32 caps remain CPU, estimate 5 minutes, timeout 20 minutes, 0.02666666666666667
unit, 0.08 units/hour, reserve 0.0044444444444444444 unit, max export 64 MiB,
lease 180 s. Its resume declaration and maximum two outcome-blind completion
attempts are unchanged. This manually prepared operational retry does not start
an automatic retry loop.

## Exact operator commands

Each command below was run with an additional `--dry-run`; both returned
`PASS_DRY_RUN ... allocations=0 live_accounting=NOT_CHECKED`.
Neither new output/state directory nor dispatch reservation exists. These commands
perform the actual operator dispatch; they were not executed without that flag.

R30-1 L4 synthetic rehearsal:

```bash
/Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024/tools/ops/dispatch.sh r30-1-rehearsal-c1bf44c8-01 /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-c1bf44c8-r301-01/job.json --broker /Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024 --cwd /Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024 --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/broker-r301-rehearsal-c1bf44c8-01
```

R32 CPU synthetic canary:

```bash
/Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024/tools/ops/dispatch.sh r32-cpu-canary-c1ab2750-02 /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-02/canary-template.json --broker /Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024 --cwd /Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024 --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/state-cpu-canary-c1ab2750-02
```

Equivalent scripts (append `--dry-run` to repeat offline validation):

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-c1bf44c8-r301-01/DISPATCH.sh
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-02/DISPATCH.sh
```

Live account inventory and debit are deliberately not queried by dry-run. Actual
dispatch performs fresh locked accounting through the unchanged qualified
supervisor. Prior failed attempts remain charged to their series. R30's known
failed canary lasted 369.2257 seconds and R32's 543.2162 seconds; future main
admission must include them and any other prior liabilities, and cannot reuse a
full original series envelope. No main is admitted by this preparation.

## Immutable preparation evidence

- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/sol-verification/lifecycle-c1bf44c8-20261024-01/preflight.json` — SHA256 `1eeaf536bb67176201bcef70cb1316516a18784a05a9cbb643391220ed59b193`.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-c1bf44c8-r301-01/job.json` — SHA256 `7fdc3dd500c5510c73813869813092bde480e45c2208d77757cd3ddffccaca88`.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-c1bf44c8-r301-01/PREPARATION.json` — SHA256 `97a06f5d853111230e4ed6284cdd0e8843667836756102277c3d4ab779e7a098`.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-c1bf44c8-r301-01/DISPATCH_DRY_RUN.json` — SHA256 `7282f18f50d2328ecfec6e9696594e3581f0a252203b7928b4d994a6d9e4d71f`.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-02/canary-template.json` — SHA256 `5c3c2c8a45096b3daaf86bb2c87369978c0bece15b710f531ef4d6bf54775696`.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-02/REDISPATCH_PREPARATION.json` — SHA256 `e0b4a9bb13f5d038620859e1b5ea20f9759d544105ca4d87f79ac40e0fba79bf`.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-02/DISPATCH_DRY_RUN.json` — SHA256 `5e28b7ab52c32548288366ce0f7e07180e487b3e2dc58a12dd1661c8a83c743a`.

Independent Codex-V LIGHT report is recorded in the sibling
`round30_round32_mechanical_codex_v_light_20261024.json` with verdict **ACCEPT**, zero actionable findings. The reviewer independently
verified all 257 preflight source hashes and reran both exact dry-run commands.
The reviewer owns no product edits and performs no operational side effects.

Untouched tmux sessions observed: `git-sync`, `sol-r30r32relaunch`,
`sol-v8fastpath`, `watchd`. Released checkpoints, serving pins and previous
hash-bound files were not modified. The released B8 file was read/copied only
as the unchanged initializer for the synthetic canary.
