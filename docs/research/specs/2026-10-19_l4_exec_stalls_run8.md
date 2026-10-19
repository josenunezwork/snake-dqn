# Spec: L4 exec stalls kill the arm wherever they land (run 8)

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT. Follows
`2026-10-19_l4_acceptance_lease_stall.md` (fix 2e38e645).

## Evidence

Run 8 (source 2e38e645, final-only):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792395748/`

- CPU arm: PASS again (all CPU gates; ~0.022 units).
- L4 arm: INCOMPLETE at 247 s, stop verified. Error:
  `TransferError('upload .../bootstrap.sh called at supervisor.py:1227: deadline reached during telemetry recovery')`.
  L4 `cli_calls`: call 5 upload 42 s, call 6 upload failed; ...; calls 18-22 = bootstrap.sh
  upload (19.7 s) then lease-heartbeat uploads (39.4 s, 1.0, 1.1, 1.1 s); call 23 = exec
  timed out at 120 s; the bootstrap transfer total reached 142 s and the upload budget
  ended the arm. The 64 MiB download and watchdog never ran.
- Run 7 (`acceptance-1792394444`) L4 died the same way at a different call site: heartbeat
  uploads (41 s) then a 120 s exec timeout inside `poll_rpc`.
- Runs 3-6 (same artifacts dir, earlier `acceptance-*`) were INCOMPLETE on transport/timeouts.

## Open question for you to answer from the receipts

Why do 120 s exec hangs happen on the L4 arm (twice in two runs, both shortly after
Contents heartbeat uploads), while the CPU arm absorbs its stalls? Candidates include:
Contents PUT / exec contention on the Jupyter server, the runtime still busy with bootstrap
or dependency install, kernel restarts on GPU runtimes, or plain provider latency. Use the
`cli_calls`, `transfers`, `contents_transfers`, poll/usage files and job logs across runs
3-8 (both arms). Write what the evidence supports and what it cannot settle.

## Required outcome

1. Fix the **cause** if the evidence identifies one (e.g. serialize heartbeats against exec,
   or delay heartbeats until the runtime is ready). Otherwise, make every exec-bearing
   operation on the pre-download path (source delivery, upload verification, probe/phase
   polling) survive one ≤120 s exec hang through the same verified-idle recovery `poll_rpc`
   now uses, within an explicit bounded budget. Do not raise caps or weaken any gate; no
   resubmission while the kernel may be busy; no extra allocation; stop stays verified.
2. Fake-Colab regression replaying run 8's sequence (42 s upload, failed upload, heartbeat
   burst, 120 s exec timeout during upload verification) that refuses on 2e38e645 and passes
   the L4 final-only path after the fix; keep the over-budget refusal test.
3. ACCEPTANCE.md: short run-8 evidence paragraph and your diagnosis.
4. `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_colab_cli*.py -n 4`
   all green; Black/isort/flake8 on changed files.

## Rules

Work only in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch
`t31-colab-safety-20261018`); commit and push when green. No live Colab commands, no tmux,
no edits under `snake-dqn-artifacts`, no S1 study code. Final message: diagnosis, change,
test count, what remains unverified.
