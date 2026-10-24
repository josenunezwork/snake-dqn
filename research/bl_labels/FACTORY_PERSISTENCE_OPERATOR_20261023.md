# Factory persistence operator handoff — 2026-10-23

Operational pin: `caafdf5d079bd73bcfbde6f9356dc0f60de42e0f`, published on
`origin/main` and `t34-f0-nice-interim-20261023`. This handoff is a later
documentation record; dispatch the exact operational pin above.

Binding policy: **ALLOW_PERSISTENCE_30S**, ID
`factory-nice15-persistence30/v1`. Astra ruling:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_PERSISTENCE_20261023.md`,
SHA256 `35295b5b4669bec32a61736c1d5dbb5d9c70015208d6f06db06e5f064bca7c32`.
Retained interim ruling SHA256:
`ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd`.

The ordinary-production priority veto now tracks `(PID, native Darwin birth
tick)` with the raw OS precision recorded. Seven consecutive qualifying
observations must span at least 30 seconds. No CPU threshold is used. An absent
or nice >=15 observation clears the streak; relevance survives reparenting and
exec, and known foreign identities cannot become factory-owned. Runtime pending
candidates produce telemetry without pausing admissions or changing slowdown
windows. Confirmed trips latch and use the existing cooperative stop/drain.
Failed or stale observations fail closed, with a shared 15-second full-observation
deadline across bounded commands. Verified shared/custom-socket tmux servers
cannot seed pane descendants; the actual executable must match the installed
trusted tmux binary, and its socket must report that exact server PID.

Startup and required recovery checks use the same seven-observation window.
Cleared transients may pass; final pending candidates return
`HOLD_PENDING_FOREIGN_PROCESS`. Recovery remains under the common serial-test
lock, records final health/top consumers and a digest-bound process journal,
and emits only `bl-labels-f0-slowdown-host-check-persistence/v3`, bound to policy,
both ruling hashes, exact operational pin, trip epoch and window metadata.
V2/foreign receipts remain immutable and cannot qualify a v3 check. No absent
Codex, CPU-budget, or full-companion-monitor qualification is claimed.

`tools/ops/dispatch.py` starts its Colab pane and watcher shell through a verified
nice floor with exec replacement. The real Colab launcher also reaches effective
nice >=15 before its caffeinate/supervisor chain. `tools/ops/agent_nice.sh` already
floors and execs its reviewed agent route; persistent callers must use `exec` so
they do not leave an ordinary-priority waiting shell. An inherited nice 19 is
preserved. Factory-owned workers retain their existing nice 10; the factory
coordinator is not an agent/supervisor priority exemption. No live process was
reniced or restarted by this implementation.

Private per-attempt `*.factory-process.jsonl` journals contain only operational
metadata and argv digests. Admission/refusal summaries and terminal receipts
retain first/last sightings, maximum streaks, cleared transients, trips, sensor
errors, stop reasons and telemetry hashes. Trip records are flushed/fsynced
before drain. No label values, rescue outcomes or command-line secrets enter
these journals.

## Verification and boundaries

Independent focused Codex-V: **PASS**, 69 new acceptance cases (60 process/clock,
6 coordinator preempt/resume, 3 actual launcher priority), serial commands at
nice 15, CPU-only and one numerical thread in the environment. Product and
acceptance-fixture bytes at the final operational pin match the independently
executed candidate `85f4106ba8a30bb9ff5cbd677a85e3135971f048`; subsequent commits
corrected only two migrated fixture assertions. Additional focused regressions:
147 recovery cases, 68 migrated interim cases, and 8 priority/admission cases
passed. Black, isort, flake8, shell syntax and diff whitespace checks passed.
Evidence commands, results and logs are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/factory-persistence-verification/`.
Initial fixture-migration failures are retained there and superseded by the
corrected passing records. Every completed command stayed below 120 seconds;
there was no timeout. The common lock refused one busy test submission before
execution; it was retried only after read-only lsof confirmed the lock released.

Create-only independent verdict, SHA sidecar and completion-last marker:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/CODEX_V_FACTORY_PERSISTENCE_20261023.json`.
Receipt SHA256: `50d9736a61947b335e06694b922419e646f0811458fd86564bf924cfe553938e`.
The repo contains the identical verdict triplet. The receipt records the exact
commands, source hashes, reused evidence, findings resolved before qualification,
and final operational pin.

Frozen D256 science remains clean and detached at
`61900af28932e47e22e332840fc4406559adb062` in
`/Users/josenunez/Projects/ml/snake-dqn-d256-resume-science.noindex`.
The unchanged scientific/export closure and slowdown implementation retain their
applicable prior determinism/slowdown evidence. A new opaque synthetic coordinator
check covers this patch's preempt/resume path, immutable prefix and verified-unit
skip. Worker-second/accounting expressions, cumulative charges, caps, drains,
allocation, ledger semantics, retry rights and baseline rules are unchanged.
Timing/health/slowdown/monitor faults retain their immediate stops; simultaneous
slowdown and persistence reasons are both retained. The interim numerical-test,
baseline/rearming/W12-sanity restrictions and complete 120-second launcher
deadline still apply. This work launched no factory, production host check,
calibration, timing/strict/serving run, remote runtime or long job.

## Exact future D256 continuation

**Do not stop or hot-patch the D256 attempt at `387b1dde`.** Its tmux session
`mac-bl-f0-d256-guard-w12-06-387b1dde8241` was observed and left untouched.
Switch only after its next natural preemption or completion and verified stop.
The existing continuation then verifies the prior chain and union ledger,
preserves all charges, and admits only verified missing units when its other
registered conditions permit. No automatic redispatch is authorized. A natural
completion may leave no eligible work to dispatch.

The requested `bl-f0-d256-guard-w12-07` was unused during command preparation,
but another operator consumed it at `387b1dde` while this task was finishing.
Final read-only inspection found that pinned worktree and its runner/log at
20:12 PT; this task did not launch or alter them. Reusing `-07` at the new pin
would fail the launcher's existing clean/detached/pin check. Preserve that
evidence. `bl-f0-d256-guard-w12-08` has no worktree, runner or log reservation at
closeout. The exact executable future operator command therefore uses `-08`:

```bash
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d256-guard-w12-08 caafdf5d079bd73bcfbde6f9356dc0f60de42e0f -- \
  /bin/bash research/bl_labels/RUN_D256_RESUME.sh --host-check \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json
```

The receipt and this operator record inform the owner of the prospective policy
activation; they do not claim that the running old-pin job has switched policy.
Observed tmux sessions left untouched across this work: `git-sync`,
`job-a1-prof-a100-canary-03`, `job-p1-repair10-04-repair`, `renicer`,
`mac-bl-f0-d256-guard-w12-06-387b1dde8241`, `sol-gpuflood`, `sol-persist`,
`sol-r33build`, and `watchd`. Session inventory naturally changed during the task;
this implementation performed no tmux process-control action.
