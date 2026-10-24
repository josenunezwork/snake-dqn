# Owner-directed F0 nice interim

Authority: owner 2026-10-23 request for the minimal qualified interim on
`5dfd4a74`, with no activation of `22c0db2f`'s prepared concurrency candidate.
This handoff governs the interim only. The full factory-companion candidate
remains PREPARED_NOT_QUALIFIED / activation HOLD.

`origin/main` already contained the candidate, so this integration restores
`mac_driver.py`, `slowdown_resume.py`, `slowdown_guard.py` and their existing
driver/integration tests to `5dfd4a74` before applying the two narrow guard
changes. Prepared companion modules, tools, receipts and their historical
handoff remain available but have no hooks in the active driver, host check or
median guard. No immutable prior receipt or ruling was edited.

The runtime exclusivity check now reads `pid,ppid,nice,args` through `ps` and
ignores non-owned processes at nice >= 15. The existing ChatGPT desktop-helper
exclusions remain. Non-owned research Python/pytest and Sol/Codex jobs below
nice 15 still refuse admission or preempt collection. Missing, malformed or
out-of-range nice values fail closed. The slowdown host check delegates to this
same guard, including owned-descendant exclusions. Its historical receipt key
`no_sol_codex` now means no disallowed Sol/Codex job under this interim policy;
the schema, create-only publication and all escalation rules are unchanged.

`tools/ops/agent_nice.sh` launches a command at nice 15, retaining a lower
inherited priority such as nice 19. It adjusts the relative `nice` increment
against `ps`'s current value so nested wrappers do not accumulate priority
reductions. It exports OMP/MKL/OPENBLAS/NUMBA/VECLIB thread counts of 1 and
`PYTHONDONTWRITEBYTECODE=1`. Commands and their children inherit these settings;
arguments and exit status are preserved.

All code outside `_exclusive_factory_guard` and `record_host_check` in their
respective modules matches `5dfd4a74` by AST comparison. The median slowdown
implementation, saved slowdown state, D256 continuation, frozen science
loader, W12 selection/runtime, immutable exports and D256 shell entrypoint
are byte-identical to that baseline. Thermal/RSS/disk/AC/lid guards, timing
arms, caps, accounting and registered stops are unchanged. The scientific
tree remains clean, detached, and exactly pinned at
`61900af28932e47e22e332840fc4406559adb062`; no scientific import closure was
edited or requalified through gameplay.

Independent Codex-V light review: PASS, no findings. See the new
`CODEX_V_NICE_INTERIM_20261023.json` triplet for source hashes, commands and
results. Coverage includes nice -5/0/14 rejection, nice 15/19 exemption,
codex-handoff, research Python/pytest, desktop helpers, ownership, malformed
observations, and real launcher priority/child/environment checks. No factory,
actual artifact host check, timing arm, calibration or long job was launched.

After publication, substitute the full new commit from the final handoff for
`IMPLEMENTATION_SHA` below. This command is for the operator; it was not run:

```bash
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d256-guard-w12-03 IMPLEMENTATION_SHA -- \
  /bin/bash research/bl_labels/RUN_D256_RESUME.sh --host-check \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json
```

Launch subsequent Sol work through the interim wrapper (replace only the task
text and target worktree as needed):

```bash
CODEX_MODEL=gpt-6.1-sol \
  /Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/agent_nice.sh \
  /Users/josenunez/.claude/bin/codex-handoff.sh \
  -C /Users/josenunez/Projects/ml/snake-dqn-f0 -e high 'YOUR TASK TEXT'
```

This interim does not qualify or enable the full companion supervisor, its
resource budgets, companion-first shedding or protected-window policy.
