# T34 doom fit: mechanical completion attempt 1

Prepared 2026-10-30 by Astra at the owner's explicit request. **Not dispatched.**
This is attempt 1 of at most two mechanical completions after the original fit;
it grants no automatic retry, scientific extension, scoring look or release.

Original job: `t34-doom-fit-02-a100-fallback`. Its receipt records INCOMPLETE,
`RPC_EXEC_POLL_TIMEOUT`, the exact error
`Refused('kernel remains busy/unknown after telemetry hiccup')`,
169.4378738750238 seconds, `stop_verified=true`, `cleanup_error=null`, no terminal
worker result and zero exports. The exact error was checked in addition to its
failure class. Receipt SHA-256:
`ab90f86c2fbb516db2bcb1d03bd85195bccb1c4e7c02bf5b25b703a3d73245ea`.

The source packet and new operational evidence live outside Git at:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/doom-fit-retry`.
These committed files are byte-exact copies of its preparation/verifier scripts
and sealed `packet-01` manifest, lineage, verified empty ledger and commands.
Run the commands against the external packet paths below. Preparation is
create-only and has already run; do not replay it into the used packet.

## Unchanged admission and science

New job: `t34-doom-fit-02-a100-completion-1`.
New output: `doom-fit-retry/packet-01/fit-output` under the artifact root above.
New supervisor state: `packet-01/t34-doom-fit-02-a100-completion-1-state`.
Both remain absent after offline validation. Runtime pin remains
`b27c90bd397695700895e8c539c54765054733d2`.

All five uploaded input paths, bytes and hashes, the full 230-path source
inventory, script, CUDA device, environment, A100-only requirements, genuine A100
canary receipt and admission remain identical to
`doom-admit-fix/admission-01/fit.json`. Models, data, seeds, split, N, 3,000 updates,
endpoints, bars, bootstrap, fixed Mac scoring lock and look discipline are unchanged.
The useful-worker limit is 2,700 seconds; base/effective whole-session guards are
60/120 minutes, unit guards 8/16 and reserves 2/4 at the provisional 8 u/hour rate.

The outcome-blind ledger builder found no verified units; it exactly reproduces
the original uploaded empty ledger. With no changed resume input, this successor
retains the original canary and admission. The generic `resume.completion` path
rebuilds the input inventory and requires new canaries; it was not used here.
Any future completion with verified units must use the complete original-through-
current stopped-receipt history and latest eligible snapshots. Attempt 2, if
eligible and separately requested, is the last permitted completion.

Exactly five top-level manifest keys change: `job_id`, `output_dir`,
`completion_attempt=1`, `completion_lineage`, `exec_poll_recovery_seconds=1200`.
The lineage SHA binds the original admitted manifest, admission, exact executed
manifest, mechanical receipt and preparation script. The executed manifest is
normalized only for the supervisor's default 180-second lease and dispatch's
live `budget.other_assignments` inventory; every other field must equal admission.

## Qualified recovery scope

`TRANSPORT_REQUALIFICATION_20261029.md` documents base/effective exec-poll recovery
of 600/1,200 seconds; 1,200 is the qualified maximum. The successor sets it
explicitly. **The original receipt already records 1,200 seconds.** Its trace
shows failed 30-second staging execs and the busy/unknown staging refusal.
`CLI.poll_rpc` retains separate staging recovery limits and its busy-kernel fence.
This packet makes no claim to extend staging recovery or fix that transport fault.
No qualified transport code, staging cap, lease, idle proof or mutation fence was
changed. Session/caller deadlines continue to narrow recovery. The accepted
transport qualification registry remains authoritative at execute time.

## Offline verification

- Actual `tools/ops/dispatch.sh --dry-run` passed:
  `PASS_DRY_RUN t34-doom-fit-02-a100-completion-1 allocations=0 live_accounting=NOT_CHECKED`.
- Package smoke, real receipt-layout, prelook, exec-outage and RPC-recovery suites:
  **69 passed in 64.03 seconds**. No real provider work ran.
- `VERIFY_PACKET.py` verifies all 99 captured prior packet/admission/log files
  retain their original size/hash; inputs and admission remain bound; no scoring
  lock, output, state or dispatch reservation exists.
- Black, isort, flake8, Bash syntax and whitespace checks passed.
- Argon / Gemini 4 Argon was requested from its separate review worktree with the
  complete source and packet inline. It returned HTTP 429 `RESOURCE_EXHAUSTED`,
  exit 3, without a verdict. No replacement reviewer or self-approval was used.
  Both this packet review and the prior admission-code review remain outstanding.

Manifest SHA-256:
`7f1cf697ed740910fb98f4ba4ffc18486b304a9e3b5fa87e7d11d97d3650fd83`.
Lineage SHA-256:
`3dd14c9c58bcb8e4ddd5fe366304515e19ea66137b0c682cd50d3f58406cece7`.
Preparation script SHA-256:
`7607f5c85e083ed30d06aac25aec4effe0d729ee1e006f73b6eaedbfac25d6dd`.

Exact commands and logs: external `VERIFICATION_COMMANDS.txt`, `prepare.log`,
`dry-run.log`, `pytest.log`, `INTEGRITY_CHECK.json`, `EVIDENCE_BEFORE.json`,
`review/` and `SUMMARY.md`. The offline dry-run does not check live assignments
or resolve prior H100 allocation uncertainty; execute-time broker checks remain.

## Exact live command (not executed)

/Users/josenunez/Projects/ml/snake-dqn-doom-auc-smoke/tools/ops/dispatch.sh t34-doom-fit-02-a100-completion-1 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/doom-fit-retry/packet-01/fit.json --broker /Users/josenunez/Projects/ml/snake-dqn-doom-auc-smoke --cwd /Users/josenunez/Projects/ml/snake-dqn-doom-auc-smoke --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/doom-fit-retry/packet-01/t34-doom-fit-02-a100-completion-1-state --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/doom-fit-retry/packet-01/DISPATCH.txt
