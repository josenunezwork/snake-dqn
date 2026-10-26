# F4 post-mortem supervisor signal-safety repair

Package publication only. F4 remains **STOP_FUNDING**. The exporter was not run,
no real claim was created, and no payloads, D4096 or t34-round0-study-01/ were opened.

This sibling package supersedes execution pin
`1fe60c9982f99fc4b89b69f70e98f84faa53afe8` (handoff
`03bf2f40f5362b879193b27f83f84d98356ed201`). Astra's complete FIX-FIRST review and
checksum are sealed here. The original packages, ruling, development-parent
ledger and detached input identity seals are byte-preserved.

The only supervisor behavior change masks SIGINT, SIGTERM and SIGHUP around
`waitpid(pid, WNOHANG)` and the successful `reaped = True` ownership transition.
The previous mask is restored in `finally`, after ownership is cleared. A pending
terminal signal can then interrupt cleanup without signalling a reaped/recycled
PID. A nonterminal wait leaves ownership intact and existing cleanup still kills
and records its owned child. The hard ceiling, native-stall enforcement,
finalization supervision, kill-before-disk ordering, durable evidence and no-retry
rules are unchanged. `f4_postmortem_guard.PACKAGE` selects this sibling's manifest;
only package paths, implementation/test hashes and repair review/pin metadata in
that manifest differ from the previous package. Scientific schemas, counts,
bins, input identities, output path, runtime limits and ancestry rules are fixed.

Six fully mocked cases inject a pending terminal signal after the kernel has
reaped but before waitpid returns to Python, for SIGINT/SIGTERM/SIGHUP and zero
or nonzero child status. Each must restore the inherited mask and handlers, close
sockets, and make zero kill/evidence calls after successful reap. These six cases
fail against the old source with a synthetic SIGKILL(424242) attempt. Existing
native-stall, finalization and durable evidence tests remain in the suite.
See `synthetic-verification.json` for exact commands, counts and limits and
`execution-pin.json` for the full published execution pin and operator command.

**Operator prerequisite:** F5 registration must be ADOPTED and published before
diagnostic exposure; verify this before invoking anything. This package/review is
not adoption or authorization to retry. Result publication must precede outcome
inspection. No F5 execution, funding change or serving/release is authorized.

After repair acceptance and that prerequisite, dispatch exactly once using the
full pin in `execution-pin.json`:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f4pm
tools/ops/dispatch.sh --mac f4-postmortem-harm-20261025-01 IMPLEMENTATION_PIN -- \
  bash research/bl_residual/RUN_F4_POSTMORTEM_20261025.sh IMPLEMENTATION_PIN
```

The unchanged launcher/dispatcher enforces the pinned detached worktree and
existing Mac admission and resource rules. Preserve the sole spent bundle and
kill evidence on any failure; never remove/recreate the claim or retry.
The complete existing post-terminal seal/publish/verify sequence and publication
recovery commands remain in `../f4_postmortem_fix_20261025/README.md`; substitute
this package's new execution pin throughout, publish before outcome inspection,
and use the same nice-15 / one-thread environment. Those commands were not run
in this repair session.
