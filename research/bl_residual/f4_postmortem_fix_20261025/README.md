# F4 once-only harm post-mortem — FIX-FIRST package

Ratified scope: the sibling Astra ruling, SHA256
`8c4480a5e80f63ec500abd7baa6062adc757866937ce560291cbbaab68b5eeec`.
F4 remains **STOP_FUNDING**. This package produces development diagnostics only.
Implementation and synthetic verification are published before any real model or
table access. The implementation session does not dispatch the real run.

The manifest fixes eleven named input aliases (one shared NPZ), the original
immutable R-A claim, published fit/census ledgers, and the sole output directory.
The exporter uses funding.py's D256-then-D1024 call order, heldout.evaluate's
candidate/census/ancestry checks and rowwise CPU forward, and an optional observer
of fit.heldout_metrics' existing masks. It reads only the existing H from other
heldout results. No funding function, bootstrap, fit, gameplay, new harm definition,
alternative margin, threshold, confidence interval, or significance test runs.

Only natural-stream count aggregates escape the process: total and four fixed
margin-bin counters, and the complete sorted table of all 29 parents including
zeros. Unknown outcomes remain unknown; H remains C+U. No ratios are exported.
A zero denominator is undefined if a reader later calculates a descriptive share.
A retained D1024 harm-screen disagreement is an INVALID diagnostic and never a
new funding decision.

`development-parents.json` permanently burns all 29 original parents and every
descendant/unexposed sibling for successor validation, admission, funding, model
selection, strict gates and serving qualification. Future collision audits must
resolve original ancestry; the helper
`f4_postmortem_guard.require_independent_ancestry` fails closed on exposed or
unresolved roots. Renaming namespaces or indices does not restore independence.
This package does not launch or authorize any successor study.

The copied B8 and approval JSON have no adjacent producer seals. The package
therefore publishes detached identity SHA sidecars and completion markers in
`input-seals/`. Their exact paths are fixed in the manifest. These mark completion
of frozen identity metadata, not an assertion about original producer completion.
The runtime checks those seals and exact raw file size/hash before deserialization.
All other payloads and the original claim require adjacent SHA/completion markers.
Supervisor receipts lack adjacent markers; their exact bytes are sealed by the
published fit-binding ledgers. They are provenance only and expand no payload set.

## FIX-FIRST changes and prerequisite

This sibling package supersedes the execution package at
`7f3c0857dda8df0e59ad296aa1d51c859dffb68c`. The original package and ruling bytes
remain immutable. Astra's complete FIX-FIRST review is sealed alongside this file.
Only its two blockers are repaired; input allowlists, D256/D1024 order, counts,
bins, ancestry exclusions, funding status, ceiling and once-only scope stay fixed.

**Do not invoke the run command until the F5 registration is ADOPTED and published
before diagnostic exposure.** A draft, this implementation, or a code review is
not adoption. This package does not enforce F5 adoption in executable code; the
operator must verify that prerequisite. F4 remains STOP_FUNDING. Implementation
and publication do not authorize an exporter run in this session.

The result verifier requires exact local sealed inventory, outcomes, hashes and
sizes to match the binding proved from origin/main. Every bound component and
sidecar must match origin/main before outcome decoding. Outcome validation uses
the proved bytes in memory, so later local file changes cannot expose another
outcome.

The stdlib supervisor forks one worker before native imports. The worker acquires
the existing lease/wake/host admission, then waits for the parent's deadline ACK
before entering its soft guard and writing the claim. The parent independently
SIGKILLs its exact owned worker at 3600 seconds, including native calls, terminal
exports and context cleanup. The 3570-second signal remains a soft compute guard;
no Python signal callback is responsible for the hard cutoff. There is one
inference worker, one idle supervisor, and the existing caffeinate helper.

On a kill, the parent first kills/reaps the worker, then durably seals
`postmortem-harm-20261025-01.watchdog-kill.json` beside the bundle. That receipt is
operational evidence with no outcome values. The spent bundle and claim remain.
After a fully sealed claim, the parent appends the existing-schema INCOMPLETE
failure receipt and static parent ledger without opening diagnostic bytes. A
presealed diagnostic plus finalization kill therefore verifies INCOMPLETE. The
failure records observed elapsed time without clipping; only the
`IndependentWatchdogKill` failure may record time beyond 3600 due to supervisor
scheduling/reaping. No successful diagnostic may exceed the ceiling. Partial
immutable files remain partial and must fail sealing; no repair/retry is allowed.
A kill before claim completion may leave only the spent partial bundle plus the
sibling kill seal. Preserve both; a missing terminal result never authorizes retry.

## Operator sequence

Replace `IMPLEMENTATION_PIN` below with the full published implementation commit
supplied in the handoff. Run the single dispatch command exactly once:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f4pm
tools/ops/dispatch.sh --mac f4-postmortem-harm-20261025-01 IMPLEMENTATION_PIN -- \
  bash research/bl_residual/RUN_F4_POSTMORTEM_20261025.sh IMPLEMENTATION_PIN
```

The dispatcher creates a detached pinned worktree and an owned tmux session.
The exporter freshly fetches main and verifies local source/ruling/manifest/test
evidence/exclusion bytes. It waits for the shared exclusive Mac lease; it never
preempts another job. AC, open lid, thermal state, wake lock, nice 15, CPU and all
six thread limits are checked. One inference process and one math thread run.
The 30-minute base / 60-minute effective claim-to-terminal ceiling includes
finalization. Only elapsed-time, count/progress and hash telemetry is printed.

The fixed output directory itself is an atomic attempt fence. `plan.json` is
published before `claim.json`, and the claim is sealed before any permitted real
input is hashed/opened. Every output has a SHA sidecar and completion marker
written last. A crash during plan/claim creation spends the directory too. Never
delete or replace this directory, make a second claim, or retry after any failure.
The original R-A claim is verified and never changed.

After the terminal bundle is sealed, run these from the implementation worktree
with the same one-thread/nice-15 environment. Do not cat, preview, or inspect the
diagnostic first. `verify-seal` hashes outcomes without deserializing them.

```bash
./venv/bin/python -m research.bl_residual.f4_postmortem_result verify-seal \
  --publication-commit IMPLEMENTATION_PIN \
  --bundle /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/postmortem-harm-20261025-01
./venv/bin/python -m research.bl_residual.f4_postmortem_result publish \
  --publication-commit IMPLEMENTATION_PIN \
  --bundle /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/postmortem-harm-20261025-01
./venv/bin/python -m research.bl_residual.f4_postmortem_result verify \
  --publication-commit IMPLEMENTATION_PIN
```

The publisher creates an isolated result worktree, copies sealed bytes without
outcome parsing, commits the aggregate bundle and result binding, and pushes main.
The semantic verifier first proves byte equality against freshly fetched main,
then checks exact aggregate schema, provenance and arithmetic. Only after this
publication may humans/agents inspect the immutable aggregate outcomes. No
verification command recomputes any model/table result. If publication itself
fails, preserve its worktree and resolve publication operationally; the diagnostic
run remains spent. A missing/partial terminal file never authorizes recomputation.

If the publisher committed the bundle but a Git fetch/merge/push failed, repair
publication only in its retained result worktree. Do not rerun the exporter or
change result files. These commands reveal no outcome values:

```bash
git -C /Users/josenunez/Projects/ml/snake-dqn-f4pm-result-20261025.noindex fetch origin
git -C /Users/josenunez/Projects/ml/snake-dqn-f4pm-result-20261025.noindex merge --no-edit origin/main
git -C /Users/josenunez/Projects/ml/snake-dqn-f4pm-result-20261025.noindex push origin \
  HEAD:main HEAD:refs/heads/f4-postmortem-result-20261025
```

An unrelated merge conflict requires operational resolution before publication;
it never authorizes opening diagnostics early or recomputing a missing output.
