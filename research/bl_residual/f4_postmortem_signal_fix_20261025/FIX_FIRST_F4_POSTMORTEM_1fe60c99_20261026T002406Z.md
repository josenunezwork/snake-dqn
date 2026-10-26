# FIX-FIRST: F4 post-mortem exporter re-review

Astra independent review. Execution pin 1fe60c9982f99fc4b89b69f70e98f84faa53afe8; handoff 03bf2f40f5362b879193b27f83f84d98356ed201. Worktree: /Users/josenunez/Projects/ml/snake-dqn-f4pm, branch sol-f4-postmortem-fix-20261025, clean.

**FIX-FIRST. Both prior blockers are repaired, but the new supervisor has a process-ownership race. No operator command is approved by this record. F4 remains STOP_FUNDING.**

## Remaining blocker

P2 — terminal interruption can signal an already-reaped PID. In research/bl_residual/f4_postmortem_watchdog.py:130-132, waitpid(pid, WNOHANG) can reap the child before reaped=True is assigned. SIGTERM/SIGHUP handlers or default SIGINT can raise in that interval. The finally block at :145-149 sees reaped=False and calls kill_and_record, which sends SIGKILL at :97 to a PID the supervisor no longer owns. Ordinarily this produces ESRCH; if the PID has been recycled it can signal an unrelated process. The claim that an unreaped child prevents reuse (:34-35) does not cover this interval.

Confirmed with an in-memory, source-derived function probe: AST-extracted supervise; all OS, signal, socket and path operations replaced by fakes; trace injected KeyboardInterrupt immediately before line 132 after successful synthetic reap. Observed call sequence: waitpid(424242, WNOHANG) -> kill(424242, SIGKILL) -> waitpid(424242, 0), ending in ChildProcessError. This proves the stale-PID signal attempt, not an observed real PID reuse or real collateral kill. No real process was created or signalled; no exporter imported or executed and no file written by the probe.

Repair by making successful reap and ownership-state update indivisible with respect to SIGINT/SIGTERM/SIGHUP, for example masking these signals before waitpid and restoring the mask only after reaped is updated. Ensure exceptional cleanup never signals a PID after successful reap. Add a synthetic interruption-at-reap regression proving no subsequent kill call, while retaining native-stall/finalization and durable-evidence tests. Republish the repaired pin before re-review.

## Verified repairs and preserved scope

- Publication: f4_postmortem_result.py:247-287 requires exact local sealed file inventory and returned outcomes/hashes/sizes against the origin-proved binding before outcome decoding. Each cached byte snapshot is checked directly against origin bytes, hash and length; decoding at :292/:331 uses those snapshots. Test source covers both extra-outcome directions and concurrent replacement.
- Deadline: watchdog.py:24-152 forks a separate parent supervisor, defaults to 3600 seconds, starts its monotonic timer before ACK, and SIGKILLs before disk IO. f4_postmortem.py:168-189 requires ACK before claim creation. Supervision lasts through worker exports and context cleanup. Kill evidence uses create-only exports with file and directory fsync; the spent claim persists. A sealed diagnostic plus kill gets an INCOMPLETE failure; no retry path was added. The remaining issue above is the ownership transition, not reliance on native-call Python signal callbacks.
- All 19 manifest implementation/test SHA256 entries match local files and execution-pin blobs. All 23 package files match their designated commit (execution metadata at handoff, others at execution pin); six present package targets and their checksum seals verified. Detached identity seals match the original package; no payload targets were opened or hashed.
- Diff 7f3c0857..1fe60c99 contains only the two repair implementations, associated tests and sibling package. Original package, fit.py, heldout.py, counts module and run script are byte-unchanged. Original ruling and development-parent ledger are identical. Manifest input paths/hashes/sizes, scientific schema, bins, counts, ancestry exclusions, once-only scope and original runtime ceiling are unchanged; added metadata describes repairs, supervisor and F5 prerequisite. Handoff adds only execution-pin metadata and sidecars.
- git ls-remote origin refs/heads/main returned 03bf2f40f5362b879193b27f83f84d98356ed201; execution pin is its ancestor. git diff --check passed. Sol's 145 combined and 59 focused passes are reported evidence; suites were not rerun. Synthetic regression source was reviewed. A second read-only reviewer independently identified the ownership race.

## Mandatory sequencing and boundaries

After repair acceptance, invocation still requires F5 registration ADOPTED and published before diagnostic exposure; adoption is an operator prerequisite, not enforced by the exporter or established by this review. Result publication must precede outcome inspection. No retry, F5 execution, funding change or release is authorized.

No payloads, D4096 or t34-round0-study-01/ opened. No exporter execution, commits, edits, dispatch or tmux. Only this create-only record and its .sha256 sidecar are written. Review used source/diffs, named metadata, hash/byte comparisons and one entirely mocked in-memory control-flow probe.
