# P successor implementation handoff — 2026-10-29

**IMPLEMENTED / OFFLINE VERIFIED / DISPATCH NOT RUN. NOT READY FOR REGISTRATION.**
Code pin: `9106e15caccdd1c365a421be26b5a4e60993926f` on
`astra-argon-falsifier-p-20261028`. Baseline was `2ff8671e`.
The original assessment, draft registration, scientific engine/live/protocol and
all existing hash-bound evidence remain unchanged.

## Implementation and independent review

`units.py` binds explicit successor allocations, source inventory, exact checkpoints,
canonical index and host/software identity. It publishes whole three-arm sessions
with public construction metadata and completion marker last. Ledgers hash payload
bytes and parse only public metadata. The Mac reducer verifies the complete unique
N32 union before opening outcomes; rows enter the fixed-seed bootstrap in canonical
allocation-index order independently of shard/attempt directory names. Missing N
never takes a partial look. Output-sidecar/symlink and single-thread host checks
precede the look. Statistics/thresholds remain the original frozen P definitions.

`runtime.py` runs all three arms per session in one worker, then takes the next
FIFO session. Workers publish directly; the coordinator polls immutable markers
and rechecks after child-only cleanup. No large outcome pickle crosses a result
queue. Parent stop flags do not interrupt publication; SIGUSR1, task queue feeder
cleanup, deadline retention, exact skips and serial/parallel custody are tested.
Scientific execution deliberately remains closed pending later pre-data admission.

Argon independently reviewed the first implementation in its own detached worktree
using Gemini4 Argon High through Antigravity. Its two high findings were fixed:
canonical bootstrap row order, and a real completion-packet sealer that gives each
changed ledger its own exact-input native family probe. Smaller findings were also
fixed: bound shard indices, wired transfer-margin binder, startup wall clocks,
stop-race guard, queue cancel_join_thread, Mac host and output checks. A nonconstant
multi-directory reducer fixture reproduces the ordering bug and verifies the fix.

**Post-fix Argon re-review is PENDING.** The requested follow-up returned
`RESOURCE_EXHAUSTED` / HTTP429, reporting reset in about95.5h. No alternative
implementer reviewed Astra's build and no review approval is claimed. Logs:
`astra-impl-p-successor/argon-review-02.log` and `argon-final-review.log` under the
artifact root. The first review's narrative misquoted a smoke seed/path count;
our retained causal report is authoritative: seed16844277847086931686, exactly11
canonical differing leaves at each of frames0..4. This correction was supplied in
the follow-up request before it hit quota.

## Root cause, delivery and timing packet

`state_diff.py` varies only uploaded checkpoint directories with identical verified
B8/FRP3 bytes and one frozen excluded engineering seed. Four event steps agree;
11 canonical checkpoint path fields differ at every observed boundary, and copies
of diagnostic inputs agree after replacing only that directory string. Stored raw
hashes are retained. This proves path sensitivity; unavailable historical field
states prevent certifying that every old discrepancy was exclusively path-related.
`delivery.stage` uses stable verified target paths prospectively. Three actual
short gameplay copies (serial plus two spawned workers) agree byte-for-byte on the
same Mac. This is not cross-architecture or full-session target qualification.

The packet stages and hashes both exact checkpoint files locally before sealing.
B8 uses the existing public SHA/size-bound HTTPS release route; FRP3 remains a private
supervisor upload. The Mac B8 preflight verified6562005bytes and the exact B8 hash
in1.115589s. That is availability evidence, not remote transfer-speed proof.
`bind_timing.py` requires a matching approved final transport tree and native family
receipt and enforces transfer<=105s (45s margin below unchanged150s) and staging<=300s.
`successor-startup.json` records post-bootstrap script and checkpoint-stage clocks;
use broker/runtime bootstrap clocks to price VM HTTPS fetch and environment setup,
which happen before the script and are not included in Contents upload seconds.

Artifact packet: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-p-successor/packet-03`.
Its `DISPATCH-NOT-RUN.txt` is the operator handoff. It contains two templates:

- Native30min family transport probe: same entrypoint/full inputs, three excluded
  ticks per arm; no whole-session resume unit is fabricated.
- Full natural-start H6000 serial L4-CPU timing: two frozen excluded sessions ×
  P0/P48/bare,12h effective guard (6h base),42540s work deadline. This is a runaway
  envelope, not a measured completion prediction. Record inclusive arm/session and
  publication work, constructor/final digest, chunks, actual root lengths/frames/
  search support, and CPU/software identity. No unsupported uplift is set to1.

The producer `astra-impl-colab-transport` has not published its final commit/tree.
The packet records the exact current source transport tree and a NULL target tree,
with `REPIN_REQUIRED_AFTER_TRANSPORT_LANDS`. It is intentionally not represented as
pinned to an unavailable future tree. Once the producer publishes, integrate that
transport, commit/push the new runtime source, and regenerate a FRESH packet using
`prepare_session --pin NEW_RUNTIME_SHA --transport-pin FINAL_TRANSPORT_SHA`.
That option refuses a different runtime/producer transport subtree. Re-bind source
inventory, checkpoints, bank, ledger, qualification record, manifests and seal.
Do not reuse prior native proof across source changes.

Manual completions use `prepare_session --attempt 1|2 --ledger MAC_VERIFIED_LEDGER
--prior VERIFIED_MECHANICAL_STOP_RECORD`; no auto retry is launched. Each changed
ledger gets a fresh30min native probe before the long attempt; the exact native
input SHA/size gate is preserved. Originals+two completion guards total36runtime-h,
plus three native probes1.5h =37.5h engineering series allowance; at the historical
1.54u/h this is57.75u informational whole-timeout exposure, not a unit budget.
This supersedes the earlier intent's series accounting which omitted fresh probes.

## Verification and remaining registration work

- Final successor package:62tests passed,13.90s (`final-review-fixes-suite.log`).
- First broader selection:147passed/2priority-launch failures. This tool shell
  already inherited nice15; adding nice15 saturated it to19. Both unchanged
  predecessor failures passed on corrected inherited nice15 (69.89s).
- Black, isort, flake8 on all eight new Python modules and Bash syntax passed.
- Real current-code Bash entrypoint passed locally on the isolated smoke bank;
  actual short serial/spawned gameplay parity and path causality reports retained.
- Packet03 seal/native validation passed offline; long validation correctly refuses
  missing family receipt. The ops dry run refuses the old unqualified transport
  tree before any allocation. No account allocation or live account check occurred.

Registration still needs the final reviewed transport and a fresh source pin;
Argon post-fix review; operator-run native delivery/stop proof; full searched H6000
target costs with supported tails; sustained concurrency and exact shard scheduling;
complete historical scientific seed audit, disjoint shard allocation, all original/
completion/probe envelopes under24h per assignment and account cap<=3; prospective
scientific admission/source freeze and independent Fable ratification. Fixed N32,
P48-only primary inference and owner-only serving/release authority are preserved.

Calendar: technical measurement/review work needs roughly1–2calendar days after
transport and Argon access are available, with additional time if frozen fixtures
lack long-body/late search support. With the observed95.5h Argon quota reset, plan
around Nov4–5 for possible registration readiness, conditional on transport, review,
operator execution and adequate timing evidence. This is not a completion forecast
for the unmeasured sessions, a registration promise, or dispatch authorization.
