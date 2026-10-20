# Path B v2 build details — prospective, before efficacy

This sibling records implementation choices under Amendment 2, without changing
the historical M2/M3 results or any deciding threshold. It supplements
`PROTOCOL_V2.md`; the operator's full retention pre-registration remains required.

- Search batches at most eight chronological prepared branch points, each with
  its allowed first actions, in one native E=(W*6) simulator. Prepared snapshots
  are reused throughout that window; action/active-row scratch is thread-private
  and persistent. Results are inspected chronologically, with the current live
  frame and controller epoch checked again immediately before publication.
  Expired candidates are discarded; a request can continue to later future
  candidates. Missing native batch support is an invalid request, not a serial
  search fallback.
- The descriptive unpaced arm holds its simulation clock **between ticks** until
  the same worker finishes its pending maintenance/search, then takes the next
  ordinary action. There is no rewind, extra request, policy change or wall-clock
  pacing sleep. These waits are cap-aware, outside the delivered update/timing
  path, and never occur in the paced arm. This makes it a descriptive reference
  for deadline loss rather than running simulated frames against a slower worker.
- The actual reused T3 dense bank has **64 source trajectories, 32 physical
  seeds**: `(b8_s0, env0..31)` and `(b8_s3, env0..31)`. The development pre-check
  preserves each original bare checkpoint and roster, binding original ceiling
  receipt and checkpoint bytes. It reports survivors/64 and the two source arms;
  it makes no independent-world inference. It never fabricates 64 fresh seeds or
  substitutes the unrelated Part-D mixed label-bank namespace. Its original
  namespace is `t3-dense-ceiling/v1`. The fresh registered retention test remains
  bare B8 s3 versus the frozen B8 s3 successor, with 32 independent worlds/mix.
- Terminal heroes stop after first death; all remaining horizon frames score
  zero without further simulation or pacing. Continuous Watch sessions continue
  to respawn. Diagnostic large deaths use pre-action length >=800; late deaths
  use frame >=2000. Their overlap is reported and does not introduce a new gate.
- Non-finite neural Q is an unconditional fault throughout source, shadow and
  branch dispatch. Ordinary native assembly/identity errors are incomplete
  evidence. The 100k-decision action-collapse window counts actual delivered
  decisions across sessions; dead frames never enter that denominator.
- Watch uses the pinned deployment parity profile from the accepted M1 identity
  harness and the real T15 GameSession, with an audited Grid mirror. Both arms
  include bridge overhead in full-tick measurements. Its profile must be frozen
  explicitly in the registration. These probes do not qualify another profile,
  client network latency, Watch8/12, Play, or a production serving release.

All values above are fixed before any v2 efficacy bank is run. Implementation
smoke observations are development-only and cannot decide Claim B.
