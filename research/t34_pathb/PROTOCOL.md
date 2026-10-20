# T34 Path B deadline feasibility — frozen development protocol

Owner authorization: 2026-10-19 chat, “Implement the spec
docs/research/specs/2026-10-19_t34_pathb_deadline_check.md in this worktree”.
Written before implementation smoke or efficacy data. This implements the ratified
SUCCESS_CONTRACT, not a strict gate or a serving release. No policy fit occurs.

## Controller identity and information

Only slot-zero B8 s3 (SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`)
in mixed Watch6 is eligible in the product. Flag `SNAKE_SERVE_PATHB=1`, default OFF.
Watch8/12, Play, Train, other checkpoints and refused mixed rosters remain bare.
The hero is visibly **assisted** whenever this controller is enabled, including
when every request misses. Rollback: unset the flag and pin
`SNAKE_SERVE_CHECKPOINT=b8-s3` on the next session construction.

Trigger: living hero length >=600 and target frame divisible by 5. No learned
risk threshold. Verification horizon is 40 world transitions, including the
target transition; no extension near an evaluation horizon. The pre-tick world
is captured after the preceding real tick, for target frame=current frame+1.
First frame/no submitted request uses bare B8. This staging is essential: work
gets only the real inter-tick interval. Requests never rewind an executed frame.

The worker receives an isolated serialized live GameState and current Python
and NumPy RNG states. These states are available to the running server; they
are captured before future outcomes exist. Food maintenance, respawn, actions,
growth, collisions and rewards use the existing GameState implementation.
All opponents continue with their own actual served policy and wrapper, with
their carried observation/enemy memory. The worker has its own process RNG;
it never resets or consumes the live process RNG. B8 also continues bare.
Branch clones share immutable model weights within the worker only. No replay
from an observed future, teacher labels, future food or hindsight selection.

## Complete verify/expand action rule

Reuse the T3 continuation engine with a live-world adapter (no new simulator).
First verify the bare action for 40 transitions. If it survives, keep it.
Otherwise expand all other advisory-legal actions together in lockstep, using
the real normal-only fallback mask if advisory is empty. Each continuation
uses bare B8 after its forced first action. Among full-horizon survivors choose
the highest original masked B8 Q. If no action survives, maximize transitions
survived, then original B8 Q. At every tie prefer the bare action when it is in
the tied set, otherwise lowest action index. No speed-mode preference. A
malformed/nonfinite result is a bare fallback. Proposed and executed actions
are separate fields. Runtime legal-mask/bare-action mismatch discards a result.

## Deadline, identity, resource bounds

Exactly one spawned CPU worker, one mathematical thread; one parent CPU thread.
No Colab/RunPod, services or tmux. No automatic retry. At most one request
exists, including an expired request still computing. No queued replacement.
Serialized request <=64 MiB, response <=64 KiB; recent log ring <=256 decisions.
Full harness decision logs stream to disk. Worker has bounded 6-branch storage
and 40-transition work, and checks cancellation between transitions.

Deadline=post-tick submission time + 1/fps (12 or 24). Consume nonblocking at
the target hero's action selection only. Require matching session generation,
target frame, hero life, checkpoint SHA, unchanged pre-tick world fingerprint,
bare action and action mask; both completion and consumption must precede the
deadline. Otherwise execute bare. Deadlines include serialization and transfer.
Control changes, reset, rebuild, death/respawn and checkpoint changes invalidate
the pending request. Busy, too-large, worker-error, no-request, stale, deadline,
mask/action mismatch and not-triggered fallbacks are retained. Expired work
cannot be applied to a later frame. Shutdown is explicit in harnesses and does
not terminate another process or service.

## Fixed development design and reporting

CLI fixes bank counts before running; defaults 5 independent bank blocks,
96 scripted/48 frozen/48 mixed terminal worlds per block, plus 25 continuous
mixed Watch6 worlds per block. H5000. These are development blocks of the same
fixed B8 checkpoint, not training seeds. Fresh namespace
`t34-pathb-deadline-20261019/v1`; smoke uses a disjoint smoke namespace. Audit
all allocated seeds against available research/docs/scripts registrations and
historical FRP exclusions before gameplay; reject the entire allocation on any
collision, never silently replace seeds. Create-only artifacts bind protocol,
source commit, input checkpoint hashes, ordered rosters, bank and CLI settings.
One analysis after the complete registered bank; no early efficacy peeks,
extensions, tuning or automatic reruns. Incomplete/capped/error => INCOMPLETE.

Each world runs paired bare, asynchronous-real-deadline, synchronous-unlimited
arms at both 24 and 12 fps. Unlimited is descriptive and never substitutes for
the deadline arm. World RNG starts identically; opponents remain fixed by mix.
Terminal hero: no respawn credit, dead frames contribute zero through H5000.
Mass/length are logical length. Record mass integral, alive fraction, no-death
survival, boost decisions/share. Per block equal-weight the three mix effects;
report one-sided Student-t LB90 across blocks. Skill clauses: pooled gain >=40,
LB90>0; scripted no-death gain >=.03; survival band v2: pooled no-death
noninferiority -.05, each mix floor -.075, absolute each mix >=.30; also report
alive-fraction guards -.03 per mix and -.15 scripted per block, and absolute
no-death >=.50 per mix. Boost share [.005,.10]. Missing uncertainty at n<2 is
NOT_ESTIMABLE and cannot pass. No fits or training tripwires are claimed.

Continuous mixed Watch6: paired deaths/1000 actual hero decisions (negative
95% Student-t CI upper bound, target >=20% relative reduction), paired
full-horizon relative hero mass lower95 >=-.05; report median/p95 length,
deaths at length>=800 and late deaths at frame>=2500, including their overlap.
Arena mass/kills and label accuracy are diagnostic only. Analysis uses paired
worlds within blocks and independent block means for confidence intervals.

Timing reuses T15's actual step+snapshot timer, ON/OFF paired worlds at 12/24.
Pacing matches the real engine: process then sleep 1/fps, without compensating
away work. Include update and full-tick p95/p99, achieved cadence, zero-overrun
checks, separate natural L>=600 tails, and worker-contention tails. Report
unobserved tails as NOT_MEASURED. Ceiling 24/12 unchanged; update p95<=11.688ms,
cadence deterioration<=5%, full-tick latency p95 deterioration<=10% is a local
proxy for client latency. Browser/network client latency is NOT_MEASURED by
this Mac-only in-process harness and requires later serving qualification.

Smoke is plumbing evidence only, with short horizons and tiny counts. Real
bank sizing and execution belong to the owner. A development pass authorizes
only fresh Mac strict-gate preparation, never release. Acceptance is incomplete
until every skill/product/timing clause (including client latency) is measured.
