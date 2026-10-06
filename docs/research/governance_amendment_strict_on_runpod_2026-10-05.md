# Governance amendment: strict gates on RunPod serverless, 2026-10-05

Status: **proposed amendment** to the Tier-2 strict gate rules in
[governance_tiers_2026-09-26.md](governance_tiers_2026-09-26.md), as amended by
[governance_amendment_sequential_gates_2026-10-02.md](governance_amendment_sequential_gates_2026-10-02.md)
and [governance_amendment_paired_bands_2026-10-03.md](governance_amendment_paired_bands_2026-10-03.md).
It records an owner decision of 2026-10-05, taken under the final-call authority the user
delegated, after the user asked whether strict gates must run on the Mac. The ratification
section at the end is left for the owner.

**Scope.**
- **Who it applies to.** Only Tier-2 strict promotion gates whose pre-registration is written
  after this amendment is ratified, and only when that pre-registration opts in (sequential
  strict template v3, see "Implementation").
- **Never retroactive.** The v7 and v8 STRICT_PASS receipts stand as they are. No earlier or
  running gate is re-judged, re-run or relabelled under this rule.
- **What stays.** A gate that does not opt in runs exactly as today: on the Mac, 2 CPU slots,
  under the existing runner. Template v1 and v2 intents, plans, receipts and audits are
  byte-identical (pinned by golden-hash tests).
- **Serving qualification is not covered.** It stays on the Mac (condition 4).

## The rule

A strict promotion gate **may** run its episodes on RunPod serverless CPU workers when all four
conditions below hold. The run's orchestration (look boundaries, receipts, ledger, write-once
records, the independent audit, closeout) stays on the Mac in every case.

1. **The platform is named in the pre-registration.** The protocol names the execution
   platform (a line `Execution platform: runpod-serverless`), and `intent.json` freezes the
   platform, the remote configuration document (by sha256), the worker sizing, the cost cap,
   the identity-check sample and the fallback rule before anything is played.
2. **Per-world single worker.** Every arm and every control episode of one `(mix, world)` unit
   runs on one worker, in one job. A unit is never split across workers or jobs. Different
   worlds may run on different workers and CPU models.
3. **Pre-gate identity check.** Before the gate starts, a pre-registered sample of the
   candidate's own gate worlds (default: 3 worlds per mix from the first look's prefix, both
   arms, full horizon) is played on the Mac and on RunPod. Every record must be byte-identical
   after dropping wall-clock, platform and dispatch fields. Any mismatch, or a check that did
   not complete, sends the whole gate to the Mac (2 slots) under the same intent. The check
   stores digests only, so no gate outcome is seen when the backend is decided.
4. **Serving qualification stays on the Mac.** It remains the final check on the platform the
   agent is actually served on.

## Why it was local, and why it can move

Strict gates have run on the Mac for three reasons:

- **Same machine as serving.** The served agent runs on the Mac. Linux x86 workers use a
  different BLAS (MKL/oneDNN instead of Accelerate), so raw network Q-values differ by about
  5e-5 absolute (max ULP diff about 3000; [x86 consistency report](runpod_x86_consistency_2026-10-03.md)).
  That never changed a decision in our checks: 28 of 28 H5000 episodes in the x86 consistency
  check and 9 of 9 in the runner end-to-end test were bit-identical to the Mac records. Over
  thousands of episodes, though, a near-tie could flip one decision and that episode would then
  diverge.
- **The audit setup assumes it.** The strict runner's safeguards were built for the Mac:
  source re-hash before every segment, AC and lid guards, CPU slot locks, write-once outputs
  and an independent audit.
- **Comparability.** The v7 and v8 gates ran on the Mac.

It can move because:

- **The comparison stays fair.** The gate's evidence is paired per world. When both arms of a
  world run on one worker, the paired delta compares the two policies on the same platform,
  whatever CPU that worker has. This is the per-world rule the Tier-1 runner already uses
  (v9 screen amendment 1, "platform consistency per world, not per run").
- **Every RunPod check matched the Mac exactly** (28/28 and 9/9 above).
- **Speed.** A strict gate takes about 5-8 hours on 2 Mac slots (the v8 strict gate: 1170
  episodes, 34 s mean, 5.7 h to its look-2 stop). Batched serverless runs it in about an hour,
  for very roughly $5-15.

## What the identity check bounds, and what it does not

The ~5e-5 Q difference is the risk this amendment accepts. The conditions bound it as follows.

- **A flip cannot bias the paired comparison by platform.** Both arms of a world share one
  worker, so a flipped decision changes that world's trajectory on that platform, for that
  arm, exactly as a different seed would. The gate's error guarantees hold for the platform the
  gate ran on.
- **The identity check is direct evidence on the candidate's own decision paths.** It replays
  the candidate's (and incumbent's) own gate worlds, not stock worlds, so a candidate whose
  decisions sit near ties often enough to diverge is likely to show it. It is a detector, not
  a proof: with 18 episodes, a candidate that diverges on 15% of episodes passes the check with
  probability 0.85^18 = 0.054, and one that diverges on 5% passes with probability 0.40. Rare
  flips can pass it.
- **Serving qualification on the Mac is the backstop.** Whatever passes on RunPod must still
  pass serving qualification on the serving platform before it is served.
- **Duplicates must agree.** Whenever a unit is replayed on another worker (worker loss), every
  episode that came back twice must be byte-identical (deterministic bytes) or the run stops
  `INVALID_STOP`. Cross-platform divergence inside a gate is never tolerated or averaged away.

## Fallback rule (condition 3, exactly)

- The identity check runs **once** per intent, after `prepare` and before `run`. Its outputs
  are create-only, and it can never be re-run or re-rolled.
- `run` reads its result before anything is played:
  - **passed** (every sample episode byte-identical on both sides): the gate runs on RunPod;
  - **failed** (any byte difference, a missing episode on either side, or a side that could
    not run): the gate runs on the Mac, 2 slots, with the same intent, plan, banks,
    interleaving and caps;
  - **started but not completed** (crash or interrupt): treated as failed, so the Mac;
  - **not run**: `run` refuses to start.
- The fallback is recorded in `started.json`, every segment marker and the closeout, and the
  independent audit checks that the backend used is the one the identity result implies.
- No gate outcome is visible when the backend is decided: the check keeps per-episode digests
  of the canonical record bytes (and per-field digests to locate a divergence), never the
  records or their metrics.

## Cost cap and the per-step speed-up rule (4x)

- **Cap.** The pre-registration states a dollar cap for the gate and one for the identity
  check. The endpoint's worst-case reservation (workers x reserved rate x the remote wall cap)
  must fit the cap and the shared RunPod ledger's account floor and project cap.
- **Balance-delta hard stop.** While remote work runs, the account balance is read every few
  minutes. If this run's spend (the balance drop since its start, less what other runners'
  ledger reservations could have spent in that time) reaches the cap, the run cancels its
  jobs, deletes its endpoint and closes out `INVALID_STOP`. It is never relabelled.
- **Per-step speed-up rule** (the user's standing rule of 2026-10-04, 5x; lowered by the user
  to **4x** on 2026-10-05, "change rule to 4x not 5", before this amendment was ratified): the
  plan must project at least 4x faster wall-clock than the same gate on 2 Mac slots, counting
  cold starts, seeding, the identity check and look barriers. `prepare` refuses a plan below
  4x (`remote_policy.json` `min_speedup`; the code refuses a policy below 4x) unless it is
  explicitly forced; a forced plan is recorded in the intent and the audit report.

## Failures

- **Worker loss** (job failed, timed out, lost or returned no output): the whole unit is
  re-dispatched to one worker, up to a pre-registered attempt cap. The loss is recorded.
- **Episode exception on a worker** (the study's own code raised): the run stops
  `INVALID_STOP`, as a worker crash does on the Mac.
- **Returned records** are checked against the spec before they are written: intent and arm
  identity, source-closure bytes on the worker, roster row hash, pinned horizon, and a
  platform stamp shared by every episode of the unit.
- **No capacity before admission** (no worker starts, quota busy): `run` refuses, nothing is
  started and the namespace is not consumed. **After admission** any remote failure, the
  remote wall cap, battery or a closed lid is final (`INVALID_STOP`, or `INCOMPLETE` for a
  cap), exactly as on the Mac. There is no resume and no mid-run switch to the Mac.

## Effective date

Prospective: from ratification, for strict pre-registrations written after it that opt in.
Until it is ratified, no strict gate may use the remote backend in production; dry runs and
plans are allowed. A production v3 `prepare` binds this document by sha256 and refuses unless
its ratification section records the decision as ratified.

## Implementation

`research/sequential_strict_template/` template v3 (opt-in through `prepare --remote-config`),
added on branch `strict-runpod` after this document:

- `remote_backend.py`: the plan (speed-up vs 2 Mac slots, refuses below 4x unless forced),
  the remote executor (world-unit jobs, batching, re-dispatch, duplicate check, balance hard
  stop, endpoint teardown, watchdog) and the identity check;
- `remote_worker.py` and `strict_sls_handler.py`: what runs on the worker, from the frozen
  commit's git archive on the private volume;
- `sequential_runner.py`: template v3 intents, the backend choice from the identity result, the
  lid guard, per-world platform stamps;
- `sequential_audit.py`: platform named in the intent and protocol, per-world single platform,
  identity result present and consistent with the backend used, look segments that contain
  exactly the planned units.

Seeding the strict runtime on the volume, allow-listing a new candidate checkpoint and every
spending step remain owner actions.

## Disclosed limits

- The identity check detects gross platform sensitivity, not rare flips (see the numbers
  above).
- The gate's statistical guarantees are about the platform it ran on. Transfer to the Mac
  rests on the near-equality shown so far and on Mac serving qualification.
- Remote workers are slower per core (about 3.6x) and their speed varies by CPU model, so
  remote wall-clock caps are estimates. A cap reached ends the run `INCOMPLETE`.
- The balance read is account-wide. The hard stop subtracts other runners' ledger
  reservations; spending outside the runners (a hand-made pod) can trip it.

## Ratification

Pending. To be completed by the research loop owner (or the user):

- Decision: ______ (ratified / ratified with changes / rejected)
- Date: ______
- Changes, if any: ______
