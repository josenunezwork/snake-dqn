# A-2 reseal-03: staggered initial time-limit coverage

This successor corrects the schedule rejected in Argon's 2026-10-26 review at
`snake-dqn-artifacts/big-lever/exp/argon-20261026/review-sbx-a2/answer.md`.
It supersedes the schedule, validity, and current-packet statements in `A2.md`;
all earlier packets and verification files remain historical evidence.

The fixed matrix remains E128/E1024, H128, limits 64/512, one excluded warmup
and three measured cycles, with cached/collection/integrated phases. The new
packet binds `recipe_parameters.stagger_initial_age=true`. Immediately after
initial reset, world e receives age `floor(e * episode_limit / E)`. This fixed
setup schedule spreads the first boundaries across the limit period, including
the three measured horizons of the 512-limit cells. It consumes no RNG or extra
roots. Initial age is an artificial time-limit phase, not a claim that the
spawned state has evolved for that many ticks.

Staggering runs once during setup, before the excluded warmup. Ordinary death
and truncation resets still set age to zero. No state/age is injected between
measured cycles, no survival is forced, and no outcomes select a schedule or
retry. The finite endpoint, seed bounds, GAE and deferred-bank capacity are
unchanged. This is a corrected timing workload, not a byte-identical trajectory
or a measured A100 speedup claim.

Every measured production collection/integrated cycle must contain at least
one live truncation. Terminal/truncation counts accumulate on device and are
copied to the host only after all timed cycles. If mortality removes a cohort,
the cell still exports its immutable checkpoint/timing pair with `status=INVALID`
and the offending cycle indices. Remaining cells continue. The run summary
propagates INVALID, and aggregation refuses an invalid cell or summary as
qualified throughput. A zero process exit means the finite sweep/export finished;
it does not override the result's validity. Other collection/numerical faults
and runaway limits retain their existing fail-closed behavior.

The regression runs the actual H128 / warmup1 / cycles3 shape with E4 and both
limits, `tiny=False`, real full-arena DeviceSim masks/dynamics/reset handling,
deferred bootstrap, GAE, integrated PPO and immutable exports. Only its actor
and expensive observation features are replaced with cheap shape-correct
tensors to retain the existing 60-second dispatch admission bound. A nonzero
critic exercises bootstrap values. Independent existing observation oracles
remain enabled. Separate real published-B8/full-observation/full-arena E8
checks cover both limits without those test substitutions. Their measured
cycle counts and source hashes are in `VERIFICATION_A2_RESEAL03.json`.

The dispatchable packet is `packet-a2/reseal-03`; the published runtime pin,
committed ten-line manifest and exact operator commands are in its seal and
`DISPATCH.txt`. No Colab allocation is performed by preparation or dry-run.
The existing A-2 attempt-01 output/state identity is still fresh, since no
rejected seal was allocated. The four-hour/eight-hour technical runaway guard,
audited runtime closure and qualified transport tree remain unchanged.
