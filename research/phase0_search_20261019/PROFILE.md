# Phase 0 search teacher baseline profile

This is the pre-optimization reference captured from commit
`45069720ed116d5f6ee681b530d60c3bcd805961`. It uses the registered N=8,
H=128 search at a synthetic hero length of 800, six legal first actions, one
native thread, and a warm-up run excluded from measurement. The three timing
fixtures are the disjoint `scripted`, `frozen`, and `mixed` rows in
`WORLD_BANK.json`; the real strict roster policies are retained.

| mix | total seconds | sim | observation | B8 forward | frozen forward | scripted logic | clone | Python residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scripted | 4.1761 | 0.1373 | 1.0886 | 1.7241 | 0.0000 | 0.0088 | 0.0207 | 1.1967 |
| frozen | 4.9171 | 0.1714 | 1.2045 | 1.8350 | 0.0889 | 0.0000 | 0.0183 | 1.5991 |
| mixed | 5.9023 | 0.1820 | 1.5099 | 2.2090 | 0.0712 | 0.0160 | 0.0198 | 1.8945 |

The dominant measured work is repeated hero observation construction and B8
forward. The mixed fixture executes 7,516 B8 forwards and 3,748 ego2s raster
builds for one searched decision. `GridBatchSim.step` accounts for 129 calls
but only 0.1820 s in the mixed timing. The clone/replicate step is one call and
about 0.020 s. The residual includes Python dispatch and Vector61 runtime work;
the frozen category wraps only `Vector61SimdPolicy.q_values`, so its runtime
observation and mask preparation remain in that residual.

The exact raw values, call counts, score vectors, source hashes, and command
environment are in [PROFILE_BASELINE_RECEIPT.json](PROFILE_BASELINE_RECEIPT.json).
The score vectors were recorded to make the pre-change action-selection result
auditable; the winning action is 3 for all three fixtures.

The category timers are wrappers around the existing `BatchSim.step`,
`build_ego_raster`, `ego2s_policy.forward_q`,
`Vector61SimdPolicy.q_values`, `FastProfileAnchorSimdPolicy.actions`, and
`teacher._replicate` functions. Python residual is total elapsed time minus
those wrapped intervals, so category sums equal the measured decision wall
time apart from timer precision. This is a profiling baseline only; it does not
authorize Phase 0 selection execution, Colab/RunPod use, serving changes, or
any checkpoint promotion.
