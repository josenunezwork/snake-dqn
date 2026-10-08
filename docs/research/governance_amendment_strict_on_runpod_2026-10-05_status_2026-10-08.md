# Status note: strict gates on RunPod amendment (2026-10-08)

This note records the status of
[governance_amendment_strict_on_runpod_2026-10-05.md](governance_amendment_strict_on_runpod_2026-10-05.md).
It is a separate file because that document is **hash-bound**: template v3 `prepare`
(`research/sequential_strict_template/remote_backend.py`, `amendment_status`, via
`remote_policy.json` `amendment_path`) reads it, binds its sha256 into the intent and checks its
ratification section. Its bytes stay unchanged. Its "Ratification: Pending" section is
historical, and this note governs its status.

## Status

**NOT RATIFIED. Superseded on 2026-10-08** by the research loop owner, under the user's
standing authority.

- **Never used.** No strict gate ran on RunPod. The only strict gate since the amendment was
  drafted is the FRP-v3 gate (`frp3-m3s12-strict-20261005/run-v1`). It ran on the Mac with
  2 slots, because the RunPod worst case exceeded the balance
  ([result](frp3_strict_result_2026-10-06.md)). The v7 and v8 STRICT_PASS receipts were
  produced on the Mac before the amendment and are unaffected.
- **Condition 3 does not work as written.** Mac↔x86 bitwise identity checks fail at random.
  The cause is exact float32 Q-value ties, which the Mac (Accelerate) and x86 pods (MKL AVX2)
  break differently because they sum in different orders. The rate is about 1 in 50 H10000
  episodes. This is not a code bug. FRP-v5-S2's 1/54 identity failure was traced to such a tie.
  FRP-v5-H's 54/54 pass was a lucky draw. Diagnosis: branch `diag-divergence`,
  `docs/research/mac_pod_divergence_diag_2026-10-07.md`, and Claude memory
  `mac-pod-float-ties.md`. A byte-identity check on a large sample would therefore send most
  gates back to the Mac, or force a waiver that the amendment does not allow.

## What a future strict-on-pods rule needs

Any future strict gate on pods (or serverless) needs a **new** amendment, ratified before its
pre-registration, that includes at least:

1. An identity check that **explains** divergences. It must record a per-decision action-hash
   stream with the top-2 Q margin, and accept a mismatch only if the first divergence is at a
   near-exact tie (within about 4 ulp) on the Mac side. Epsilon tie-breaks alone are not
   enough. Float64 action selection is an option, but it changes behaviour on exact ties, so
   the reference records would have to be regenerated.
2. The **per-world single CPU model** rule: every arm and control of one `(mix, world)` runs
   on one worker with one CPU model.
3. Serving qualification stays on the Mac.

Until such an amendment is ratified, strict gates and serving qualification run on the Mac.
For current rules, see [STATUS.md](../STATUS.md). The campaign record is
`governance_strict_on_runpod.final_outcome` in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/campaign.json`.
