# F5 W round08 transport diagnosis (offline proposal)

The plan job stopped after a successful 0.533-second source PUT: a 30-second
verify exec timeout was followed by a 30-second idle-helper timeout. Recovery
had only 60 seconds total, and helper budget exhaustion lost its transient
timeout classification. The timing job's 63.439-second whole-file failure was
a foreground heartbeat timeout, not a failed checkpoint PUT. Its preceding
PUT completed in 23.798 seconds. Successful attempt 04 includes a 45.108-second
PUT and 37.022-second background heartbeat, so 30 seconds is a demonstrated
fragile boundary for foreground heartbeat transport.

The create-only receipt analysis and report live at:

`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/TRANSPORT_DIAG_20261026/`

The inventory contains 178 allocation receipts and 25,265 CLI calls, with a
separate recent cohort, operation-tagged cohorts, timeout-censoring notes,
source hashes, and operation overlap observations. Mixed revisions and
acceptance fault runs must not be interpreted as a provider failure rate.

## Proposed change

- Keep staging exec attempts at 30 seconds; extend their shared recovery
  allowance from 60 to 120 seconds, clipped to all existing real deadlines.
- Preserve read-helper exhaustion as a typed timeout. This permits another
  idle probe within the outer recovery budget; it never proves idle itself.
- Give foreground staging Contents heartbeats 60 seconds, matching the
  successful run's background allowance. This bypasses only the artificial
  30-second helper clip. It does not add heartbeat retries or bypass staging,
  session, or priced deadlines.
- Leave existing three-attempt transient immutable Contents retries and
  120-second chunk allowances unchanged. The observed data PUTs succeeded.

The 150-second verified-transfer acceptance bar and 300-second staging cap
remain unchanged. Kernel idle must still be proved before exec replay;
assembly remains UUID-fenced and requires a durable acknowledgement.
Authentication, hashes, kernel errors, allocation, job retries, credentials,
watchdog generation fencing, and stop verification retain their existing rules.

## Qualification boundary

This branch is an offline repair proposal. Do not dispatch its transport or
copy its hashes into the qualification registry. The external
`snake-dqn-artifacts/ops/colab-cli-qualified.json` records tree
`c148431277f4e33c3f589251df892ea158407e25` at commit
`de5bde2d7866eb6cd01060f7c2fe4ed8d1253759`; the proposed executable bytes differ.
An operator-run source-bound final-only acceptance, including the applicable
unchanged staging/transfer bars and watchdog/stop checks, is required before
operational use. No Colab call or dispatch is part of this diagnosis.

Later reruns can pass, as attempt 04 demonstrates, but the heterogeneous
receipt sample does not support a calibrated pass probability. Prefer repair
review and requalification before another operational attempt. Repeated
transport stalls can still exceed the fixed bars and fail safely. Heartbeat
overlap is observable; these receipts cannot prove contention causes latency.
