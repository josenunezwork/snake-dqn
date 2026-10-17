# T12 preflight record

No non-smoke qualification episode was run before this package's protocol, producer and
auditor were committed. The released checkpoint default remains frp3-s12.

The passing freshness authority is **SEED_AUDIT_T12_V2.json**, SHA256
`798402fbe40986047d2a69e5d45547b55c412ae2bb964c2d66d4e923d4de5015`.
`SEED_AUDIT_T12.json` is retained byte-for-byte as a preliminary scan with incomplete
hidden/ignored-file coverage; its PASS field does not authorize the run. V2 adds the
T7-compatible `--hidden --no-ignore` scan, covers 410 partitions, and finds no collision
for the same 73 prospective seeds. This corrected coverage before any gameplay outcome.

Preflight review also repaired the pin tool to recompute T7's result from all 5,805
hash-bound records, corrected fallback selection and attribution, and made ego2s Train
refusal return Play to Watch with the required message. These were implementation and
smoke corrections before the qualification freeze, not retries of qualification evidence.

Short parity smoke v1 failed because its SIMD observer hook was not activated; v2 and
v3 passed after enabling the observer and retaining trace arrays. The failure and later
results remain under the artifact root's `preflight/` directory. Episode and measurement
smokes exercised the real registry-selected B8 and stayed at or below 100 frames/probe.
No latency qualification or full serving episode occurred in these smokes.

Raw test logs, smoke records, pin-fill output and the validation manifest are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t12-b8-serving-20261017/preflight/`.
The real intent binds the final passing validation logs and exact source/config bytes.
