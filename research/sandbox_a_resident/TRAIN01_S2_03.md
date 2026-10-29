# TRAIN-01 S2-03 — deterministic corruption test and fresh admission

S2-02 is **BURNED**. Its H100 assignment request returned 503 without an endpoint;
the admitted A100 fallback then refused before allocation because a compact-state
corruption test failed (107 passed, 1 failed). No training result or completion retry
was consumed. Preserve both old packets, manifests, logs and the H100 receipt.

The failure was in the test: `sample_state()` drew a float32 actor weight from the
global, unseeded Torch RNG, then the test added `1e-7`. At magnitudes above 2, that
addition can round to the original float32 value. For example, 3.0 has an upward
spacing of `2**-22`, approximately `2.3841858e-7`; the test asked the verifier to
refuse an unchanged state. Under the recorded nice15 and seven single-thread caps,
200 independent pytest processes failed 10 times (5.0%) before the fix and zero
times (0.0%) after it.

The test now uses private seeded fixture draws and `torch.nextafter` to change the
selected weight by one representable value. It asserts that the bytes changed
before requiring `CompactResumeRefused`. All original refusal assertions remain.
Boundary regressions reproduce the old no-op at 2.0, 3.0 and -3.0, check deepcopy
isolation, and require rejection of actual one-bit changes in zero, signed zero,
infinity and two NaN payloads. The production verifier is unchanged: it hashes
tensor bytes and compares dtype, shape, stride, offset, autograd and structure.
No verifier hole was exposed by this incident. Packet-15 continuation r6's recorded
runtime `dccf7e25a5fb3ebc2fe303b2dc072aeb83efd213` uses the byte-identical compact
module (SHA256 `518b0d7853ee1ae1aefb63018414613dd335db27ae6f73f369c253a5bd2be41e`).
This provenance check is not an audit of r6's training outcome.

Fresh initial identity: `sandbox-a-train01-20261028-s2-03`. Retain allocation index 2,
training seed offset 2,000,000,000, empty initial ledger and the existing byte-exact
seed audits. Neither S2-01 nor S2-02 executed training. Keep the 384-update recipe,
snapshot interval 16, compact full-state transport, proxy schedule and numerical
runtime unchanged. The runtime closure differs from S2-02 only in the compact
source-packet allowlist admitting S2-03; the continuation builder likewise admits
this family and preserves its per-attempt guard for c1/c2.

Freeze the same base/effective attempt envelope: estimate 690 minutes, timeout
1,380 minutes (23 hours), H100 138/276 units at 12 u/h provisional ceiling,
A100 92/184 units at 8 u/h, zero reserves. Price the entire timeout. The unchanged
full-workload A100 forecast remains 2,009.04 minutes; the estimate is an attempt
window, not an endpoint forecast. Continue only the latest verified eligible
full-state snapshot after verified mechanical failure/release and missing-unit
checks, with at most two fresh manual completion attempts. No automatic retries,
scientific look, release, serving change or F5 crossing is authorized here.

Evidence, the independent Argon review, exact admission replays, real wrapper
dry-run and operator command are recorded in `VERIFICATION_TRAIN01_S2_03.json` and
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-s2c/SUMMARY.md`.
