# F5 W attempt-04 verification

**PASS — environment capture and disjoint CUDA transport only.** Evidence was published first in `969efe7cf782af049612055cb8f7150d23c7ee52`, then verified against origin/main and recovered byte-for-byte before this report was created.

| Check | Verified result |
|---|---|
| v5 environment admitted | Yes; no acquisition, validation or admission refusals |
| Twelve fixture parity checks | 12/12 Q allclose at atol=rtol=0.0002 and exact selected action equality |
| Source-native CUDA observer | cuda:0, Tensor.argmax, shape [1,6], research/bl_residual/f5_qualification.py:135 |
| Transfer | 82.782004376175 s <=150 s: PASS |
| Whole staging | 101.841377332923 s <=300 s: PASS |
| Mac export recovery | All 10 payloads, sidecars, completion markers, hashes and sizes verified |
| Terminal and runtime release | PASS, stop_verified=true; saved sessions={} after stop; assignments=0 |
| Total elapsed | 939.303517124965 s |
| Refusals | None; one initial renewal-heartbeat CLI failure recovered by the existing supervisor |

Fixture PASS rows attest both parity predicates in the pinned executable; raw Q/action vectors were not exported. Runtime disappearance is verified from preserved supervisor evidence, not a new account query. Existing verifier tooling rederived identical canary evidence from measured clocks, scanner inventory, verified uploads, recovered exports and source/generation-bound observer event.

Full source timing, disjoint optimizer/resume calibration, Mac companion timing, production feasibility, actual L4 plan sealing, final-inventory canary and independent ACCEPT_F5_FIT remain gates. Bootstrap paired files are calibration-only, and no optimizer ran. No Colab calls, dispatch, scientific payload reads or scientific outcomes occurred in this verification.
