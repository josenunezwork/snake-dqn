# I0 native compile prerequisite

Prepared only for
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-integration-i0-20261024-03`.
Runtime pin remains `968f02765f7c86ca27d2947347896e2fe4e6ce23`; broker remains
`/Users/josenunez/Projects/ml/snake-dqn-gpuint-qualified-03.noindex`.
No jobs were dispatched. Existing I0 component/accounting/main admission still applies.

`tools/ops/integration_compile.py prepare` adds
`packet/native-compile/native-compile.json` using the frozen native compiler pack
and sparse upload inventory pattern. It requests L4 only, compiles A100/L4 targets
`8.0;8.9`, and runs no kernels or gameplay. Its unchanged base/effective envelopes
are 10/20 minutes, 0.6/1.2 units, and 0.1/0.2 reserve units, priced at 3 units/hour;
the five-minute estimate remains an unmeasured forecast for this pin. The upload
is 48 exact source files in a 137,321-byte archive plus one 249-byte requirements
input, within the 1 MiB per-file bound. The canary retains its original bundled
metadata upload and all original runtime/source/input/cap settings.

The root `dispatch-compile.sh` and repaired `dispatch-canary.sh` invoke the tracked
Mac companion. The canary verifies same-pin compile PASS, terminal success,
verified stop, saved manifest digest, native source/compiler flags, architectures,
payload hash/size, and local checksum/completion files. It creates the sibling
`packet/integration-canary-compiled.json` only after full validation. Existing
hash-bound manifests, preparation records, and handoff receipts stay intact; the
original wrapper is saved as `dispatch-canary.before-compile.sh` with sidecars.
The new wrappers use the corrected host dispatcher and frozen qualified broker.

The passing sparse compiler receipt at
`gpu-sparse-metadata-20261024-02/packet/native-compile/output/receipt.json` was
inspected and accepted by the corrected reader. Both receipt and terminal list
only `native-compile.json`; its sidecars are separate local files. The former
reader incorrectly required sidecar inventory rows. Portable regression tests
now use the payload-only shape and retain tampering/missing-proof refusals.

Full generated-helper dry runs use that real receipt's structure, replacing
manifest identity, commit, native hashes/specifications, and payload digest/size
for this I0 pin. The diagnostic fixture and exact dry-run logs are confined to
`compile-proof-verification/` under the I0 artifact. Its `VALIDATION.json` is
offline evidence only. Dry runs stage real uploads, run qualification and actual
supervisor/dispatcher validation, and create no dispatch reservation, bound canary,
canonical compile output, or allocation. Live accounting is checked at dispatch.
`tests/test_integration_compile.py` repeats these local acceptance checks, while
portable native/dispatcher/supervisor regressions cover the underlying controls.

Run the compiler first:

```bash
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-integration-i0-20261024-03/dispatch-compile.sh
```

Wait for its terminal PASS receipt with verified stop, then run the canary:

```bash
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-integration-i0-20261024-03/dispatch-canary.sh
```

For the dispatcher's existing exact accounting allocation retry, set
`NATIVE_COMPILE_RECEIPT` to the actual `output-retry-1/receipt.json` on the canary
command. Receipt identity is verified and changing an already-bound manifest is
refused. No workload retry or synthetic proof is used by the default commands.
