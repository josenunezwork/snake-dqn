# Final V3 verification

**232 passed in 127.70 seconds**, using the exact thirteen-file command in
`VERIFICATION.md`, under `nice -n 15`, CPU, `pytest -n 0`, and all six numeric
thread controls plus OMP_THREAD_LIMIT set to one. The additional test asserts that
the 64 MiB environment export allowance exceeds the known 13,419,976-byte raw
reference size. No CUDA or runtime qualification was executed.

Black (`--check --workers 1 --line-length 100`), isort (`--profile black
--check-only`), flake8 and whitespace checks passed for the V3-changed Python
files. The earlier full sixteen-file quality checks and Bash syntax check remain
applicable. No measured/scientific code changed in V3. The final static reviewer
found no issue in the append-only export-capacity correction and predecessor link.

Final executable source commit:
`c7d293a0f7e5aa855df8d20707d287f4bda7ab68`.
The final package pins this commit. Later commits contain only create-only
seal/verification metadata; source inventory bytes remain identical.

V2's seal, package, publication and verification record are preserved. V3 links
the rejected original and the V2 interim seal. The V2 package is not operational
authority: final artifact inspection found its 8 MiB export capacity inadequate.
V3 fixes that bound and changes only engineering identity/capacity metadata.

The V3 package and actual driver are exercised offline against the complete staged
input inventory and the dedicated empty component ledger. The publication record
binds all resulting manifests, envelope, raw byte identities and dry-run results.
The finite CUDA parity qualification manifest is validated by the unchanged
supervisor; the measurement manifest must refuse without a matching SHA-bound
family-canary receipt. Validation makes no provider call and creates no runtime.

The package remains `SEALED_PENDING_INDEPENDENT_ACCEPTANCE`. Matching genuine
current-driver L4/CUDA/native-v6 evidence, complete measured costs and independent
engineering/package acceptance are not supplied by these tests. Unknown cost
forecasts remain HOLD. No acceptance placeholder, numerical look, fit, timing
supplement, dispatch, Colab call or serving action was made.
