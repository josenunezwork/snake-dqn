# Packet 06 final appended reseal: reproduced host preflight timeout

Use `packet-ct-1028-06/reseal-02` after independent Argon acceptance. Both earlier
packet-06 seals, all prior packets and the failed/successful evidence remain
unchanged. The external registry retires both earlier packet-06 manifests by
exact SHA, while keeping the active job identity available for the unused attempt.

The requested loaded c2 repetitions reproduced a concrete cause of the previously
unexplained wrapper failure. `tools/ops/dispatch.py:require_sandbox_admission`
called C/T's nested `ct_preflight` through `command()` without overriding its
generic 60-second host limit. A retained direct-launch diagnostic in
`astra-impl-ct6/chain-admission-final.log` shows **157 sandbox tests passing**, then
**247 C/T tests passing in 51.69 seconds**, followed by the actual nested command
timing out after **59.99997749994509 seconds**. Admission and process startup/
teardown consume the remainder. Four concurrent validations plus two owned CPU
load processes reproduce this failure; it is not a scientific or provider fault.
This explains the newly reproduced failure class; the packet-05 run lacked the
underlying stderr needed to assign its particular historical exit conclusively.

Only C/T's nested host preflight now uses the already-existing **180-second**
`SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS` test window. Ordinary validation and
other study routes retain their existing limits. Registered job wall/unit caps,
runtime lease, science, allocation, source transport and retry policy are unchanged.
The outer offline validation window remains bounded at its existing value.

Before dry-run temporary logs disappear, dispatch now includes their final 64 KiB
in its refusal stderr. Regression tests exercise the actual C/T routing call and
prove the exception includes a nested diagnostic written through the real log
stream. The existing deadline-clipping and ordinary timeout tests remain in force.

The final source repeats the complete qualification and twenty full c2 dry-runs
under the same four-validation/two-load-process configuration, preserving stdout,
stderr, exact argv/environment, timings and load cleanup receipts outside
dispatcher-owned log names. Earlier failed runs are retained as diagnosis, not
counted as passing final-source evidence. No account/provider action, live dispatch,
automatic retry, push or serving change is performed.
