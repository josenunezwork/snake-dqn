# Packet 06 appended reseal: full qualified CLI terminal log

The first packet-06 seal remains byte-identical. Its source is retired by exact
manifest SHA in the external supersession registry; use `packet-ct-1028-06/reseal-01`
after independent Argon acceptance. Neither seal is a release or dispatch approval.

The first corpus exercised the real `Broker.run` producers and all receipt/output
formats. A final entrypoint check found that the driver's exception handler did
not capture `supervisor.main`'s argparse terminal stderr line. That distinction
matters for concrete CLI upload/exec timeouts: the final line contains `failed`
and `TimeoutError:`, which the generic log denylist correctly rejects without
additional receipt-bound interpretation.

The appended committed `ct-supervisor-06-main` corpus executes the unchanged
qualified `supervisor.main` for clean completion and every registered mechanical
fixture, capturing its actual stdout and terminal stderr. Only construction of the
real broker is adapted to attach its existing offline observer. The new
`ct-supervisor-06-entrypoint-main` capture runs the actual C/T shell job under the
same CPU-only test shim, real local runtime supervision, fake provider transport,
real CLI terminal receipt and stderr. These augment, never replace, earlier goldens.

Admission accepts an `INCOMPLETE:` footer only when the entire cause and absolute
receipt path exactly equal the bound qualified receipt. It validates the receipt,
nested execution facts and scientific-fault precedence before that exemption.
There is no substring whitelist for a line containing an allowed timeout.
Different cause, different receipt path and appended fault regressions refuse.
Production writer/preflight passes the actual provenance receipt/path into log
scanning, so external log files do not depend on adjacent guessed files. Logs
inside an output tree can use only their adjacent retained receipt. Playback
explicitly rebinds relocated paths in its private copies; original goldens remain
unchanged.

All earlier M1–M6 and minor fixes remain in force. The science, numerical code,
seeds, endpoint, 2x envelopes, export caps and retry maximum are unchanged.
The appended seal uses a new source pin, source archive/plan hashes and unchanged
qualified transport. Qualification is repeated on that source, including the
two-continuation integration and twenty full c2 dispatch dry-runs under load,
retaining per-run stderr outside dispatcher-owned log names. No real dispatch,
provider/account operation, push, automatic retry or serving change is authorized
or performed by this implementation.
