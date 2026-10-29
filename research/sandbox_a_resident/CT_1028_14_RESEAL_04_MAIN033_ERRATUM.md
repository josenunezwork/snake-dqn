# Reseal-04 host merge after origin/main 033c9275

The packet, seal, runtime af2bba1c and archive remain unchanged. During offline
qualification origin/main advanced from 041a01b6 through 033c9275, adding bounded
admission-test receipts, process-tree cleanup and detached allocation leases.
The final broker includes that merge plus the accepted C/T continuation fixes.
The task SUMMARY records the final broker commit; the seal's host_code_commit
continues to identify the original preparation code.

Generic sandbox admission uses the new 1200-second pre-allocation host window.
C/T keeps its sealed 180-second inner and 420-second outer bounds. Both GPU
variants receive a separate create-only admission receipt before allocation.
Refreshed/native admission reopens structural, source, input and ledger evidence;
only full CPU tests may reuse an exact code/config/input/environment identity.
C/T --checks-only still validates the complete committed packet and reports
that tests were not run by that invocation. Full tests strip inherited host
receipt/lease controls. An invalid packet cannot use checks-only to pass.

Preferred dispatch now routes each variant's own receipt and allocation lease
through live admission and detached startup. The ops lock remains continuously
held across both variants, each bounded by its own 600-second window inside the
1200-second pair envelope. A lease is bound to that variant's live-manifest SHA,
parent PID and deadline; primary grants are revoked before fallback, and the
final grant is revoked on exit. Unknown/changed/missing evidence still refuses.
The 100-second slow-fallback and timeout-negative tests exercise real timeout
calculation, receipt routing and child lease acquisition for C/T and generic
sandbox bounds, without a provider. No automatic retry or release is added.

New generic packets use the current cfd50f45 transport pin; C/T still admits its
exact frozen 5833e54e runtime through the preserved qualification. No runtime
source differs beyond the three already recorded Mac-only hash pairs. This is
an integration/admission correction, not a science or runtime reseal. Independent
review of the additions remains required before operator dispatch.
