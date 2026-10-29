# Reseal-04 host-admission erratum

The packet, seal, runtime pin and archive remain immutable. Broad integration
against the new host transport exposed two additional historical assumptions.
This sibling supersedes the implementation note's statement that only two Mac
source pairs differ: the exact third pair is ct_cuda_capture.py, a Mac-only
historical fixture validator. It now hashes capture-era Git objects at the
capture's sealed full commit rather than today's host supervisor files.
Recorded evidence, captures and runtime source bytes are unchanged.

Sandbox host and runtime preflight admit exactly the previously qualified
5833e54e transport and the owner-authorized offline-qualified cfd50f45 transport.
The live ops wrapper still separately requires current HEAD qualification and
an exact current or SHA-bound previous runtime qualification. Foreign trees,
dirty transport, invalid pins and missing capture-era Git objects refuse.
No qualification registry, account configuration or runtime source is modified.

The reseal-04 broker is fast-forwarded to include these host-only corrections;
the seal's host_code_commit records the original preparation code, not the
final broker HEAD. Final broker HEAD and verification live in the task SUMMARY.
Independent review of these additions remains required before operator dispatch.
