# F5-W refused capture reference-admission repair

Status: implementation submitted for independent review; attempt-03 remains DRAFT,
not accepted or dispatched. No Colab/provider call, scientific fit, outcome look,
serving change or release occurred.

Astra rejected source pin 2c8bc80b06bbc4244b1ecdd8ab184a7fbc21cf8c because a
refused capture exported a bare environment-raw.json without its acquisition or
configuration failure. Parseable SMI stdout from a nonzero command could validate,
compare and enter sealed control admission despite the originating refusal.

Receipt v4 now requires capture_status=SUCCESS. Acquisition errors remain inside
the raw receipt, including command output/stderr evidence. Any acquisition,
validation, configuration or reference-admission failure marks that receipt
REFUSED and retains its failure list. Refusal-only fields are outside the exact
admissible schema. The export boundary also marks any refused raw export before
publishing the immutable raw/diagnostic pair and exiting nonzero. No environment.json
is published on refusal. Raw source observations and aggregate diagnostics survive.
The mandatory status also refuses old status-free receipts; a fresh capture under
this prospective v4 implementation is required.

Regression tests exercise failed-but-parseable SMI and configuration-only refusal
through receipt validation, same/cross-assignment comparison, and control admission
against a fresh successful capture. Both wrapper and raw content refuse. The
reference-consumer sweep additionally covers canonicalization, assignment
boundaries, environment evidence, qualification bootstrap (renamed raw content),
and resume evidence. Missing, malformed and REFUSED statuses cannot admit.
Hashing remains an identity operation, not admission; actual resume validates.
Source inspection found no other production refusal-reference bypass.

The seven science files and schemas v1-v3 remain byte-identical to the rejected
pin and prior baseline. Receipt v4 is prospective and remains subject to independent
acceptance. All prior receipts, packets and drafts remain immutable.

final-08 carries the new exact source closure, schema, regression/lint evidence,
frozen-byte audit and a new create-only attempt-03 draft successor. The successor
retains job attempt 03, all historical elapsed/debit accounting, unused output/state,
and the owner's 30-minute runaway timeout (including its exemption). Independent
implementation acceptance and current-account/transport preflight remain pending.
No dispatch instruction is granted by this packet.
