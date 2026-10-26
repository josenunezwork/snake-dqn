# F5-W v5 memory-domain repair and attempt-04 draft

Status: prospective implementation for independent review. Attempt-03 remains
REFUSED/non-admissible. Attempt-04 is an operational DRAFT, not dispatched.

Attempt-03 (b9e48d882049622a8506f05cf00947f3500b1e1a) collected exactly one
validation/admission failure: `F5 environment parsed GPU memory differs`. Its
acquisition failure list is empty. Torch reports 23,659,151,360 bytes (22,563 MiB),
while SMI reports 23,034 MiB (24,152,899,584 bytes). The 471 MiB difference is
approximately 2.045% of the SMI total. These are separate capacity domains.

Receipt v5 replaces only the within-receipt sub-MiB equality with the inclusive
physical-consistency relation `95 * smi_bytes <= 100 * torch_bytes <= 100 * smi_bytes`.
Both capacities must be positive integers (Torch bool is refused). The fixed 95%
minimum permits up to 5% unavailable/reserved capacity while refusing impossible
usable capacity above framebuffer capacity, unit confusion, and major capacity
shortfalls. This is a conservative policy margin, broader than the one observed
471 MiB offset; it is not an offset calibrated to one assignment. NVIDIA documents
that framebuffer accounting depends on ECC and driver reservations:
https://docs.nvidia.com/deploy/nvidia-smi/ . The exact source of these 471 MiB is
not inferred from the totals alone. The rule uses integer arithmetic, no floats.

The raw Torch byte count and raw SMI MiB string remain separate exact invariants
across assignments; a one-byte or one-MiB change still refuses comparison even
when both receipts independently satisfy physical consistency. No value is
rounded, replaced or normalized. Schemas v1-v4 stay byte-identical.

The complete real attempt-03 environment-raw.json is published in the ledger as
lossless split gzip evidence with SHA sidecars and completion markers. The full
v5 regression verifies its original SHA256 before decoding, preserves every
acquired field and all package/wheel/content evidence, and changes only envelope
bookkeeping (v5 schema/pin, prospective SUCCESS, removal of the historical refusal
annotation). This counterfactual test passes full validation and classification;
it does not make the refused historical capture admissible. A fresh successful
capture under v5 is mandatory. The historical raw REFUSED status still refuses.

All seven science files remain byte-identical. Capture-history packaging now
accepts three ordered stopped failures and carries attempt-03's two diagnostic
exports without permitting admitted environment or science exports. Prior debits
and elapsed time remain disclosed; no refund, history reset or current-account
claim is made. Attempt-04 retains the owner's 30-minute runaway timeout (15 x2),
including its exemption, and the same capture/disjoint CUDA transport scope.
Independent v5 acceptance and current-account/transport preflight remain pending.
No dispatch, Colab/provider call, fit, outcome look, serving change or release is
part of this repair. Passing the real-data regression supports expected v5
capture admission on attempt-04; it cannot prove a future assignment or CUDA
transport execution.
