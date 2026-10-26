# F5 W receipt v4 and complete refusal diagnostics

Status: operational DRAFT for independent review. No dispatch or Colab calls.
No scientific fit, look, release or serving change is authorized.

Attempt 02 at a840fc2cf54aea4ee324bcc98a84a7fa6b4611bc refused before
science with `F5 environment GPU provenance fields disagree`. The original
receipt records 558.1542476670584 seconds, CHILD_NONZERO_EXIT, no exports and
verified stop. Its 49 original output payloads are now copied byte-identically
under `ledger/F5_W_CAPTURE_AFTER_22e86986_REPAIR_02/output-02/`, each with
create-only SHA256 and completion sidecars. The sibling
`ATTEMPT_02_FAILURE_PUBLICATION.json` binds the inventory and retains both
failed attempts' elapsed/accounting evidence. The two elapsed debits total
979.9846945420141 seconds. Observed account debit and priced elapsed exposure
remain separate observations; no debit was erased or refunded.

The root cause is confirmed against the pinned PyTorch 2.9.1 UUID formatter:
Torch emits a bare UUID while nvidia-smi emits `GPU-<uuid>`. The old direct
comparison necessarily disagreed for that representation. Old regression
fixtures incorrectly supplied a prefixed Torch value and concealed the defect.

Receipt v4 retains Torch `gpu` and `gpu_uuid`, SMI parsed values, and complete
raw SMI stdout. The sole UUID mapping requires and removes exactly one leading,
case-sensitive `GPU-` from SMI only, then compares the remainder case-exact to
Torch. Missing, empty, repeated and wrong-case prefixes refuse. GPU names
compare exactly after CSV boundary trimming, with no aliases; runtime hardware
admission now requires exactly `NVIDIA L4`, so L40/L40S cannot pass a substring
check. The v1/v2/v3 schemas remain unchanged.

The cross-source audit found no other confirmed normal-L4 representation
mismatch. Capability `[8, 9]` versus `8.9` and byte versus integer-MiB memory
comparisons retain their explicit existing conversions and strict sub-MiB
bound. Host CPU count and allocated affinity, CPU model and architecture,
driver and CUDA/cuDNN version domains, and kernel/platform strings remain
separate raw observations. Torch/NumPy runtime/distribution versions compare
exactly. The existing broker checks host count before bootstrap; F5 requires
six usable affinity CPUs and reports restricted cpusets in diagnostics. Broker
transport code is unchanged. Attempt 02 never exported its raw environment,
so its hidden name/memory/kernel/package observations cannot be reconstructed
or claimed valid from this repair.

Acquisition now observes independent fields and packages once, continuing after
source failures. Unavailable fields remain null, with explicit errors and raw
subprocess stdout/stderr evidence (base64 when bytes). Malformed SMI stdout is
stored before parsing. Validation enumerates independent receipt, nested
field, package, installed-content, metadata and source-agreement failures.
Package validation traverses inventories in linear time without making a full
inventory copy for each file.

The capture CLI publishes `environment-raw.json` and
`environment-diagnostic.json` with completion markers before any refusal exits
nonzero. Diagnostics carry `evidence_only=true` and
`scientific_use_authorized=false`; refused capture never publishes
`environment.json` and shell fail-fast prevents qualification/science. The
unchanged runtime scanner and supervisor retrieve completed exports before
raising the child failure. Successful captures publish the admissible receipt
as well as diagnostics.

The final-07 packet binds the current source, v4 schema, complete comparison
audit, realistic L4 regression tests, and seven science-file hashes. It carries
an attempt-03 operational DRAFT with job ID ending `-03`, unused output/state,
30-minute timeout and matching exemption under the exact owner ruling
`OWNER_RULING_COLAB_BUDGET_20261025.txt`. Historical debit remains disclosed;
unit history does not block this owner-approved runaway timeout. Historical
pricing/account observations are stale: independent implementation acceptance
and current account/transport preflight remain required before an operator may
dispatch. This work performs neither.
