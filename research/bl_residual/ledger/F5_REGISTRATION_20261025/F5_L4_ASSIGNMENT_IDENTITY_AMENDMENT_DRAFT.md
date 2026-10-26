# F5 L4 assignment identity amendment — draft for Astra

**DRAFT_PENDING_ASTRA; PRE-DATA; NOT ADOPTED.** This draft authorizes no implementation change, capture, canary, dispatch, fit, scientific look or release. The rejected implementation remains unaccepted. Written in response to `REJECT_F5_W_IMPLEMENTATION_36561254.md`; source checkout remains `2e20aa1c7686bc71920faba63c1c671215c306e8`.

## Authority and reason

The adopted registration `F5_REGISTRATION_ADOPTED_20261025.md` (SHA256 `4e218263f169334415968f02eb2d767e1ff136dfc3ccf470090dd474b31dd6a8`) incorporates common-environment/RNG amendment c6b2b5df: prose SHA256 `20df1c5ec2a80d2f81ce4711c8bbb163814c181a0fe3de29573042465ef28064`, machine binding SHA256 `bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5`.

The machine binding's `hardware_receipt.equality` explicitly includes GPU UUID. Its `hardware_receipt.publication` says: “Later arm/stage receipts reference and compare with the first admitted receipt. Missing identity or drift refuses execution.” These are binding requirements, not implementation conveniences. Separate capture, prepare, qualification and paired-fit L4 jobs cannot guarantee that physical identity: the qualified supervisor releases each assignment and has no supported retained-L4 route. No equality rule has been relaxed in code.

## Proposed narrow supersession

Subject to Astra's adoption BEFORE affected execution, supersede only cross-assignment equality in `hardware_receipt.equality`, `hardware_receipt.publication`, and the prose amendment's corresponding common-field comparison. Continue recording the complete original raw capture in every create-only, hash-verified receipt.

1. **Between distinct L4 assignments:** compare the same sealed source, configuration, checkpoint, lock and applicable plan identities; exact Python build/cache tag/compiler; GPU model/capability/memory and separately parsed driver version; Torch version/git/build, CUDA runtime and cuDNN; NumPy/Numba/llvmlite/SciPy and BLAS/LAPACK build identities; installed package versions; wheel URL/SHA and portable installed-build identities; and every precision, determinism and thread control. Retain CPU model/ISA, vCPU count, OS/kernel/build equality and the registered one-L4/minimum-six-vCPU admission. Missing required evidence or any mismatch in these common identities refuses work.
2. **Assignment provenance:** physical GPU UUID, the UUID component of the driver query, CPU-affinity processor numbers, host node name, and installation-root-dependent metadata are recorded per assignment rather than required to match a prior assignment. CPU-affinity cardinality remains subject to the common vCPU admission. The driver query must be parsed so its model/driver/capability/memory checks remain enforced. Raw RECORD/direct-url metadata remain exported; portable build comparison must additionally verify the locked wheel identities, METADATA/WHEEL and canonical installed library contents, excluding only demonstrably installation-root-dependent bytes. A version string alone is insufficient. The exact normalization and exhaustive field classification require independent implementation review; no unclassified field may be silently ignored.
3. **Within an assignment:** capture before work and compare every original common field, including physical UUID and raw installed metadata, at the relevant plan/qualification/arm/resume boundaries. C then W MUST execute in one verified runtime/assignment. This amendment supplies no permission to split the paired fits, weaken RNG/full-state continuation checks or resume across assignments; that would require separate prospective review.

Plan preparation remains separate from transport qualification. The subsequent implementation must supply the source-native CUDA six-action/observation path on sealed-disjoint registered timing fixtures, actual timing/parity proof and full science-upload/producer-export coverage required by the unchanged broker. CPU-only preparation or hardware capture cannot qualify a CUDA family. Independent acceptance of that implementation and all existing actual-plan, publication, timing, feasibility and packet gates remain required before fitting.

## Unchanged scope and stop

W/control science, selected support/exclusions, loss and optimizer, seed 3403, the shared 2,000-by-256 ordered plan, RNG states, N, allocations, one-look discipline, timing bars, registered envelopes, Mac placement and release authority remain unchanged. Original adopted files and rejected packets remain immutable. No scientific payload, checkpoint content or protected result was opened for this draft; no runtime/Colab/dispatch call or scientific execution occurred.

**Requested Astra decision:** adopt or reject this narrow prospective cross-assignment comparison amendment. Until adoption and subsequent independent implementation acceptance, STOP: no repaired implementation or new executable review packet is asserted by this draft.
