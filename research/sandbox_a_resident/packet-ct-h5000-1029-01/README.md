# H5000 successor packet 01

Prepared only. No allocation, gameplay collection, branch scoring or fitting has
been dispatched. Runtime source is a55f6d2e8272f7b2452d44cd057390a73c95dbd4.
The source pin is local to astra-ct-h5000-impl and must be published by the
operator before a provider can fetch it. The implementation task does not push
or merge. An independent Argon review is retained in the artifact handoff.

This packet tests natural collection/support feasibility only. Read
`../ct_h5000/PREREGISTRATION.md` for unchanged bars and terminal incomplete rules.
The H100 primary and exact derived A100 sibling share science and caps. A100
fallback is only the qualified dispatcher's allocation fallback; never manually
run both siblings, and never use fallback to retry a scientific shortfall.

`DRY-RUN.txt` validates both hardware variants with zero allocations.
`DISPATCH.txt` is the future operator command; it is not executed by Astra.
Both commands use tools/ops/dispatch.sh, a fresh output/state identity, immutable
uploads and an empty verified ledger. Do not edit this packet after sealing.

H100 timing/yield and mature-state host-memory/exports remain unqualified.
A completed H5000 bank missing support ends INCOMPLETE_SUPPORT. Expiring the
7200-second collection clock ends INCOMPLETE_COLLECTION with no partial bank
admitted or support look. The whole-session timeout is separately 300 minutes
to allow startup/export/cleanup; it does not extend collection.
