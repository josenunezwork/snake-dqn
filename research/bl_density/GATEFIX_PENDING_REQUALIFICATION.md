# DENSITY-2 gatefix status — pending requalification

This is a prospective operational status note. It does not edit or supersede
the hash-bound `METRICS.md` or amendments 01–04.

The historical DENSITY-2 blocker required a CUDA action decision for every L4
family canary even though this package explicitly runs CPU inference on L4 for
its 12 vCPUs. The corrected family gate requires that observation only when the
job declares CUDA, while retaining real transport, export-sidecar, scanner and
verified-release evidence for every job.

Changing `research/colab_cli` invalidates its prior final-only qualification.
The current DENSITY-2 source may therefore be prepared only with
`--pending-requalification`: its source pin must be published on an origin
branch, its source qualification is recorded as `PENDING_REQUALIFICATION`, and
both manifest and runtime `scientific_admission` flags remain false. This is not
dispatch authorization.

Later offline admission requires all of the following: the exact source pin on
`origin/main` with its tracked source closure clean; a fresh real final-only
acceptance result validated by the existing transport validators; and a passing
supervisor preflight. Production additionally requires the genuine recovered
12-episode canary package proof, matching family receipt and remaining-budget
finalization. No field may be hand-toggled to simulate any of those facts.
