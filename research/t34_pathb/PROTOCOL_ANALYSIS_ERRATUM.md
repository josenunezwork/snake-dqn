# T34 Path B analysis implementation clarification — 2026-10-19

Prospective for every real H5000 efficacy bank. No real-bank efficacy data exist.
The completed H4 plumbing bank is smoke evidence only and is excluded from every
claim, threshold choice and uncertainty estimate.

The survival-band-v2 reference in the binding success contract means paired
**alive-frame survival_fraction confidence bounds**, as specified in the ratified
governance amendment section “The rule”; the original protocol incorrectly
described its band checks as no-death point comparisons. Supersede only that
description: pooled equal-mix alive-fraction lower95 must be strictly >-.05,
each mix's paired alive-fraction lower95 strictly >-.075, and each mix's absolute
alive fraction >=.30. This one-look development bank uses one-sided 95% Student-t
bounds across the registered independent blocks. The future strict gate still
uses its own frozen RCI/OBF plan and fresh banks. The other registered no-death,
alive-fraction, scripted, mass and boost guards remain additional checks with
their original thresholds. No point estimate substitutes for a confidence bound.

Completed efficacy must have a create-only completion receipt bound to its
manifest; any incomplete marker defeats completion. Timing outputs are validated
even for smoke, without turning smoke timings into acceptance evidence.

For repeatable implementation checks only, each explicitly invoked smoke names
a fresh `--smoke-key` within the disjoint smoke namespace. Keys are frozen before
that smoke starts and included in its manifest; collisions still reject the
entire smoke allocation. No automatic retry occurs. Smoke cannot emit efficacy
estimates or qualify a controller. The real-bank namespace and single-look/no-
extension rules are unchanged.
