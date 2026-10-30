# Operational clarification 01 — 2026-10-30

Recorded before any registered 3,000-update fit or GPU allocation. The excluded
Mac three-update preflight ran after the original registration was committed;
no assessment AUC or other real held-out metric has been computed.

The qualified H100-first wrapper preserves the H100 reserve in its A100 sibling
(`research/compute/gpu_preference.py:variants`). Consequently the actual common
base/effective **reserve** is 2/4 units for the main, and 0.5/1 unit for the
canary, on either hardware. These replace only the A100 reserve arithmetic in
REGISTRATION.md; the preparer records those actual values for both SKUs.
Reserve/unit fields are informational under the standing 2026-10-29 rule.

All timeouts, unit caps, useful-work limits, data/split, update schedule, masks,
targets, seed, bootstrap, GO/GRAY/KILL bars, one scoring look, retry maximum and
release scope remain exactly as frozen. No runtime envelope is enlarged and no
immutable manifest or original registration is edited. Both original registration
and this clarification are shipped as hash-bound inputs in the successor packet.
Operator execute-time admission and independent review remain separate requirements.
