# PLAN_v2 WP-1 operator boundary

Prospective authority: AMENDMENT_PLAN_V2_WP1.md (published before preparation).
Sol-S owns this package; Codex-V is the independent reviewer. Claude dispatches.
No Colab calls or selection-world gameplay are part of implementation.

The original BANK_L2 bytes are fixed at SHA256
0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3.
Eight deterministic immutable chunks cover exactly 160 worlds, 8/8/4 per chunk.
The root plan and each chunk plan bind the analyzer and runtime input hashes.
Only finalize_v2 is the official one-look entrypoint: it requires eight terminal
supervisor receipts, verified release, exact prepared manifests and exported
shard hashes, then the existing complete-row validator. Faults print INCONCLUSIVE
with no partial metrics. A create-only look-opened.json fences the scientific
look. FUND_MAC_SYSTEMS maps to FUND_SYSTEMS; PACKAGE_KILL is unchanged from A08.
Thresholds/statistics/operator/bank and survival reporting are unchanged.

Transport: one <=6 MiB compressed evidence envelope contains the compact bank,
original nine-root fixture archive, authorities and platform/parity references.
Checkpoints use the existing public catalog and hash verification at runtime.
No expanded L2 root directories travel over ContentsClient: the world builder
reconstructs from the fixed seeds/rosters. The only permitted source exception
is the exact historical chunks.py hash; every other ruler/operator dependency
must match. ruler_v2 preserves the original fixture object and canonical hash.
No K-1 dependency change is admitted automatically.

transport_v2 runs pinned source.prepare and the real supervisor upload_file
primitive on all bootstrap/local/fallback bytes. The local ContentsClient
boundary serializes actual JSON/base64 and executes pinned transfer assembly /
remote-hash RPC code locally. It makes zero Colab calls and does not measure
provider latency. Both normal and fallback upload forecasts must be <=150s;
fallback fetch+upload <=300s. Every receipt has .sha256/.complete.

Each assignment first checks source, unchanged bank and repriced envelope, then
runs the three excluded worlds and compares all semantic row fields with the
Mac reference (only timings/checkpoint directory presentation removed). It
creates runtime-local parity before any selection. Failed parity rejects the
assignment; no automatic retry or partial analysis. This is development parity,
never Mac strict/serving qualification.

Pricing uses actual per-mix worker batch timings from all270 Y1 records, 578
batches/world, exact interleaved scheduling on12 workers and30min per assignment
for setup, parity and export. Timeout >=2*estimate and <=720min; eight chunks
share200u including40u reserve (25u/chunk including5u reserve). Thus the equal
allocation's usable600min is stricter than720min. Never clip an estimate.
Mac Y1 /2 is sizing only and cannot open before Thu Oct22 12:00 PT, and only if
a Y1-L4 receipt is absent. Measured semantically matched Y1-L4 per-mix timings
replace the fallback; the old scalar R>=2.18 boundary no longer admits/refuses.

The actual Mac fallback is NOT_READY: estimate565.974723587min/chunk;
required2x timeout1131.949447175min; liability301.853185913u plus40u reserve =
341.853185913u. Even the average-work lower bound exceeds the envelope.
Prepared chunk manifest proposals retain the actual estimate and720min hard
ceiling; supervisor validation REFUSES them. They are never SCHEMA_VALID or
dispatchable. A measured Y1-L4 receipt is required for a fresh fitting family.
Y1 separately has estimate48min/timeout96min,4u including0.8u reserve.

Deadlines PT: Thu10:00 code; ThuAM Y1-L4; Thu12:00 first chunks; Thu20:00 latest
first start for Sat16:00 print. deadlines_v2 writes create-only PLAN §2 slip
schema after a missed gate, under exp/week1-20261021/stage0/slips/. A slip blocks
dependents only and cannot alter bank/bars/budget. No premature slips.

After every chunk actually completes and release is verified, seal its wrapper:

```sh
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.finalize_v2 \
 --plan /absolute/family/plan.json --seal-index 0 \
 --supervisor-receipt /absolute/family/chunk-01/runtime-output/receipt.json \
 --shard /absolute/family/chunk-01/runtime-output/exports/l2-shard-0.json
```

Repeat with indices0..7 and corresponding01..08 paths. After all eight terminal
wrappers exist, from the published pinned source checkout:

```sh
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.finalize_v2 \
 --plan /absolute/family/plan.json --out /absolute/family/decision.json
```

Before ANY assignment Claude must independently refresh usage/inventory, bind
positive rates for all other assignments, inspect previous terminal evidence
and verified release, and retain the exact real-transport receipt for the final
manifest bytes. An inventory edit changes manifest bytes: write a fresh file
and rerun transport_v2 before dispatch. Existing preparation never grants
assignment, release or serving authority. No paid idle, no automated retries.
