# Sandbox B: frozen-B8 successor head/loss sweep

This is successor development under the [owner-adopted Standing A100 Development Sandbox](../../docs/research/governance_a100_sandbox_20261026/ADOPTED.md). The runtime verifies the registered D1024 NPZ hash and exact row/life/parent membership before opening outcome arrays. It uses the fresh parent split salt `a100-dev/sandbox-b/split-v1`: first eight SHA256 bytes of `salt + '/' + ultimate_parent`, big endian, modulo five equals zero for development validation. No rows or lives cross parent partitions.

The fixed family has 60 configurations: zero-initialized F4 linear residual, length/margin-gated two-layer MLP residual, and cosine-normalized residual, each with ordinary KL weights {1,2,4,8,16} crossed with survival-pair weights {1,2,4,8}. Each configuration gets seed 3403, 250 Adam updates, batch size 512 and learning rate .001. B8's 256-dimensional fusion features are frozen and cached once. On A100 the cache and training tensors stay on the GPU; `auto` falls back to CPU locally. The endpoint is 15,000 total updates.

The survivor-reference target is the retained F4 exact/legal/served-selectable survivor-over-death pair target, `softplus(Q_death - Q_survivor)`. Unknown branches never become training negatives; dying branches' mass never defines a reference. Ordinary KL retains the bare B8 action distribution. Validation reports row and life-macro rescue retention, plus the conservative ordinary-harm proxy: changed B8-survive→successor-die or changed unknown selected/base outcome. These ordinary rows are conditioned on selected rescue lives, so this proxy is not a natural-deployment harm rate. The three heads share the same data, split, seed and update count.

Outputs include `results.json`, `results.csv`, per-config checkpoints and SHA sidecars, `checkpoints.tar`, `verified-ledger.json`, `timing.json`, and immutable per-config unit bundles. Checkpoints/results/timings carry `simulator-D`. All results are development evidence at most. This package does not select or export a Mac evaluation candidate automatically. Only one frozen SHA may later cross through a separate judge-reviewed registration and fresh Mac bank.

## Prepare and validate

The committed `dispatch-manifest.json` and ten-line `sandbox-manifest.json` are the ready original attempt. `TIMING_FORECAST.json` records the measured local smoke and conservative CPU-equivalent extrapolation, including a separate bootstrap/staging allowance. The A100 forecast is not a measured GPU rate.

For another fresh packet, first create an empty ledger (or a ledger of already verified units):

```sh
./venv/bin/python -m research.sandbox_b_heads.resume empty-ledger \
  --data /absolute/registered/d1024-allowed.npz --b8 /absolute/B8.pth \
  --output /absolute/fresh/verified-ledger.json
./venv/bin/python -m research.sandbox_b_heads.package \
  --data /absolute/registered/d1024-allowed.npz --b8 /absolute/B8.pth \
  --ledger /absolute/fresh/verified-ledger.json --estimated-minutes MEASURED_FORECAST \
  --output /absolute/fresh/dispatch-manifest.json --output-dir /absolute/fresh/runtime-output \
  --sandbox-manifest research/sandbox_b_heads/sandbox-manifest.json
```

Commit the manifest before preflight. Use the qualified broker's offline `--dry-run`, then:

```sh
./venv/bin/python -m tools.sandbox.preflight research/sandbox_b_heads/sandbox-manifest.json
```

The manifest's unit guard is only the prospective wall-time liability at an 8 units/hour pricing ceiling; it does not ration development credits. The broker checks actual calibrated prices, capacity, assignment isolation and coexistence at dispatch. Claude dispatches; these modules never call Colab.

## Completion and resume

Each fixed allocation index 0–59 has an immutable tar unit containing checkpoint, report, internal hashes and full frozen identity. The fingerprint binds the source bytes, input/B8 hashes, salt, seed, updates, batch size, learning rate and smoke selection. Payloads publish atomically; SHA sidecars precede atomic completion markers. The qualified exporter can pull completed units throughout the run. The terminal ledger is sealed once, so exporter polling never observes a mutated completed file.

After a **verified mechanical failure and verified stop**, preserve all original evidence and have the Mac build the missing-unit ledger from verified exports. This command reads completion identities and payload path/size/hash only; it never opens checkpoint/report contents:

```sh
./venv/bin/python -m research.sandbox_b_heads.resume build-ledger \
  --exports /absolute/verified-original-exports \
  --data /absolute/registered/d1024-allowed.npz --b8 /absolute/B8.pth \
  --output /absolute/fresh/completion-ledger.json
```

Pass that ledger to `package`. It binds and uploads every verified bundle, rewrites runtime paths to portable basenames, verifies each bundle before restoring its checkpoint/report, skips exactly verified configurations, and aggregates the original and completed units into the same 60-row table. Duplicate IDs, foreign namespaces/allocations/recipes, missing or corrupt payloads refuse. The Mac ledger builder refuses duplicate allocation indices rather than accepting the same unit twice.

Mechanical triggers are runtime/lease termination, transport interruption, or incomplete export after a stopped attempt with missing configuration units; numerical or outcome-based selection is not a trigger. At most two completion attempts follow the original. Every attempt gets a fresh manifest, job/dispatch identity, output directory and receipt; the same registered 60 units and per-attempt 2× envelope remain fixed. No automatic retry or allocation occurs in this package. Configuration boundaries are the resume unit: an interrupted unverified configuration restarts from its original seed; completed configurations are never recomputed.
