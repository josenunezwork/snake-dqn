# Sandbox B: frozen-B8 successor head/loss sweep

This is successor development under the [owner-adopted Standing A100 Development Sandbox](../../docs/research/governance_a100_sandbox_20261026/ADOPTED.md). The runtime verifies the registered D1024 NPZ hash and exact row/life/parent membership before opening outcome arrays. It uses the fresh parent split salt `a100-dev/sandbox-b/split-v1`: first eight SHA256 bytes of `salt + '/' + ultimate_parent`, big endian, modulo five equals zero for development validation. No rows or lives cross parent partitions.

The fixed family has 60 configurations: zero-initialized F4 linear residual, length/margin-gated two-layer MLP residual, and cosine-normalized residual, each with ordinary KL weights {1,2,4,8,16} crossed with survival-pair weights {1,2,4,8}. Each configuration gets seed 3403, 250 Adam updates, batch size 512 and learning rate .001. B8's 256-dimensional fusion features are frozen and cached once. On A100 the cache and training tensors stay on the GPU; `auto` falls back to CPU locally. The endpoint is 15,000 total updates.

The survivor-reference target is the retained F4 exact/legal/served-selectable survivor-over-death pair target, `softplus(Q_death - Q_survivor)`. Unknown branches never become training negatives; dying branches' mass never defines a reference. Ordinary KL retains the bare B8 action distribution. Validation reports row and life-macro rescue retention, plus the conservative ordinary-harm proxy: changed B8-survive→successor-die or changed unknown selected/base outcome. These ordinary rows are conditioned on selected rescue lives, so this proxy is not a natural-deployment harm rate. The three heads share the same data, split, seed and update count.

Outputs include `results.json`, `results.csv`, per-config checkpoints and SHA sidecars, `checkpoints.tar`, `verified-ledger.json`, `timing.json`, and immutable per-config unit bundles. Checkpoints/results/timings carry `simulator-D`. All results are development evidence at most. This package does not select or export a Mac evaluation candidate automatically. Only one frozen SHA may later cross through a separate judge-reviewed registration and fresh Mac bank.

## Prepare and validate

The root-level `dispatch-manifest.json`, `sandbox-manifest.json`, and timing forecast are sealed historical packet-01 evidence. Prepare the corrected ready attempt under `packet-02/`; it does not turn the small local CPU smoke into an A100 duration forecast. It uses the owner-adopted 60-minute base / 120-minute effective runaway guard, which is a technical stop boundary and not a compute budget or performance estimate. [RUNAWAY_GUARD.json](RUNAWAY_GUARD.json) records that prospective replacement.

For another fresh packet, first create an empty ledger (or a ledger of already verified units):

```sh
./venv/bin/python -m research.sandbox_b_heads.resume empty-ledger \
  --data /absolute/registered/d1024-allowed.npz --b8 /absolute/B8.pth \
  --output /absolute/fresh/verified-ledger.json
./venv/bin/python -m research.sandbox_b_heads.package \
  --data /absolute/registered/d1024-allowed.npz --b8 /absolute/B8.pth \
  --ledger /absolute/fresh/verified-ledger.json \
  --output research/sandbox_b_heads/packet-02/dispatch-manifest.json \
  --output-dir /absolute/fresh/runtime-output \
  --sandbox-manifest research/sandbox_b_heads/packet-02/sandbox-manifest.json
```

Commit the manifest before preflight. Use the qualified broker's offline `--dry-run`, then:

```sh
./venv/bin/python -m tools.sandbox.preflight research/sandbox_b_heads/packet-02/sandbox-manifest.json
```

The transport-required unit field equals the same 120-minute technical liability at an 8-units/hour ceiling (16 units). It does not ration development credits or impose an earlier economic stop: an admitted calibrated rate must fit that ceiling, so the timeout remains the first guard. The broker checks calibrated prices, capacity, assignment isolation and coexistence at dispatch. Claude dispatches; these modules never call Colab.

The packet runs the entire finite development sweep through the sandbox admission route in `tools/ops/dispatch.py`. It is a sandbox job, not a `probe` job: the committed ten-line manifest and sandbox preflight replace the family-canary receipt for this narrowly admitted route. It does not change `research/colab_cli`, and it grants no registered-study authority. Real validation action selection runs on the requested device and feeds the reported metrics. This single-job packet does not use retained main-job chaining.

## Completion and resume

Each fixed allocation index 0–59 has an immutable tar unit containing checkpoint, report, internal hashes and full frozen identity. The fingerprint binds the source bytes, input/B8 hashes, salt, seed, updates, batch size, learning rate and smoke selection. Payloads publish atomically; SHA sidecars precede atomic completion markers. The job emits outcome-blind JSON progress telemetry with done/total, elapsed time, and ETA at start, at least every two minutes during a long configuration, and after every sealed unit export. The qualified exporter can pull completed units throughout the run. The terminal ledger is sealed once, so exporter polling never observes a mutated completed file.

After a **verified mechanical failure and verified stop**, preserve all original evidence and have the Mac build the missing-unit ledger from verified exports. This command reads completion identities and payload path/size/hash only; it never opens checkpoint/report contents:

```sh
./venv/bin/python -m research.sandbox_b_heads.resume build-ledger \
  --exports /absolute/verified-original-exports \
  --data /absolute/registered/d1024-allowed.npz --b8 /absolute/B8.pth \
  --output /absolute/fresh/completion-ledger.json
```

Pass that ledger to `package`. It binds and uploads every verified bundle, rewrites runtime paths to portable basenames, verifies each bundle before restoring its checkpoint/report, skips exactly verified configurations, and aggregates the original and completed units into the same 60-row table. Duplicate IDs, foreign namespaces/allocations/recipes, missing or corrupt payloads refuse. The Mac ledger builder refuses duplicate allocation indices rather than accepting the same unit twice.

Mechanical triggers are runtime/lease termination, transport interruption, or incomplete export after a stopped attempt with missing configuration units; numerical or outcome-based selection is not a trigger. At most two completion attempts follow the original. Every attempt gets a fresh manifest, job/dispatch identity, output directory and receipt; the same registered 60 units and per-attempt 2× envelope remain fixed. No automatic retry or allocation occurs in this package. Configuration boundaries are the resume unit: an interrupted unverified configuration restarts from its original seed; completed configurations are never recomputed.
