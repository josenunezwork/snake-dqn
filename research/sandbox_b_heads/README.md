# Sandbox B: frozen-B8 successor head/loss sweep

This package is Tier-1 development work under the adopted Standing A100 Development
Sandbox. It takes only a registry-filtered D1024 NPZ projection, verifies its exact
row/life/parent membership before outcome arrays are opened, and uses the fresh
parent split `a100-dev/sandbox-b/split-v1`.

It sweeps three residual heads (F4-compatible linear, length/margin-gated two-layer
MLP, and cosine-normalized) over KL weights 1/2/4/8/16 and survivor-reference
pair-loss weights 1/2/4/8. It exports one SHA sidecar per configuration, a result
table with dev-validation rescue retention and ordinary-harm proxy, and an atomic
resumable verified-unit ledger. Results are development labels only and cannot gate
or feed F5.

Prepare an admitted projection with `tools.sandbox.registry.prepare_input`, then:

```bash
./venv/bin/python -m research.sandbox_b_heads.package --data INPUT.npz \
  --b8 B8.pth --ledger verified-ledger.json --estimated-minutes MEASURED \
  --output dispatch-manifest.json --sandbox-manifest sandbox-manifest.json
./venv/bin/python -m tools.sandbox.preflight research/sandbox_b_heads/sandbox-manifest.json
```

The resulting dispatch manifest is only a packet. Claude/operator dispatches it via
the qualified supervisor and `tools/ops/dispatch.sh`; this package does not call Colab.
