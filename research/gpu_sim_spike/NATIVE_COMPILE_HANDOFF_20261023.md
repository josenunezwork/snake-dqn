# Native CUDA compile gate

The sparse canary at pin `3ab2ef04` failed nvcc overload resolution before any
exactness result. `vector_geometry.cu` now uses project-specific `vector_rn_*`
names for all five double helpers. Each helper still calls the corresponding
`__d*_rn` intrinsic; operation order, final `__double2float_rn`, and production
`--fmad=false --prec-div=true --prec-sqrt=true` remain unchanged. The other four
native sources have no matching helper-name collisions.

`native_compile.py` inventories every `.cu`, its `.cpp` twin and Python loader.
It extracts each production `load()` name and flags with a fail-closed AST check,
then builds, links and loads all five extensions in a fresh cache for both
A100 (`sm_80`) and L4 (`sm_89`). It never calls extension methods, launches
kernels, runs a model, or evaluates gameplay. Its host-compiler `device=cpu`
declaration is deliberate; the L4 allocation supplies the CUDA toolkit.
The result is compile proof only, never A100 parity or performance proof.

Prepare the short L4 job from a clean published source pin on the qualified
transport tree:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.gpu_sim_spike.native_compile_pack \
  --commit FULL_PUBLISHED_PIN --artifact-dir /absolute/fresh-pack/native-compile
```

The command prints the exact `tools/ops/dispatch.sh` invocation. The effective
20-minute / 1.2-unit cap includes 0.2 units reserve, at a 3 u/hour admission
ceiling; base limits are 10 minutes / 0.6 units / 0.1 reserve (owner's 2x rule).
The five-minute estimate is a provisional forecast, not a measured CUDA build.
Charge it inside the sprint's remaining allowance. No automatic workload retries.

A100 GPU-sprint canaries (including an allowed A100 substitution) now require a
SHA-bound compile supervisor receipt before allocation side effects. Verification
requires same-pin PASS, verified stop, a bound saved manifest and successful
terminal, exported payload/sidecar inventory, exact source hashes, production
flags and both architecture targets. Missing/stale/tampered proof refuses.
No transport code or qualification registry is changed by this gate.

`sparse_pack` without `--native-compile-receipt` prepares pending-compile manifests
and dispatch helpers. `--validate` requires the proof. `dispatch-canary.sh` reads
`native-compile/output/receipt.json`, binds it and creates the dispatchable canary
only after verification. Once prepared, the helper recovers a custom proof path
from the canary manifest. For a verified exact accounting-allocation retry use
`NATIVE_COMPILE_RECEIPT=/absolute/native-compile/output-retry-1/receipt.json` on
the helper invocation. Existing manifests and failed attempts remain immutable.
The main job still requires the separate same-pin sparse A100 canary PASS.

Mac CPU checks and Codex-V light passed; this is not a native compile PASS.
The final artifact package's dispatch instructions identify the published frozen
runtime pin and exact compile/sparse commands. Main retains its current transport;
the frozen runtime uses the exact already-qualified transport tree.
