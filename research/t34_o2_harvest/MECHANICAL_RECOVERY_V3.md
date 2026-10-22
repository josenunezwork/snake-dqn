# O2 V2 mechanical transport recovery — 2026-10-22

Status: **PREPARED_NOT_DISPATCHED**. V3 identifies a transport attempt, not a
new study. The V2 registration, rules, bank, source pin, caps, CPU workers,
excluded runtime timing admission, and sole scientific look are unchanged.
No Colab command, account query, allocation, study run, or analysis was performed.

## Root causes

1. **Chunk 00: shared initial-upload deadline.** Its preserved receipt records
   36.021 seconds uploading/verifying bootstrap, 160.833 seconds for source,
   and 127.146 seconds for the 1,195,740-byte registration. The shared
   300-second deadline was exhausted before the 2,409-byte catalog could
   upload; its failed call lasted only 0.000159 seconds. The error named the
   next file rather than the expensive preceding transfers. The old saved-rate
   forecast was optimistic for this connection; it was never execution proof.
2. **Chunks 01/02: CUDA masked before shared bootstrap.** The manifest injected
   `CUDA_VISIBLE_DEVICES=""` before `research/colab_cli/bootstrap.sh`. That script
   detected L4 through `nvidia-smi`, installed the CUDA wheel, and failed its
   first `torch.cuda.is_available()` assertion. The CPU-only scientific launcher
   had not run. The mask belongs in the existing `launch_v2.sh`, after bootstrap.

The supplied V2 logs and receipts under the artifact root remain unchanged.
Both remote-failure receipts report `INCOMPLETE`, no exports, and verified stop.
These are mechanical failures, not scientific outcomes.

## Repair and identity

All changes are additive files inside `research/t34_o2_harvest`:

- `make_manifest_v3.py`: reuses the unchanged V2 builder in its exact source
  snapshot, keeps runtime/script identity, omits the early CUDA mask, and uses
  qualified hash/size-bound public URL inputs for registration and catalog.
  Only the 896-byte timing receipt remains a local Contents upload.
- `bootstrap_preflight_v3.py`: executes the pinned bootstrap and launcher with
  the manifest environment in a simulated runtime filesystem. It relocates only
  the Mac-inaccessible `/content/snake-cli/*` root guard. Installers, GPU/Torch
  hardware, public checkpoints and the scientific runner are offline probes;
  real Bash/Python execute the original bootstrap, CPU launcher and export code.
  The original mask reproduces `AssertionError`; the O2 boundary reports an
  explicit CUDA/expected-wheel diagnostic. A wrong-wheel probe also refuses.
- `frozen_sources_v3.py`: validates the original registration and exact Git
  archive, including qualified transport bytes. Incoming main changed a frozen
  preparation dependency (`t3_rollout_labels_20261015/bank_engine.py`), so current
  main cannot stand in for the registered runtime. No current file is reverted.
- `analyze_v3.py`: sibling admission for the corrected bootstrap/CPU boundary.
  Actual analysis constructs its adapter directly from hash-verified, pinned
  `analyze_v2.py`, changing only that placement block and an internal function
  name. All science and dependencies execute inside the frozen snapshot. Both
  entry boundaries reject registration copies before side effects. The original
  canonical `LOOK_O2_V2.json` remains the sole create-only look marker.
- `prepare_v3.py`: create-only generation, local validation, bootstrap simulation,
  HTTPS input verification, and full Z-9 upload qualification for ten chunks.
  It preserves old DISPATCH bytes and sidecars before updating the requested pointer.
- `test_transport_v3.py`: bootstrap regression/reproduction, explicit diagnostics,
  analyzer AST equivalence, ten-chunk identity, isolated scientific dependencies,
  and copied-registration/sole-look protection.

The literal bare assertions in `research/colab_cli/bootstrap.sh` remain unchanged
because the owner explicitly prohibited editing `research/colab_cli`. The new
O2 admission/preflight supplies explicit diagnostics without changing those
qualified transport bytes or disabling assertions.

| Identity | Value |
|---|---|
| Pulled baseline | `cda1cf5c5099eb978350f9b756905571bb537210` |
| Unchanged runtime commit | `f19d3f39050adfedc6eaef001a3364318fbb28bc` |
| Registration publication | `040a6fcd853d04a57bc36da47dcbebca65c62023` |
| Registration SHA-256 | `82c1986ff42c7a98161b1f6ab657db0a61f82ab527cdd8bba4620410b698ce88` |
| Catalog SHA-256 | `736f7b4e7730f760b8b81db00772d14abcf84c74b261a8dc7630732fdd383a10` |
| V3 validation SHA-256 | `33534ed3bd17670ad72a96fdb9aa89b43c8fa59b78d77f8cbc3b2d5a88626ad3` |
| Updated DISPATCH SHA-256 | `2d3bb0d6955490f95df96ebd66fef85813a743c5274903fbaf9e5e14ddc3d1f8` |

## Prepared publications

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o2-20261021`.

- `v3-manifests/chunk-00.json` through `chunk-09.json`, fresh outputs
  `v3-chunk-00` through `v3-chunk-09`; no output directory exists.
- `v3-manifest-validation.json`, ten `v3-upload-qualification-NN.json`
  receipts, and `v3-public-input-preflight.json`, with SHA/completion sidecars.
- Root `DISPATCH.txt` and `v3-manifests/DISPATCH.txt` contain identical new
  local validation, future dispatch, and V3 analysis commands.
- Prior V2 `DISPATCH.txt` and its original sidecars are preserved under
  `v2-dispatch-preserved/`; V1 and V2 execution evidence is retained.

All ten manifests pass the supervisor's local validator and exact source
preparation/full upload qualification. Normal initial upload falls from
1,214,804 to **16,655 bytes** (including bootstrap/source transport files),
with one local input. Saved-rate forecast: **65.374398 seconds**;
fallback plus fetch bound: **211.037206 seconds**, within the unchanged
300-second cap. Actual Mac HTTPS verification retrieved the registration in
0.587687 seconds and catalog in 0.395339 seconds. These are input-transfer
measurements and prospective forecasts, not Colab runtime evidence.

The eight 130-world and two 110-world chunks still cover all 1,260 worlds.
Ten CPU workers, L4 placement, estimates 231/199 minutes, timeouts 462/398
minutes, hard caps 15.4/13.266667 units and aggregate 149.733333 units remain
unchanged. Failed-attempt spend remains chargeable; new output directories do
not create new funding. Future operator admission must refresh account rates,
remaining study headroom and qualification; account assignments remain at most
three. No automatic retry and no dispatch authority is added here.

## Verification

Use the project venv at `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`
with `SNAKE_DQN_DEVICE=cpu` and OMP/OpenBLAS/MKL thread caps of 1.

```bash
python -m pytest -q research/t34_o2_harvest research/bl_upload/test_qualification.py tests/test_colab_cli_source.py tests/test_colab_cli_supervisor.py -k 'not slow'
python -m pytest -q research/t34_o2_harvest/test_transport_v3.py
python -m black --check --line-length 100 research/t34_o2_harvest/*v3.py
python -m isort --check-only --profile black research/t34_o2_harvest/*v3.py
python -m flake8 research/t34_o2_harvest/*v3.py
git diff --check
```

Independent read-only review found and verified the fixes for scientific
dependency drift and copied-registration look bypass. No remaining actionable
finding. Frozen-source tests prove unchanged scientific ASTs, not efficacy.
Bootstrap probes do not qualify real GPU dependencies or provider behavior.

Broader O2/shared-upload/source/supervisor regression: **169 passed in
575.66 seconds**. Final focused regression: **8 passed in 59.99 seconds**. Black, isort,
flake8 and diff whitespace checks pass. The real local bootstrap-preflight CLI
also passes using the generated chunk-00 manifest, reaching the CPU worker
probe and original export path.

No protected package, registration, bank, checkpoint, serving pin or active
tmux session was changed. The prohibited sealed artifact directory was never
opened. Release and serving remain owner-only.
