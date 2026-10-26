# REJECT F5-W implementation 22e86986

**REJECT implementation acceptance. Two independent P1 defects prevent the corrected L4 capture from completing. Claude must dispatch nothing from final-04.** This decision concerns operational implementation, not W selection or the registered scientific contrast. Obtain a repaired source pin, immutable sibling packet and independent implementation acceptance before capture. This document is not ACCEPT_F5_FIT, operational acceptance, a registration amendment, or release permission.

Astra, independent implementation reviewer. Checkout `/Users/josenunez/Projects/ml/snake-dqn-f5w`, branch `sol-f5-w-build-20261025`; HEAD `b26f143eac48a404da6cd232a99d4df014289887`; implementation pin `22e869686f3132a128607ce32f6a8506df474be2`. Project working tree was clean before and after review. Code remained read-only. No project commits, dispatch, Colab calls, tmux, installations or scientific executions occurred. Synthetic test Git fixtures and temporary artifacts were confined to the authorized Astra directory. No label/outcome payload, training archive, checkpoint payload, D4096 output or `t34-round0-study-01/` was opened. References to such files in governance were not followed.

## Evidence identity

Recomputed the packet, bundle and dry-run hashes:

- `F5_W_PREFIT_20261025/final-04/REVIEW_PACKET.json`: `974c094996c6f554df52ed5c851337b4e382e8eb22a28c391eb5ed4dfa3a7cfd`.
- `REVIEW_BUNDLE_v2.json`: `3e6e95ff55c9b79b1dadaa5458a3518c2e6ece1e4d7909bc1b3d5ccb2ee7f5c1`.
- `dry-run.json`: `50739532da39c04d062f0650ce484500220890a3cc134f1f0ce8aaf5d328a8c9`.
- Adopted assignment amendment: `a1dc92ffb8fa341fb51738a56fe0097d75e15b0aff7194940c12979adbb04d62`.
- Prior rejection: `b672b7b62bd7744e3226aabbd516a8a3a562b6d57f97c128b16c7f130d1e5aed`.
- Frozen environment schema: `67a21676ec05d7ad32945a10d6c85c9d0148a2a44254d20cc5e4cf1ee5c923d2`.
- Frozen calibration schema: `102c78435abcf49c57c6400b0a6983d07117766c84a09d3b19d28debd011f33c`.

Independently verified all 28 metadata artifact hashes/sizes in the v2 bundle, and all 192 packet source hashes against both working files and the exact implementation Git object. HEAD has no changes to the implementation's F5 source closure, src or configs relative to that pin. The bundle's own declaration of offline preparation is consistent: actual CUDA evidence and measured feasibility are pending.

## Blocking defects

### P1 — Capture requires a distribution absent from the frozen environment

`research/bl_residual/f5_environment.py:75` puts `scipy` in mandatory NUMERIC_PACKAGES. At `:931-933`, capture constructs the numeric projection with `all_packages[name]` for every mandatory name. The frozen `requirements-f5-l4-cu126.lock` contains 69 named packages and **no SciPy**. The launcher creates an isolated venv with `uv venv --seed` and installs only the frozen lock (`f5_colab.sh:18-24`), without system-site-packages. Therefore the actual locked package set raises **KeyError('scipy')** during capture, before validation or CUDA qualification.

Independently enumerated the lock and reproduced this error through the actual `capture_environment()` function with only hardware/build I/O seams replaced. This is not a provider run. Even injecting ambient SciPy would not establish the required locked-wheel closure: `validate_environment()` also requires every numeric package in `locked_wheels` (`f5_environment.py:438-440`), while `_wheel_manifest()` requires exactly the existing lock closure (`:819-825`). Existing synthetic receipts invent a SciPy package and wheel and consequently miss this contradiction.

Required repair: reconcile the exhaustive receipt with the **actual adopted locked environment**, explicitly and prospectively representing an absent optional package if that is the lawful environment. Do not silently add an unregistered dependency or omit evidence of an installed dependency. Any proposed build/lock change requires its separate pre-data authority. Test capture against the exact lock package closure, including absent SciPy, without substituting a hand-built complete receipt. Freeze the corrected schema and digest in a new packet/control; preserve final-04 and all adopted bytes.

Reproduction receipt: `F5_IMPLEMENTATION_REVIEW_22e86986/lock-evidence.json` plus SHA sidecar. The first local reproduction omitted the Linux affinity seam and stopped on macOS AttributeError; the corrected host seam then reached and reproduced the source KeyError. This did not touch project files.

### P1 — GPU-memory consistency check compares incompatible units

`f5_environment.py:834-835` queries `nvidia-smi memory.total` with `nounits`, retaining the numeric MiB value. `capture_environment()` records `int(device.total_memory)` from PyTorch in **bytes** (`:910`). The validator converts the SMI string to int and requires exact equality with that byte value (`:425-433`). A lawful real capture therefore refuses with **F5 environment parsed GPU capability/memory differs** once the missing-SciPy defect is resolved.

Independently reproduced both the validator failure and the actual capture-function failure using representative metadata: SMI `23034` MiB versus PyTorch `23034 * 1024 * 1024` bytes. Existing environment and handoff tests use `23034` or `24576` for BOTH fields, so they cannot catch the dimensional error. These are synthetic demonstrations, not measured hardware values.

Required repair: declare and retain units for both raw observations, compare each invariant across assignments, and apply a documented unit-aware consistency rule. Do not discard the driver-memory field, weaken invariant equality, or assume unreviewed byte-for-byte equality between APIs that can have rounding/accounting differences. Test realistic API units through capture as well as comparison, preserving original raw query bytes and the canonical mapping.

Reproduction receipt: `F5_IMPLEMENTATION_REVIEW_22e86986/independent-evidence.json` plus SHA sidecar.

## Independently verified repair scope and limits

**Schema and drift protection.** The source-pinned JSON and `policy_binding()` now bind the adopted amendment and schema digest into controls/comparison reports. Exact top-level/nested maps, leaf classification, and manual typed validators reject unknown, missing, malformed and overlapping classifications. The tested legitimate cross-assignment receipt changes UUID, embedded driver UUID, uname node, affinity indices and installation roots while retaining canonical identity. Same-assignment mode requires the full raw receipt digest. METADATA/WHEEL and library bytes remain invariant; RECORD, direct URL and generated shebang normalization retain auditable raw evidence. The installed package closure and locked wheels are compared in addition to versions. The two capture defects above prevent acceptance of this otherwise improved architecture against the real frozen environment.

**C/W identity.** `f5_job.py` admits source/build invariants, captures assignment evidence, observes rather than repairs numerical settings at boundaries, and compares full raw identity before C, after C and after W. Both arms run in one process/assignment. Resume snapshots bind the raw assignment digest; foreign-assignment resume refuses. Synthetic handoff/continuation tests passed. Actual CUDA continuation remains unproved here.

**Science bytes.** I independently ran the Git diff and compared bytes from `365612540f16b068c7748eb166e150fcb6750a9e`, `22e869686f3132a128607ce32f6a8506df474be2`, and working files. All seven files below are byte-identical; this conclusion does not rely on Sol's identity receipt:

| File under research/bl_residual/ | SHA256 |
|---|---|
| f5_fit.py | 5559f96c83741bcb2e97025a30eed457ca86e3f5556edab7e11ff5ce50440460 |
| f5_data.py | 8cda84310ace854b1143239bb80711b86151a71327c53b4b345a71a248ee1f9f |
| f5_protocol.py | 34ba9233feebbe81daac76a7d9753a67a037e5039919f4fb88287a3815eb5ccc |
| f5_rules.py | 41491605f7cbac6a5971070c9b1580a0a137a66bcaabb945ba9a1ec0bca70174 |
| f5_timing.py | a449048c99a840b16181ac2e6939834374474e7c8b6e898f78089d741852b983 |
| fit.py | 33558d90f43553c8dd54da2b1a5c0ceb7e9faa45d41e1757e7ec246c4c631ec9 |
| model.py | e00d30e0fd2e752bcd634676a364e0750f316c9ab3fb47251edcd50d87f66c04 |

Thus this repair does not change W/C weighting, denominator, masks, optimizer, common sampler/plan, seed or registered rules in those files. Synthetic regression tests corroborate this; no scientific inputs were decoded.

**Genuine CUDA path exists in code; actual proof remains pending.** `f5_qualification.py:101-137,451-487` builds production GridBatchSim/bv8 observations at all twelve canonical sealed timing seeds, performs a real B8 fp32 CUDA `(1,6)` forward and masked CUDA Tensor.argmax, then compares CPU Q values/actions with explicit tolerances. This replaces CPU-only preparation as the family proof path. `f5_colab.sh:26-33` injects the unchanged real observer into the custom venv. It does not fabricate observer rows. Capture-qualification performs capture and then a separate transport-only CUDA probe; prepare-qualification materializes the CPU-generator plan and then separately probes CUDA. Stream-specific exact roots/full rollouts belong to the full timing path; generic seed-keyed transport observations alone do not qualify those stream workloads or feasibility.

`bind_qualification_inventory()` (`f5_package.py:330-397`) binds exact eventual science upload destinations/hashes/sizes and immutable sidecars after final acceptance files exist. The unchanged real canary (`research/colab_cli/canary.py:280-398`) rederives evidence from actual transfers, observer events, terminal scanner inventory, recovered export hashes and verified release. Qualification emits explicitly unusable timing envelopes at producer paths via real immutable export helpers. Their recovery demonstrates the mechanism, not scientific checkpoint validity, representative full-state export cost or measured production cache feasibility. Earlier bootstrap proof cannot cover final files that do not yet exist.

Full timing supplies registered root/branch/natural/Watch workload measurements and a separate source-derived synthetic C/W optimizer calibration with serialized continuation. It explicitly leaves production cache/export cost and Mac companion timing HOLD. No real CUDA observer receipt, numerical parity result, continuation result, transport timing or release was produced or independently validated by this review.

**150/300 seconds and cumulative clocks.** Byte-identical `f5_timing.py` and unchanged canary admission retain inclusive transfer <=150 s and whole staging <=300 s, with finite clocks and the private inventory ceiling. Boundary/overrun tests passed. No D1024 transfer exception is admitted. All capture, prepare, timing, final inventory proof and mechanical completion attempts consume the original 40-minute qualification allowance inside the 240-minute paired-fit allowance; 10 minutes capture plus remaining substeps collectively 30 plus 200 reserved fit are not fresh allowances for each new manifest. Actual elapsed receipts and complete L4/Mac forecasts remain prerequisites.

**One look and publication.** The unchanged protocol creates once-only source/selection/plan-bound claims and refuses duplicate/foreign units and excessive completion attempts. The v2 registration R10a requires prior-stage disposition, complete ledger, clocks/forecasts and immutable hashes before each stage, and consumption of its one-look token only after every required unit is sealed. Pure analyzer/protocol tests do not establish a production S2/S3/S4 orchestration runner. I did not take any scientific look or repeat the diagnostic selector.

## Independent tests

Ran `./venv/bin/python -m pytest` at nice 15, one process, CPU, OMP/OMP_THREAD_LIMIT/MKL/OpenBLAS/Numba/vecLib threads all 1, bytecode and pytest cache disabled. macOS launcher used torch/lib in DYLD_LIBRARY_PATH and execve into the interpreter. TMPDIR and basetemp were confined to `F5_IMPLEMENTATION_REVIEW_22e86986/` under the authorized Astra root.

```text
-m pytest -q -n 0 -p no:cacheprovider -o addopts= --tb=short
--basetemp=<Astra-root>/F5_IMPLEMENTATION_REVIEW_22e86986/pytest
  tests/test_bl_residual_f5_environment.py
  tests/test_bl_residual_f5_handoff.py
  tests/test_bl_residual_f5_calibration.py
  tests/test_bl_residual_f5_job.py
  tests/test_bl_residual_f5_package.py
  tests/test_bl_residual_f5_qualification.py
  tests/test_bl_residual_f5_broker.py
  tests/test_bl_residual_f5_fit.py
  tests/test_bl_residual_f5_data.py
  tests/test_bl_residual_f5_rules.py
  tests/test_bl_residual_f5_timing.py
  tests/test_bl_residual_f5_protocol.py
  tests/test_bl_residual_fit.py
  tests/test_bl_residual_f4_fit.py
  tests/test_bl_residual_f4_fit_binding.py
  tests/test_bl_residual_factory_masks.py
  tests/test_f5_environment_amendment.py
  tests/test_colab_cli_canary.py
  tests/test_colab_cli_canary_observer.py
```

**283 passed, 3 skipped in 18.72 s; exit 0.** Skips are the real-hardware qualification, observer and CUDA calibration tests. Passing synthetic tests do not clear the two reproduced capture defects. A separate read-only reviewer independently found the SciPy/lock contradiction and checked CUDA/transport admission; the root verified the decisive code and reproduced both blockers.

## Exact next step for Claude

**Now: return this rejection to Sol. Do not dispatch the current `bl-f5-w-capture-22e86968-01` or any final-04 draft.** In addition to the code defects, the capture draft has `scientific_admission:false` and `budget:null`; `tools/ops/dispatch.py:1078-1079` refuses it. There is no truthful currently runnable corrected capture command. Do not flip the draft flag or run bare Colab/CLI commands to bypass this gate.

Require the two bounded repairs, regression proof through actual capture with exact locked package closure/realistic units, a newly pinned immutable packet, and independent re-review. After that acceptance, Sol/Claude must seal a genuinely operational capture job with the repaired commit, accepted source/schema/bootstrap pins, lawful B8/fixture uploads, current broker pricing/accounting and fresh output/state. Preserve the existing drafts.

For an exact future delivery/dispatch contract, reserve the following **currently absent** create-only operational package path:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_W_CAPTURE_AFTER_22e86986_REPAIR_01/`.
The operational job delivered there must be `environment-capture.json`, job ID `bl-f5-w-capture-repair-22e86986-01`, mode **capture-qualification**, `F5_DEVICE=cuda:0`, one L4, >=6 vCPUs, timeout no greater than its remaining capture/series allowance, and the genuine disjoint CUDA transport probe enabled. `DISPATCH.txt` must identify the new acceptance and contain no contradictory HOLD instruction. These files do not exist now; this review neither creates nor accepts them. **Only after their publication and independent acceptance**, Claude's exact combined capture-plus-CUDA-canary wrapper command is:

```bash
/Users/josenunez/Projects/ml/snake-dqn-f5w/tools/ops/dispatch.sh \
  bl-f5-w-capture-repair-22e86986-01 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_W_CAPTURE_AFTER_22e86986_REPAIR_01/environment-capture.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-f5w \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-f5w \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/bl-f5-w-capture-repair-22e86986-01 \
  --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_W_CAPTURE_AFTER_22e86986_REPAIR_01/DISPATCH.txt
```

This is one combined job: raw capture followed by `f5_qualification --transport-only --bootstrap`, not a CPU preparation job presented as a CUDA canary. No scientific fit or full-timing admission follows automatically.

## Publish before any subsequent scientific read

1. Preserve and verify externally verifiable publication of the existing adoption/selection chain. Publish this rejection and the subsequent repaired implementation acceptance/new source and schema pins before the affected capture.
2. Following accepted capture, publish the complete raw environment capture and invariant/provenance comparison evidence, hash-verified exports, genuine CUDA observer/parity/scanner evidence, measured transfer/staging clocks and verified runtime disposition. Keep timing-only calibration distinct from science. A CLI exit code alone is insufficient.
3. Publish the actual L4-materialized 2,000x256 ordered plan and its Mac verification, all typed initial/final generator states, corpus/source/environment/allocation/control bindings, and the bound initially empty paired ledger **before either fit**. A synthetic plan is not this plan.
4. Publish current-source full L4 plus Mac timing/feasibility proof, production cache/export forecasts, cumulative elapsed/remaining envelopes, independent actual-plan/operational fit acceptance and exact final science upload inventory. Run and verify the final inventory canary after those actual files exist; earlier bootstrap proof cannot cover them. Only the separately accepted packet may fit C then W in one assignment.
5. Before each S2/S3/S4 numerical read, publish/seal all required immutable units and complete Mac-verified ledger, prior-stage disposition, source/input/plan/selection bindings, stage claim, measured forecast and remaining envelope. Consume the sole stage look token after completion and before the registered Mac aggregate reads outcomes. Remote partial summaries remain sealed. No repeat look, refit, bank substitution or cap reset follows a scientific failure.

All resulting F5 conclusions remain exploratory; Mac strict/serving qualification and owner release authority are unchanged.

Created UTC: 2026-10-26T05:23:52.997801+00:00
