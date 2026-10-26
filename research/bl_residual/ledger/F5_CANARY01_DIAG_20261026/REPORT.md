# F5 family canary 01: offline environment-refusal diagnosis

**Verdict: HOLD. The canary is operationally INCOMPLETE. Exact differing canonical keys cannot be recovered from the saved evidence.** Host CPU/ISA/SIMD or GPU driver/memory variation is the leading explanation; no genuine source-commit, checkout-path, or installation-path bug was found. No schema or invariant was changed.

This report and its supporting receipts are create-only. Work was offline except authorized Git publication: no dispatch, Colab/provider call, GPU allocation, scientific label/outcome inspection, or retry occurred. The accepted once-only dispatch remains consumed.

## Publication and source identity

- Worktree: `/Users/josenunez/Projects/ml/snake-dqn-f5envdiag`; branch `sol-f5-canary-envdiag-20261026`. This clean worktree/branch was already supplied at `4e829370ed7dbcd6bd93539cb041f7c174ecb96b`; it was reused without modifying other worktrees.
- Evidence commit: `0c64e8e52cc6bca9674d7c2e4136f44d0fa575eb`, published to `origin/main` and the task branch.
- Diagnostic fix commit: `d313736f41e882b31eae85e7c2e7469c02b4a308`, published to `origin/main` and the task branch.
- Publication index: `research/bl_residual/ledger/F5_FAMILY_CANARY_57a89e7d_20261026_01/OUTPUT_PUBLICATION_01.json`, SHA256 `a74ea1f8f89e60b2220f7e438c2e1159a45a403840e5e968fd7f1e61dabed97b`.
- All 64 saved `output-01` files, including 28 polls, terminal, hardware, receipt, accounting/transfer evidence and seven staging files, were copied byte-for-byte with SHA/completion sidecars. The root independently reverified all 64 against the originals. Existing preparation evidence was preserved; new state/account credential material was excluded.
- Original receipt SHA256: `21dbee1ef000d454c6d389bad2becca7a4c4ba9e7cdc7d0d39499ba886fdf7ea`. Original accepted source: `57a89e7df09687432228978ca162207051f9efdb`.

## What the saved evidence establishes

The broker receipt records `INCOMPLETE`, `CHILD_NONZERO_EXIT`, elapsed **650.3527150410227 seconds**, `stop_verified=true`, `cleanup_error=null`, no exports, and saved post-stop accounting with zero assignments/rate. The terminal records return code 1 and the cross-assignment invariant refusal. These are saved cleanup observations, not a fresh live account query.

The final poll traces the failure to `f5_qualification.run -> f5_job._environment_evidence -> f5_environment.compare_environment`. The qualification process started environment capture at 2.653 seconds, entered comparison at 68.154 seconds, and failed at 77.702 seconds, with **zero fixtures completed**. Thus the 650-second broker elapsed time is not 650 seconds of comparison or fixture execution. Failure happened before the initial assignment boundary and before CUDA model loading, parity, timing-unit publication, or skip proof.

`hardware.json` contains only NVIDIA L4 / capability 8.9 / ready. Polls and the receipt contain no actual environment receipt, driver version, CPU identity or canonical diff. Empty terminal exports and scanner inventory agree. No saved file can supply the missing actual values.

The offline audit revalidated capture-04 and capture-05 under the frozen comparator and found **zero canonical differences**, canonical SHA256:

`5fb26f266ab4247e6d99f79ebe1ed70580ba99f356084a706bc88f26f9d2d853`

The canary control's environment object equals capture-05 exactly. Supporting bindings and all 64 saved-file hashes are in `OFFLINE_ANALYSIS.json` beside this report.

## Likely differing fields, with limits

The leading group is legitimate host variation in fields that the adopted schema currently treats as exact invariants. The evidence cannot rank CPU variation against driver variation or name a confirmed key.

| Candidate field(s) | Bound capture-05 value / reason it could differ |
| --- | --- |
| `cpu_isa.model name`, `cpu_isa.flags` | Intel(R) Xeon(R) CPU @ 2.20GHz, with a fixed full flags string. CPU model/ISA can vary while the accelerator remains L4. |
| `numpy_blas_lapack_build`, `torch_build` | NumPy includes detected SIMD extensions; Torch reports `CPU capability usage: AVX512`. These are host-sensitive observations despite their build-oriented names. A CPU difference can therefore produce multiple canonical differences. |
| `gpu_driver_query.driver_version`, `gpu_driver_query_raw` | Reference driver `580.82.07`. The raw CSV retains the driver version after only UUID normalization. |
| `gpu_driver_query.memory_total`, `gpu_memory` | Reference 23,034 MiB from SMI; 23,659,151,360 bytes from Torch. Their physical-consistency check permits different source quantities, but cross-assignment equality remains exact. |
| `cpu_count`, `platform`, `os_kernel.release/version/processor` | Reference 12 CPUs, Linux 6.6.122+, kernel version `#1 SMP Thu Apr 30 18:17:14 UTC 2026`, x86_64. A different host/image could differ. Lower-priority possibilities with no canary values. |
| `installed_packages.*`, metadata/content hashes, `installed_versions_sha256` | Installed closure drift remains possible, but pinned wheels and matching seed-version logs make it weaker than host variation. No canary package-content receipt was exported. |

The GPU name and capability match the reference in saved hardware evidence. GPU UUID, host node and CPU-affinity indices are already assignment-local; changes in those values alone cannot explain this refusal.

## Source, paths, packages and the two interpreter environments

`f5_environment.py` and `f5_job.py` at capture-05 source `e25f889b` and canary source `57a89e7d` are identical Git blobs. `f5_colab.sh` differs in later qualification/canary routing and ledger arguments; its environment installation prefix is identical. The real canary traceback reached the same environment comparison implementation. Current diagnostic changes do not modify that canonicalization.

No source commit, repo path, input-package path or uploaded checkpoint/archive identity directly enters `canonical_environment`. Those identities are separately enforced by control/source admission. `Path(__file__)` locates the frozen schema/lock/descriptor, whose **content hashes**, rather than paths, are captured. Uploading a different set of workload inputs does not itself change the installed distribution closure.

Installation roots, wheelhouse URLs, generated root-bound bin shebangs and corresponding RECORD rows already use the reviewed canonical mappings, while raw observations are retained. Generation roots use the same 32-character shape. METADATA/WHEEL and ordinary installed bytes remain exact. A hidden root leak is not mathematically excluded without the actual canary receipt, but no reproducible path-dependence defect was found and no path normalization was broadened.

The canary source `.pth` adds the repo and observer hook, but it is not a distribution RECORD member and does not create an installed distribution. The observer wraps action-selection functions, not environment capture/metadata APIs; it does not explain an observed canonical field difference.

Both captures contain 70 installed distributions and 69 locked wheels. The extra package is uv-seeded **pip 26.2.2**. `uv venv --seed` does not lock pip, which is a prospective reproducibility concern, but `poll-000010.json` explicitly shows the canary's F5 venv used **CPython 3.12.3 and pip 26.2.2** as well. Seed-version drift is not supported by this run's logs; byte/content drift remains unobservable. The generic supervisor bootstrap also created a separate Python 3.12.15 venv. F5 then explicitly created its pinned Python 3.12.3 venv; the generic interpreter version is not the F5 receipt's Python version.

## Implemented diagnostic and admission safety

`f5_environment.compare_environment` now raises `F5CrossAssignmentRefusal`, a subclass of the existing refusal, with the original error text. Only the existing canonical-inequality branch adds diagnostic data. Validation, canonical mappings, same-assignment comparisons, frozen schema bytes and success report are unchanged.

The diagnostic includes sorted differing canonical keys as escaped JSON pointers, original raw and canonical object hashes, and the actual captured fields from that same observation. There is no recapture. Lists remain exact compared values. Missing/different package keys and nested content/metadata hashes are represented explicitly.

The exported nested raw receipt is marked `capture_status=REFUSED` and gains refusal metadata; all other observed fields are retained. The enclosing diagnostic has its own schema and `admissible=false`, `scientific_use_authorized=false`, `fit_usable=false`. Both wrapper and nested raw receipt are rejected by the actual environment validators/comparator, preventing use as a reference, timing unit or fit evidence. The original raw-object hash is calculated before refusal marking.

Qualification/canary mismatch output is:

`output/environment/cross-assignment-refusal.json`

Prepare/fit mismatch output, only within a fresh owned output, is:

`<operation-output>/environment-cross-assignment-refusal.json`

The existing create-only exporter writes payload, SHA sidecar and completion marker last. The unchanged runtime scanner discovers the completed diagnostic on a nonzero child exit; the unchanged supervisor collects it before classifying terminal failure. No successful qualification receipt or completed timing unit is emitted on this refusal. Existing prepare output is rejected before capture; existing fit output receives no new diagnostics, and its verified-unit skip path remains intact.

## Verification

| Check | Result |
| --- | --- |
| F5 test glob, CPU, two workers | 591 passed, 3 CUDA-only skips; 22.75 s |
| Final root diagnostics/job/resume subset | 26 passed; 1.97 s |
| Final independent review subset | 38 passed |
| Essential runtime/supervisor/F5-transfer subset | 16 passed; 51.31 s, including failed-child export collection |
| Ops test glob | 292 passed, 2 failed; 54.99 s |
| Reproduction of both ops failures at baseline `4e829370` | Same 2 failures; 0.57 s |

The ops failures are pre-existing: `factory_interim_launcher.py:1` contains calendar literal `2026-10-23`, and a retained-chain test imports the unavailable `research.colab_cli.retention` module. They were reproduced in an isolated baseline worktree; no unrelated fixes were made. Broad Colab CLI collection also encounters three pre-existing missing-module import errors. The 592-item Colab CLI sweep was interrupted at a 10:26 elapsed bound after 72%; the 111-item supervisor/runtime sweep was stopped at about three minutes after 49 tests. Neither incomplete sweep is counted as a pass. No task-owned pytest processes remain. Exact commands and limits are in `VERIFICATION.log`; raw available test logs are preserved under `test-logs/`.

Root's essential 16-test command, run to completion:

```sh
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_colab_cli_runtime.py::test_read_state_returns_only_completed_hash_verified_exports tests/test_colab_cli_supervisor.py::test_finished_large_inventory_does_not_reenter_telemetry_per_export tests/test_colab_cli_f5_timing_transfer.py
```


Changed implementation/test files passed Black (line length 100), isort and flake8; `git diff --check` passed. Independent read-only review cleared the final implementation after the freshness correction; the final review reran 38 tests successfully. New tests cover all differing keys, JSON pointer escaping, retained observations, non-admissibility, success/no-export behavior, admission wrapping, real canary CLI refusal/scanner collection, create-only refusal on repeat, malformed/same-assignment behavior, existing-output preservation, and verified-fit skipping.

The CLI test isolates hardware/source/input admission with synthetic seams; it does not claim real CUDA execution. No live provider, transfer timing or CUDA parity qualification was performed. Full project CI was not run; verification targeted F5, transport and ops boundaries. Archived raw publication scripts are preserved as evidence rather than reformatted: the sealed evidence publisher has a Black-formatting caveat outside `make lint`'s product-source scope, with runtime execution and byte/hash checks passing.

Audit-tooling recovery: the first offline audit stopped before writing its JSON because it overrequired whole-shell-script equality. The create-only `analyze_v2.py` instead verifies equal environment/job blobs and the shared installation prefix; it also counts all 64 files including seven staging files. No source evidence or receipt was altered. The evidence publisher similarly verified its partial first-pass copies before completing its create-only publication after a self-copy refusal.

## Transport qualification and required Astra decisions

**`research/colab_cli` did not change.** Its Git subtree remains `5833e54e9c5396121cbbf99a5cd4e52006e564fd`, equal to the accepted transport. `tools/ops`, `f5_colab.sh`, the environment lock/descriptor and `f5_environment_schema_v5.json` also remain unchanged by this fix. There is no transport-code requalification requirement introduced by this patch. Existing exact-source package acceptance does not automatically authorize a new F5 source/control/package, and no replacement package, registry write, acceptance execution or dispatch was prepared/performed here.

Astra must decide prospectively:

1. Whether to authorize a fresh, separately sealed **outcome-blind diagnostic canary** at the new source, using the unchanged comparison policy and fresh output/state. The previous once-only authorization cannot be reused. The report/patch itself is not dispatch authority.
2. After observing actual differing keys, whether to retain exact host equality or adopt an explicit, field-specific schema/invariant amendment before affected scientific data. CPU/ISA, runtime SIMD/build strings, driver and memory identity must not be silently ignored. This run supplies no evidence to ratify a specific relaxation.
3. Whether to pin the seeded bootstrap pip identity in a prospective environment package/reference. Its version matched this run, so this is reproducibility hardening, not a demonstrated cause or a unilateral lock/schema change.
4. Review the successor's new source inventory/control hashes, reference compatibility, scope, required exports, freshness, timeout/resume rules and admission/authority bindings. Transport subtree identity can be retained, but the consumed `57a89e7d` family-canary approval cannot stand in for concrete successor acceptance.

No scientific fit, D4096 access, promotion, serving or release is authorized. No tmux session, checkpoint, serving pin, old evidence byte, immutable manifest, or sealed research registration was changed.
