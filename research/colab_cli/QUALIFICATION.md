# Colab CLI source qualification for T34

The recorded transport passed live `final-only` acceptance. **The accounting fix
in this branch changes these bytes: final-only acceptance must be rerun by the
operator before T34 dispatch.** Offline tests do not renew live qualification.
No live Colab, pod or tmux action was taken by this implementation task.

T34 manifests must run a colab_cli whose files hash-match this record; any
colab_cli change requires re-running only the final-only acceptance (~1.5 u,
operator runs it). Both T34 builders compare the complete local executable
inventory and the pinned commit inventory before producing a manifest and again
before supervisor validation. Shared broker admission repeats the check for
saved T34 manifests immediately before any allocation. Added, removed, or
modified Python/shell files refuse; documentation changes alone do not change executable qualification.
After a reviewed operator acceptance PASS, replace this record with the new
receipt's source manifest and receipt identity. Never bless changed hashes by
copying current source hashes without that acceptance evidence.

## Evidence and scope

The spec identifies `ca24910674181f3d83866ea5d4c4adec781185b2`, an earlier
transport revision. The actual PASS receipt's `final_plan.commit` is
`cd766c639c37acfb956d677ab95fdf7359a1bb61`; its entire 18-file source inventory
matches that commit and the reviewed `origin/t31-colab-safety-20261018` tip
`bdccf3a64ca6bae1bfeb94cf6ed2d4fb0683b9d9`. Compared with `ca249106`,
the PASS includes `inputs.py` and different `runtime.py`/`supervisor.py` bytes.
The receipt hashes, rather than the earlier commit description, select the
qualified transport. Neither that worktree nor branch was changed.

Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-public-release-20261019/acceptance/acceptance-1792425613/acceptance-result.json`.
SHA-256: `54e9f2f84dda243f138ab3f374e2d4b4be13c22b4977939f3c810020bf40e101`.
It records `qualified: true`, `status: PASS`, and `return_code: 0`.
CPU and L4 receipts both record verified stop and no cleanup error. L4's
intentional watchdog-loss terminal status is `INCOMPLETE`; the acceptance PASS
is based on its required observed safety gates and verified backend release,
not a claim that this fault arm was ordinary successful training.

Limits: 4.0 units total including 0.8 reserve (3.2 usable); CPU at most 20 minutes
at a priced ceiling of 0.5 units/hour (cap 1/6 unit, lease 300 seconds); L4 at
most 30 minutes at 2.0 units/hour (cap 1 unit, lease 180 seconds). First wall or
unit cap wins; no automatic allocation/job retry or purchase. Three account
assignments maximum, including notebooks. The measured final L4-to-Mac export
is exactly 64 MiB (67,108,864 bytes), SHA-256
`4a73a12d5c9c8daae8b860c26f3127886627ce71a091b647ca027c66a37ff917`,
43.160703166 seconds, lower-bound rate 1,554,860 bytes/second. This measurement
does not qualify arbitrary artifact sizes; those acceptance manifests used the
supervisor's generic default 1 GiB export limit, separately from the measured
64 MiB payload. T34 retains its own 64 MiB export bounds.

Final-only chunks require verified immutable outputs and prospectively priced
lost-work recovery. This PASS does not qualify exact continuation or a full
optimizer/replay/actor/target/RNG resume. Only jobs claiming exact continuation
need full-state acceptance.

## Reviewed import boundary

Review command (run against the pre-import main and the specified branch):
`git diff origin/main origin/t31-colab-safety-20261018 -- research/colab_cli docs/colab-cli.md tests/`.
The selected source files exactly matched the receipt before the accounting fix.
This is a selective source-bound import, not a wholesale branch merge.

| Selected file | Reviewed purpose and safety boundary |
| --- | --- |
| `research/colab_cli/ACCEPTANCE.md` | Separates final-only and exact-resume profiles; retains account, wall and unit caps. Scope clarified for final-only chunks in this change. |
| `research/colab_cli/acceptance.py` | Source-bound profile gates and measured final export; full-state gates remain required for exact resume. |
| `research/colab_cli/acceptance_faults.py` | Exercises profile-specific faults, immutable final export and verified release without treating skipped full-state gates as observed. |
| `research/colab_cli/acceptance_probe.py` | Bounded profile-specific synthetic probe; terminal artifact is not trainer-resume evidence. |
| `research/colab_cli/admission.py` | Imports source-bound acceptance checks for existing S1 entrypoints without S1 edits; this fix also checks saved T34 manifests at shared broker admission. |
| `research/colab_cli/inputs.py` | Validates HTTPS input size/hash before allocation with bounded, read-only retries and a shared deadline. |
| `research/colab_cli/policy.py` | Separates 120-second RPC attempts from 300-second recovery budget while preserving first caps. |
| `research/colab_cli/runtime.py` | Fetches pinned public objects inside the supervised child group; verifies bytes before trainer exec and retains watchdog cleanup. |
| `research/colab_cli/supervisor.py` | Profile admission, public-input preflight and UUID-fenced recovery; uncertain incomplete mutations refuse replay. |
| `research/colab_cli/transfers.py` | Uses durable completion fencing and idle proof for transfer acknowledgements, retaining bounded immutable verification. |
| `docs/colab-cli.md` | Operator profile guidance; stale qualification status replaced with receipt scope and pending accounting rerun. |
| `tests/fixtures/fake_colab.py` | Offline bounded fault/transport/public-input fixture; cannot allocate a real provider runtime. |
| `tests/test_colab_cli_acceptance.py` | Profile requirements and source-bound final-only acceptance regression coverage. |
| `tests/test_colab_cli_admission.py` | Rejects absent, changed or wrong-profile acceptance evidence. |
| `tests/test_colab_cli_fault_properties.py` | Fault recovery, caps, one allocation and cleanup invariants in the fake transport. |
| `tests/test_colab_cli_latency_policy.py` | Recovery timeouts remain inside shared deadlines and caps. |
| `tests/test_colab_cli_public_inputs.py` | Hash/size and total-timeout preflight refusal without live inputs. |
| `tests/test_colab_cli_rpc_recovery.py` | Replays absent/completed operation records only after idle proof; partial mutation refuses. |
| `tests/test_colab_cli_run8.py` | Reproduces the recorded L4 transport stall sequence offline, including exhaustion refusal. |
| `tests/test_colab_cli_runtime.py` | Public input verification precedes trainer execution; failed fetch stops/releases. |
| `tests/test_colab_cli_supervisor.py` | Fake broker coverage for public-input upload avoidance and qualified timeout behavior. |

All 11 differing `tests/test_s1_*` modules were rejected: they depend on
unimported study dispatch, training, delivery or forecast changes. All 32
deletions under Phase 0/T33/T34 tests/fixtures were rejected: they reflect
branch divergence, not transport fixes. No S1, Phase 0 or other T34 worktree
files were modified. Existing local study tests remain in place.

## Implementation verification (Mac, 2026-10-19)

No live acceptance was run. All pytest runs used the primary checkout's existing
Python interpreter with `SNAKE_DQN_DEVICE=cpu` and `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` each set to `1`; parallel runs used `-n 2`.
No test runs overlapped.

- `python -m pytest -q -n 2 -m 'not slow'`: **8,299 passed, 29 skipped,
  16 failed**, 2 warnings, 2,114.60 seconds. Every test in the changed Colab,
  T34 manifest/qualification, and Mac slot areas passed. Log:
  `/tmp/snake-rules-verification-20261019-full-1792469421540.log`.
- The 16 failures are confined to three untouched test modules: 13 Torch
  compile failures loading `@rpath/libc++.1.dylib`, two frozen-pool failures
  where the frozen `v8_decide` lacks `membership`, and one snapshot graft
  `_food_array` shape mismatch. The affected source, fixtures, configuration,
  and test modules are byte-identical to the initial revision
  `830894e37b8cdaa638c0cb5045b01e4254ad7de3`.
- An isolated archive of that initial revision, tested with the same interpreter
  and thread settings using `python -m pytest -q -n 2 -m 'not slow'
  tests/test_perf_compile.py tests/test_t22_frozen_pool.py
  tests/test_t4_snapshots.py`, produced **38 passed, 3 failed** in 23.73 seconds.
  It reproduced both frozen-pool failures and the snapshot failure; the compile
  cases passed there. Log: `/tmp/snake-rules-verification-20261019-baseline.log`.
  Repeating that selection in this worktree produced **25 passed, 16 failed**
  in 18.18 seconds, including the same missing-library compile failures.
  Log: `/tmp/snake-rules-verification-20261019-current-recheck.log`.
  The compile problem remains unresolved and checkout/cache dependent; it is
  not established as a reproducible baseline failure. No shared venv,
  compiler cache, frozen fixture, or snapshot implementation was changed.
- The initial focused Colab run produced 266 passed and two fake timing
  failures. The fixture observations were corrected to begin after rate
  calibration; the required arrival/departure, unknown endpoint, and third-poll
  cases passed on rerun and in the full non-slow run.
- The focused T34/slot selection (`tests/test_t34_transport_qualification.py`,
  `tests/test_t34_exit_manifests.py`, `tests/test_t34_exit_jobs.py`,
  `tests/test_with_slots.py`) produced **43 passed** in the final rerun (4.27
  seconds), including the saved-manifest admission regressions.
- `make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint`
  passed. Black (100 columns, two workers), isort (Black profile), and flake8
  passed for all 27 changed Python files. The two imported string-wrapping
  changes preserve the accepted modules' ASTs.
- Both actual `python -m research.t34_exit_b8.make_manifest --kind benchmark`
  and `python -m research.t34_exit_b8_fit.make_manifest` entrypoints, with fresh
  absolute `--output-dir` and `--out` paths, refused on the expected transport
  hash mismatch and wrote no manifest. This is the intended pre-acceptance
  state, not a dispatch qualification.

The non-slow suite is **not green**. The three reproduced baseline failures
and the 13 unresolved compile-environment failures are reported rather than
changing unrelated research code or weakening the tests.

## Machine-readable accepted executable inventory

This is historical PASS evidence. The accounting repair, shared T34 admission
guard, and formatting change the accepted source bytes. Imported overlong
evidence/error strings in `acceptance.py` and `acceptance_faults.py` were wrapped
to the repository line limit without changing their values. T34 must remain
refused until the operator supplies a new reviewed PASS.

```json
{
  "profile": "final-only",
  "qualified_commit": "cd766c639c37acfb956d677ab95fdf7359a1bb61",
  "receipt_path": "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-public-release-20261019/acceptance/acceptance-1792425613/acceptance-result.json",
  "receipt_sha256": "54e9f2f84dda243f138ab3f374e2d4b4be13c22b4977939f3c810020bf40e101",
  "schema": "snake-dqn-t34-transport-qualification/v1",
  "source_manifest": {
    "research/colab_cli/__init__.py": "624677cd409539df23ec6c1f21ad999059fe779715ee5eab0183b7b2498851f1",
    "research/colab_cli/acceptance.py": "f435eefa409e9d7dab44b8819f0053b6fea35ef76a499ecf2e7855ac024662f4",
    "research/colab_cli/acceptance_faults.py": "bba32c41da72d9a060d9ec302eaaeaacdc8c2f7c0bfee8f9bfd7eab405f03c5a",
    "research/colab_cli/acceptance_probe.py": "5fe350618270ecbb51ecdb44b16fc80a071d80633ac94679ba94541694251610",
    "research/colab_cli/acceptance_probe.sh": "9e7163c63bda2a86c96fe2eabee9be538d77e8151c13422467fe497d55be9064",
    "research/colab_cli/accounting.py": "13b87470e781844c1859007b6342912fdb8f369258ea49c33b2b2719990817eb",
    "research/colab_cli/admission.py": "8c2b59f44748302e85f9478ba232fb518df64d302e2f23e675210d7d95f54cb9",
    "research/colab_cli/backup.py": "125b73b512c63e06f4b7656e52b534967510685314ab5b2c1b9fbbe4232239fe",
    "research/colab_cli/bootstrap.sh": "e4a7bc69ba9da3ef5d5df6755a60bcb73878364d18b38f9d5a38d6ac18bbef7e",
    "research/colab_cli/cli_guard.py": "e563610fc9fadb59df824a38a93699428800b31b6b85bce3b90aba503c583bc8",
    "research/colab_cli/inputs.py": "56cce4bd7a6c9487f52ac914c743f34051b5e818893c8e48ca0da9ff08f03ae1",
    "research/colab_cli/launch.sh": "173b51e74e5c0b44c40d0d549b0d4a3a1dc1ab787baa68eb854ca0663b50ea79",
    "research/colab_cli/policy.py": "e2c43f8e0c5bb3c45b3d25303c7e41a3a938becbbfe1dd70455adac0aeeb3014",
    "research/colab_cli/release.py": "af34a828c23045143192272ec94f62dc60aa48c55ab0324bf45026ff352cf355",
    "research/colab_cli/runtime.py": "e1ff8c5a3b13fae272f98aa884d9210d1bec26e2eedc7490a8ad1ba22eeb1797",
    "research/colab_cli/source.py": "8b5fccfc809d7345faacda58dec3b497787ddf4961218fdbbe6ef90a1b2eef6a",
    "research/colab_cli/supervisor.py": "88d32393e7dc1719090e47f61c7c51d946fdee1cc6553d8f785a07465bfb945a",
    "research/colab_cli/transfers.py": "a8fd12c425d7dff168e300a6bf0a6bec38230d75520c6251e86a59b0df83d95c"
  },
  "status": "PASS"
}
```
