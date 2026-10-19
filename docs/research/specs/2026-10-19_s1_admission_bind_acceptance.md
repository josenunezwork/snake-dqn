# Spec: bind S1 admission to the final-only CLI acceptance PASS (run 9)

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT.

## Evidence

Final-only CLI acceptance **run 9 PASSED** on source `ca249106` (branch
`t31-colab-safety-20261018`): `qualified: true`, every required gate OBSERVED.
Result: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792397953/acceptance-result.json`
- `final_artifact_transfer`: 67,108,864 bytes, 28.08 s, `bytes_per_second_lower_bound`
  2,389,970, sha256-verified on the Mac, hardware L4, profile final-only.
- Its `source_manifest` binds every `research/colab_cli/*.py|*.sh` file at `ca249106`.

S1 (`research/s1_mature_20261018/`, pre-registration APPROVED at a6cbba57) still refuses:
`ADMISSION_STATUS_FINAL_ONLY.json` has `final_artifact_transport.status: UNQUALIFIED`,
`measured_export_bytes_per_second_lower_bound: null`, and names a required field
`s1_final_artifact_transfer.bytes_per_second_lower_bound` that does not match the result's
`final_artifact_transfer` key. Its "conditional forecast exceeds usable gpu_allocation_hours"
refusal is stale: README now uses a primary-only hard reservation (19.1667 / 20 usable GPU-h)
with each optional rerun reserved atomically when needed (3e6d2a84). Confirm that in code.

## Required outcome

1. S1 admission reads the acceptance result by absolute path, verifies: `qualified == true`,
   `profile == "final-only"`, every required gate OBSERVED, `final_artifact_transfer.status
   == PASS` with `exported_bytes >= 64 MiB` (the per-attempt cap), and that its
   `source_manifest` equals the current `research/colab_cli` bytes (refuse otherwise:
   a PASS for other source does not admit). Fix the field-name mismatch at the S1 side.
2. Export liability = `final_bundle_bytes_upper_bound / bytes_per_second_lower_bound`, per the
   existing formula; the complete upper bound becomes finite. Admission must still refuse if
   the result is missing, stale, another profile, or the totals exceed usable caps.
3. S1 manifests upload the acceptance result as hash-bound input `cli-acceptance-result.json`
   (as ACCEPTANCE.md already requires).
4. Run the offline admission prepare step only (no Colab allocation, no `launch.sh` dispatch),
   write the refreshed admission status JSON, and report: status, complete upper bound
   (units, GPU-h, elapsed, Mac hours) vs usable caps, and any remaining refusal reason.
5. Tests for: PASS admitted; wrong profile / stale source / missing gate / short artifact
   refused. Full `tests/test_colab_cli*.py` plus all S1 tests green; Black/isort/flake8.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch
`t31-colab-safety-20261018`); commit and push when green. Do not change the approved
pre-registration's science (arms, seeds, steps, metrics, decision rule). No live Colab,
no tmux, never modify the acceptance artifacts. Final message: admission status and numbers,
what changed, test counts, anything unverified.
