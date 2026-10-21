# T5 input-feature separability evidence (2026-10-16)

CPU-only development diagnostic on existing branch outcomes; not gate evidence. The feature
dataset itself (`features.npz`, SHA-256 `e4ecbe99...`) stays at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/`.

**Cited by:** [t5_separability_2026-10-16.md](../t5_separability_2026-10-16.md).
[T5b](../t5b_risk_and_fuse_2026-10-16.md), [T8](../t8_probes_2026-10-17.md) and
[T10](../t10_certificate_horizon_2026-10-17.md) reuse the same bank, folds and bootstrap.

**What it supports:** the T5 **FAIL** verdict, which missed all three bars:

| Measure | Result | Required |
| --- | ---: | ---: |
| Death-AUC gain | 0.008826 | 0.05 |
| AUC of the 24 features alone | 0.747583 | 0.80 |
| T1 match | 17/46 = 36.96% | 70% |

| File | Contents |
| --- | --- |
| `fits.json` | Separability fits (`t5-separability-v1`): seed 20261016, 5 world-grouped folds, 2,000 world bootstrap draws. |
| `restoration.json` | Combined feature-restoration receipt (`t5-separability-features/v1`): counts, parts, validation, output hash. |
| `restoration-part-0..3.json` | Per-part restoration receipts from the T4e return bank, with input/output hashes. |
| `t1_check.json` | Per-row T1 check of whether `best_surviving_action` scores strictly higher on `area_log`/`boost3_area_log`; verdict `FAIL`. |
| `independent-feature-audit.json` | Independent bank/feature audit: 19,976 states, 342 worlds, 360 shards verified; `PASS`. |
| `independent-fit-audit.json` | Independent prediction/fold audit; `PASS`. |
| `artifact-manifest.json` | Artifact SHA-256 identities, session exit codes, focused tests, source commit `212d5e92`. |
| `environment.json` | Python/library versions, CPU device, thread and worker limits. |
