# F5 targeted long-body timing supplement — DRAFT, UNSEALED

Status: `DRAFT_UNSEALED_HOLD`. This package is preparation for Astra ruling step 3 only. It does not authorize a dispatch, Colab allocation, timing run, fit, science result, or `ACCEPT_F5_FIT`.

The associated implementation is `research/bl_residual/f5_timing_supplement.py`. It uses only the original 12 timing identities (allocation indices 672–683), scans each exact natural trajectory only through the already frozen H5000 horizon, and selects the first decision at length at least 700. A fixture without such a crossing is recorded as `ABSENT_WITHIN_H5000`; it is never substituted or retried.

Records are component-scoped operational timing only. They must carry the distinct `f5-targeted-long-body-first-crossing-v1` component ID and cannot enter the prior strict ledger or overwrite the completed timing units. A later sealed package must bind source, genuine CUDA parity, native-v6 environment evidence, immutable operational exports, terminal proof, complete accounting, and independent acceptance.

`SUPPLEMENT_DRAFT.md` is the frozen-workload proposal for review. It remains unsealed and has no checksum or completion marker by design.

This is a helper/workload draft. It has no admitted runtime driver or supervisor entrypoint. Before sealing, freeze that driver, its full source inventory and finite timeout/accounting envelope, then obtain matching genuine CUDA/native-v6 application qualification and independent acceptance. Existing qualified CLI transport remains unchanged.
