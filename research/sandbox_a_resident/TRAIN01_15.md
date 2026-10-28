# TRAIN-01-15: float64 `log_softmax` legal KL/entropy with standard autograd

Prepared only; no dispatch. Packet `train01-15` (`sandbox-a-train01-20261027-15`) supersedes the
undispatched `train01-14` packet (`2739de7f9d10f1e4802c64efbcd7707dc8f900dd`) and resolves the sole
MAJOR finding from Astra's review of `train01-14` without touching the reviewed compact lossless
resume transport or sandbox admission modules.

## 1. Astra TRAIN01-14 MAJOR fix — float64 `log_softmax` KL and entropy (`training_model.py`)

- Removed `_anchored_normalization`, `_PHI_SERIES`, `_kl_phi`, `_LegalCategoricalKL`, and
  `_LegalCategoricalEntropy` (no custom normalizer and no custom `torch.autograd.Function`).
- `_legal_masked_logits64(distribution, legal)` validates finiteness over canonical legal support
  (`canonical_legal_mask`), preserving constructor-parameterization inspection via
  `_probability_parameterized` (`Categorical._param` identity), promotes legal log-weights to
  `float64`, and masks illegal actions (`~legal`) to `-inf`.
- `masked_categorical_kl(reference, distribution, legal_mask)` evaluates
  `log_p = log_softmax(actor_logits_masked)` and `log_q = log_softmax(reference_logits_masked)` in
  `float64`, zeroes off-support entries with `torch.where` before multiplication so excluded actions
  contribute exact `0.0` without `0 * inf` in forward or autograd backward, sums
  `q * (log_q - log_p)` over legal actions (`q = exp(log_q)` on legal support), lets PyTorch
  autograd differentiate, casts back to `float32` (input dtype), and returns `NaN` on any row with
  non-finite legal support (preserving learner finite-check telemetry and `FloatingPointError` both
  before and after the `old_kl > kl_stop` latch).
- `masked_categorical_entropy(distribution, legal_mask)` evaluates `-p * log_p` from `float64`
  `log_softmax` over legal actions with `torch.where`, lets autograd differentiate, casts back to
  `float32` (input dtype), and returns `NaN` on any row with non-finite legal support.

## 2. Regression and verification tests (`test_training_learner.py`)

- **Astra overflow counterexamples and learner paths:** direct value and raw actor-logit gradient
  regressions when actor and saturated reference favour different actions (`[0, -1000]` vs `[0, 1]`
  -> `KL = 1.31326162815094`, grad `[-0.7310586, 0.7310586]`; `[0, -1000]` vs `[0, 1e-6]` ->
  `KL = 0.6931476593017578`; `[0, -710]` vs `[-710, 0]` -> `KL = 710.0`; `[0, -1000]` vs
  `[-1000, 0]` -> `KL = 1000.0`, grad `[-1.0, 1.0]`), plus `PPOTrainer.update` regressions on both
  the ordinary actor-update path and the `old_kl > kl_stop` latch path with diagnostics disabled and
  enabled.
- **Saturated tail retention:** earlier `[0, -104]` vs `[0, -108]`, `[0, -104]` vs `[0, -1000]`,
  `[0, -104]` vs `[0, -1e38]`, and `[0, -40]` entropy/gradient cases retain positive non-zero
  `float64` log-space contributions when `float32` `exp()` underflows.
- **Independent 100-digit `Decimal` oracle:** structured cases (including opposite-dominant
  saturation) and a 150-case seeded sweep match the oracle within `float64` `log_softmax` then
  `float32` cast rounding (`rtol = 3e-7`, `atol = 1e-15`), with exact `0.0` gradients off support.
- **Finite-difference gradient checks:** `torch.autograd.gradcheck` in `float64` and `float32`
  input/output autograd comparisons against `float64` central finite differences across partial,
  all-false fallback, and full masks.
- **Real-snapshot CPU first-update smoke:** decodes `train01-11` `iteration-0000.pt.xz`, verifies
  derivative and original sizes/SHA-256 plus all 209 decoded tensors against
  `iteration-0000.pt.xz.manifest`, restores actor/critic/trainer, calibrated temperature, and a
  6-world real restored task slice across all three opponent mixes on CPU, collects transitions,
  and verifies finite `PPOTrainer.update` losses, parameters, buffers, and `learner-finite-check`
  telemetry.

## 3. Unchanged

Compact lossless resume transport (`training_compact.py`, `training_units.py`, `training_resume.py`,
`training_preflight.py`, `train.py`), sandbox admission (`tools/sandbox/*`), `training_learner.py`,
frozen TRAIN-01 PPO recipe and control (only `source_commit` updates), seed allocation domain and
reused exhaustive seed audit, 8,388,608-row preparation cap, proxy schedule, 8 h base / 16 h
effective runaway guard, resume declaration, and qualified transport tree
(`5833e54e9c5396121cbbf99a5cd4e52006e564fd`). No CUDA timing or capacity qualification is claimed by
CPU preflight.
