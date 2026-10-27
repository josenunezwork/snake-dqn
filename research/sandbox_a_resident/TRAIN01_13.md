# TRAIN-01-13: log-space legal KL/entropy, post-latch reference finite check, and post-prep resume

Prepared only; no dispatch. Packet `train01-13` (`sandbox-a-train01-20261027-13`) supersedes the
undispatched `train01-12` packet and addresses all three findings from Astra's review of
`train01-12`:

1. **Non-finite reference after the actor-stop latch (`training_model.py`, `training_learner.py`):**
   - `_legal_log_probs` in `training_model.py` validates finiteness over canonical legal support
     (`canonical_legal_mask(legal_mask)`). Any non-finite logit (`NaN`, `+inf`, `-inf`) on a legal
     action, or any non-positive/non-finite probability on a legal action when a `Categorical` is
     constructed from `probs`, marks the row invalid so `masked_categorical_kl` and
     `masked_categorical_entropy` return `NaN` rather than masking the failure to `0.0` / `-0.0`.
   - In `PPOTrainer.update` (`training_learner.py`), when `old_kl > self.config.kl_stop` latches
     `actor_stopped`, the stop-minibatch reference logits, masked reference logits, reference KL,
     and entropy are evaluated and checked for finiteness before continuing. Any non-finite term
     emits `TRAIN01_PROGRESS` (`stage = "learner-finite-check"`, `finite = false`,
     `first_nonfinite_term`) and raises `FloatingPointError`.
2. **Exact log-space tail retention for legal probability underflow (`training_model.py`):**
   - `masked_categorical_kl` and `masked_categorical_entropy` normalize legal log-probabilities via
     `logsumexp` over canonical legal support and evaluate positive/negative KL contributions and
     entropy contributions in log-space with `logsumexp` (`ref_logp + log(|diff|)` and
     `logp + log(-logp)` in `float64`, cast back to the input dtype). Finite legal tail
     log-probabilities whose `float32` `exp()` underflows to `0.0` (e.g., `<= -103.972084`) retain
     their exact mathematical contribution (`6.1096613e-43` KL and `7.1466222e-44` entropy on
     `[0, -104]` vs `[0, -1000]`; `6.8135563e-8` KL on `[0, -104]` vs `[0, -1e38]`), while
     genuinely absent support (`~legal`) contributes exact `0.0`.
3. **Evidence digest labeling (`VERIFICATION_TRAIN01_12_ERRATUM.json`, `VERIFICATION_TRAIN01_13.json`):**
   - Preserves historical `VERIFICATION_TRAIN01_12.json` bytes and records a sibling erratum
     distinguishing `control_allocation_sha256` (canonical JSON digest) from
     `packet_file_sha256["control.json"]` (committed file byte SHA-256).

## Unchanged recipe and post-preparation resume (`iteration-0000.pt`)

The frozen TRAIN-01 recipe (`departure = 0.01`, `gamma = 1.0`, `lambda = 0.995`, `clip = 0.1`,
`entropy = 0.001`, KL schedule `0.02 -> 0.005`, differential LRs `1e-5 / 1e-4`, `minibatch = 2048`,
`epochs = 2`, `kl_stop = 0.01`, PopArt `beta = 0.999`) is unchanged. Packet `train01-13` binds the
same verified post-preparation snapshot (`a100-dev/sandbox-a/train01/iteration-0000`) exported by
`train01-11` (`e86e4b0bc8ea37550103b14652878d1b63f5b0e1`).
