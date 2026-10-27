# TRAIN-01-12: finite masked KL/entropy under calibrated temperature and post-preparation resume

Prepared only; no dispatch. Packet `train01-12` (`sandbox-a-train01-20261027-12`) fixes the
float32 categorical probability underflow in `torch.distributions.kl_divergence` that raised
`FloatingPointError("non-finite PPO actor loss")` on the first learner update of `train01-11`,
adds pre-update finite-check telemetry naming the first non-finite learner term, and resumes
from the verified post-preparation snapshot (`iteration-0000.pt`) exported by `train01-11` so
that calibration, E360 complete-life critic preparation, and the B8 baseline proxy are not
repeated. The learning recipe (`departure = 0.01`, `gamma = 1.0`, `lambda = 0.995`,
`clip = 0.1`, `entropy = 0.001`, KL schedule `0.02 -> 0.005`, differential LRs `1e-5 / 1e-4`,
`minibatch = 2048`, `epochs = 2`, `kl_stop = 0.01`, PopArt `beta = 0.999`) is unchanged.

## Root cause of the `train01-11` failure (exact numbers)

In `train01-11` (runtime pin `e86e4b0bc8ea37550103b14652878d1b63f5b0e1`), distribution
screening, temperature calibration (`T = 0.008439731318504158`, `1 / T = 118.487185`), E360
complete-life critic preparation (`held_out_late_ev = 0.4404 >= 0.10`, `passed = 1.0`), the
zero-update B8 greedy agreement check (`1024/1024`, `0` mismatches), the B8 baseline proxy,
and `iteration-0000` snapshot export all succeeded, followed by iteration-1 rollout collection
(`1024 x 256 = 262,144` rows). The first `PPOTrainer.update` call then raised
`FloatingPointError("non-finite PPO actor loss")`.

Loading `iteration-0000.pt` on CPU and running a 3-world x 24-tick rollout with the real task
code (`ResidentTrainingTask(worlds=3, allocation_offset=9)`, `T=24`) reproduces the exact
exception on update 1:

1. `torch.distributions.Categorical(logits=masked_logits(logits, legal_mask))` stores normalized
   log-probabilities `self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)` and computes
   `self.probs = softmax(self.logits) = exp(self.logits)` in IEEE-754 `float32`.
2. In `float32`, the minimum positive subnormal is $2^{-149} \approx 1.4012985 \times 10^{-45}$,
   and `exp(z)` underflows to exact `0.0` for all $z \le -103.97208404541016$. At calibrated
   $T = 0.008439731318504158$, an unscaled B8 advantage gap of $\Delta Q = 103.972084 \times T = 0.8774965$
   places a suboptimal legal action below `-103.972084`.
3. On row 71 (`tick = 23, world = 2`, all 6 actions legal), the frozen reference unnormalized
   logits are `[15.936467, 40.362400, -60.401756, 18.406952, 28.346058, -29.774279]`, giving
   normalized reference log-probabilities:
   `reference.logits = [-24.425922, -7.6293945e-06, -100.764191, -21.955441, -12.016354, -70.136703]`.
   For legal action 2, `reference.logits[2] = -100.76419067382812 > -103.97208404541016`, so
   `reference.probs[2] = exp(-100.764191) = 1.6815581571897805e-44 > 0` ($12 \times 2^{-149}$).
4. Because `old_kl` is evaluated only on the sampled rollout actions (which are >99% greedy
   action 1 where `new_logp = old_logp = -7.6293945e-06`, `ratio = 1.0`, `old_kl = 0.0 <= 0.01`),
   earlier minibatch steps update the actor without tripping `kl_stop`, shifting legal action 2's
   normalized log-probability from `-100.764191` to `-104.16975402832031 <= -103.97208404541016`:
   `distribution.logits = [-26.191460, -7.6293945e-06, -104.169754, -22.690142, -12.009279, -71.834373]`.
   Although `distribution.logits[2]` is completely finite, `distribution.probs[2]` underflows to
   `0.0` in `float32`.
5. `torch.distributions.kl_divergence(reference, distribution)` invokes PyTorch's
   `_kl_categorical_categorical(p, q)`:
   `t = p.probs * (p.logits - q.logits); t[(q.probs == 0)] = inf; t[(p.probs == 0)] = 0`.
   Even though `p.probs[2] * (p.logits[2] - q.logits[2]) = 1.6815582e-44 * 3.405563 = 5.7267e-44`
   is finite and negligible, `t[(q.probs == 0)] = inf` overwrites `t[2]` with `+inf`, and
   `t[(p.probs == 0)] = 0` does not clear it because `p.probs[2] > 0`.
6. On that minibatch, every earlier term is finite (`GAE` finite, PopArt `mean = 1050.4838`,
   `scale = 757.6935`, `norm_adv = [-2.114792, -0.251406, -0.546322, 0.808522, 0.862447, 0.695372]`,
   `new_logp = [-7.629e-06, 0.0, 0.0, -2.289e-05, -1.144e-05, -1.144e-05]`,
   `ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`, `old_kl = 0.0`, `policy_loss = 0.09102988`,
   `entropy_loss = 0.00010621`), and **`reference_kl` is the first non-finite term**
   (`reference_kl = [inf, -1.927e-08, -6.356e-08, -6.245e-07, -2.538e-07, -3.767e-08]`), making
   `reference_kl_loss = inf` and `actor_loss = inf` (whereas true mathematical `reference_kl[0]`
   in float32 is `-4.2480675e-08` and `actor_loss = 0.09102978`).
7. Secondary numerical hazard also fixed: in `Categorical.entropy()`, `logits * self.probs`
   multiplies masked illegal actions' `-3.4028235e+38` (`finfo(float32).min`) by `0.0`, whose
   backward pass `grad_output * (-3.4028235e+38)` overflows `float32` whenever `|grad_output| > 1.0`,
   producing `0.0 * inf = NaN` in `SoftmaxBackward0`.

## Numerical fix and telemetry

- `masked_categorical_kl(reference, distribution, legal_mask)` in `training_model.py` computes
  `sum(reference.probs * (reference.logits - distribution.logits))` masked to
  `canonical_legal_mask(legal_mask) & (reference.probs > 0)`, which is bit-exact (forward and
  backward) with `torch.distributions.kl_divergence` when probabilities do not underflow and
  remains finite when `distribution.probs` underflows `float32` on a finite legal logit.
- `masked_categorical_entropy(distribution, legal_mask)` zeroes `~(legal & (distribution.probs > 0))`
  before multiplying `distribution.logits * distribution.probs`, matching `Categorical.entropy()`
  bit-for-bit on legal support while avoiding `-FLT_MAX` backward overflow.
- `PPOTrainer.update` in `training_learner.py` checks all learner terms in causal order before
  stepping the actor optimizer and emits a `TRAIN01_PROGRESS` line with
  `stage = "learner-finite-check"`, `finite`, and `first_nonfinite_term` (no behavior or RNG
  change when finite).

## Verified post-preparation resume (`iteration-0000.pt`)

- `training_units.py` permits explicit source-only migration from `TRAIN01_11_SOURCE_COMMIT`
  (`e86e4b0bc8ea37550103b14652878d1b63f5b0e1`) in `source_migration`.
- `packet-train01-12/verified-ledger.json` is built by `build_ledger` from the read-only exports
  at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-11/output/exports/training`,
  binding unit `a100-dev/sandbox-a/train01/iteration-0000` (`phase = "pre-actor"`,
  `prepare_phase = "ready"`) and its 9 payload/sidecar/marker files under `resume/iteration-0000*`
  into `dispatch-manifest.json` and `sandbox.json`.
