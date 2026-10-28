# TRAIN-01-14: compact lossless resume transport and normalization-exact KL/entropy

Prepared only; no dispatch. Packet `train01-14` (`sandbox-a-train01-20261027-14`) supersedes
`train01-13`, whose identity was consumed: the qualified supervisor refused it after 325 s with
`TransferError("total upload time cap reached")` (failure class `UPLOAD_CAP`, zero assignments)
while uploading the 1,327,620,931-byte `iteration-0000.pt`. The cap is the qualified transport's
`UPLOAD_TOTAL_TIMEOUT_SECONDS = 300` whole-staging deadline (`research/colab_cli/transfers.py`),
which is unchanged; `research/colab_cli` is not edited.

## 1. Compact derivative of train01-11 `iteration-0000.pt` (`training_compact.py`)

- **Composition.** The snapshot is 1307.8 MB of task state (`task/sim/bodies` 1022.4 MB int64
  with 12,288 non-zero elements, `food_cells` 175.3 MB, `food_present`/`corpse_food` 11.0 MB each,
  opponent `random_slots` 87.6 MB) plus 13.1 MB actor and 6.5 MB critic: padded, mostly empty
  ring buffers.
- **Choice, measured.** Whole-file stream compression of the exact original bytes, measured on the
  real snapshot: zlib-6 23,884,800 B; zlib-9 23,159,514 B; **xz preset 6 8,804,064 B**; xz-9
  8,773,448 B (no material gain, 7x decoder memory). Chosen: one XZ stream, LZMA2 preset 6,
  `CHECK_SHA256`, stdlib `lzma` only (no new dependency). A whole-file codec reproduces the
  original file byte-for-byte, so the resumed state is literally `torch.load` of bytes whose
  SHA-256 is the ledger-bound `2b5478e8...` — no semantic trimming of ring buffers, no
  re-serialization, no reliance on per-tensor reconstruction.
- **Artifacts (create-only, completion-last, next to the original).**
  `iteration-0000.pt.xz` and `iteration-0000.pt.xz.manifest`, each with `.sha256` and
  `.complete`. The manifest (`sandbox-a-train-compact-snapshot/v1`) binds the original
  name/size/SHA-256, the derivative name/size/SHA-256, the codec, and a description of the
  original decoded state: per-tensor records (path, dtype, shape, stride, storage offset,
  `requires_grad`, byte count, SHA-256 of the raw bytes) for all 209 tensors, plus a canonical
  SHA-256 over the non-tensor structure (container types, key order, ints, `float.hex`, strings,
  NumPy RNG arrays). Encoding is deterministic: an independent dry-run encode produced the same
  derivative and manifest bytes.
- **Runtime.** `train.run` sees the ledger's `compact_resume` binding, stream-decodes the derivative
  into `resume/iteration-0000.pt` (`.partial`, fsync, hard-link publish; refuses an existing target,
  truncated/corrupt/trailing streams, any byte beyond the original size, and any size/SHA-256
  mismatch), then the unchanged `load_ledger` re-verifies that file against its uploaded original
  `.sha256`/`.complete` sidecars, `torch.load` reads it on CPU, and `verify_decoded_state` checks
  every tensor record and the structure digest against the manifest **before `_restore`**. Any
  mismatch raises `CompactResumeRefused`; nothing is retried. Progress stages
  `compact-resume-decode`, `-decoded` and `-verified` are emitted.
- **Allowlist.** `training_units.COMPACT_RESUME_ALLOWLIST` freezes the only usable binding:
  unit `a100-dev/sandbox-a/train01/iteration-0000` with exactly the original, derivative and
  manifest descriptors. `validate_compact_resume` (torch-free) runs inside `load_ledger`, the
  Mac-side `attach_compact_resume` builder and `training_preflight.validate_uploaded_ledger`; the
  ledger must contain exactly that one unit, whose full-state export equals the allowlisted
  original. The predecessor source-migration allowlist is unchanged.
- **Admission.** The full-state `.pt` is no longer uploaded; the derivative and manifest are, as
  exact hash-bound siblings of the Mac original, with the original's `.sha256`/`.complete`.
  `tools/sandbox` admits the new input kind `training-export-derivative` (TRAIN-01 only, canonical
  `iteration-NNNN.pt.xz[.manifest]` names, completion sidecars matching the claimed hash, complete
  original present alongside).
- **Upload projection.** Derivative 8,804,064 B = 5 chunks of 2 MiB (manifest 69,385 B). Measured
  from the receipts: wall rate 1,635,817 B/s on the train01-13 resume upload (212 parts), slowest
  aggregate per-PUT rate 810,734 B/s (train01-11, assumed serial), per-file overhead 1.4 s median /
  3.67 s worst (measured small-file times). Projected derivative upload: **7.6 s nominal (2.5% of
  the 300 s cap), 14.5 s worst (4.8%)**, target < 25%. Whole staging including the five fixed
  inputs (measured), setup and all ten resume uploads: 54 s nominal (18%), 96 s worst (32%), versus
  the original `.pt` alone at about 1,640 s worst. Details: `upload-projection.json` in the
  engineering evidence directory, bound by `SEAL.json`.

## 2. Astra TRAIN01-13 review finding 1 — normalization-exact saturated KL/entropy (`training_model.py`)

- Normalization is anchored at one legal maximum `k`: `log p = (x - x_k) - log1p(s)`,
  `s = sum_{j != k} exp(x_j - x_k)`, so the dominant action keeps its residual (previously
  `logsumexp` rounded `log(1 + s)` to zero).
- KL uses differenced normalizers: with `q`'s dominant action `m`,
  `u_j = log p_j - log q_j = delta'_j - log1p(sum_{j != m} q_j expm1(delta'_j))`, and
  `KL = sum_j q_j phi(u_j)`, `phi(u) = 1 + (u - 1) e^u >= 0` (series for `|u| < 0.1`,
  `p_j (u_j - 1) + q_j` for `u_j > 1`): a sum of non-negative, cancellation-free terms.
- Entropy is `H = log1p(s) + sum_i p_i (-(x_i - x_k))`, also a non-negative sum.
- Gradients are analytic `torch.autograd.Function`s: `dKL/dy = q - p` (as `-q expm1(u)`),
  `dKL/dx = p (u - KL)` with normalizers cancelled exactly, `dH/dx = -p (log p + H)` likewise.
  Autograd through a normalized form would cancel near-one probabilities for the dominant logit.
- Unchanged: forward KL direction, temperature, clipping, coefficients, schedule, NaN contract
  (non-finite legal support returns `NaN`), float64 internals cast to the input dtype.
- **Oracle.** New tests use an independent 100-digit `Decimal` softmax/KL/entropy/gradient oracle
  and correct float32 rounding. Astra's counterexamples now give `1.401298464324817e-45`
  (`[0,-104]` vs `[0,-108]`), `6.095648319812954e-43` (`[0,-104]` vs `[0,-1000]`), entropy
  `1.7418251854950444e-16` and tail gradient `1.699341718629913e-16` (`[0,-40]`). Ten structured
  saturated/close/tied/extreme cases and a seeded 150-case sweep are exact for KL, entropy, the
  actor KL gradient and the entropy gradient. The frozen reference's KL gradient (never used:
  reference logits are computed under `no_grad`) is bounded at float64 resolution (one float32 ulp
  only at exact float32 rounding midpoints). The prior test's float64 `logsumexp` oracle, which
  reproduced the defect, was replaced.

## 3. Astra TRAIN01-13 review finding 2 — probability validation independent of `.logits` cache

`_probability_parameterized` identifies the constructor parameterization through
`Categorical._param` identity, not lazy-cache key presence, so reading `.logits` (which caches
clamped logits) or `.probs` no longer changes validation or the distribution. A distribution
without an identifiable parameterization raises `TypeError`. Regression tests cover before/after
property reads for zero-probability (NaN) and healthy probability inputs.

## Unchanged

Frozen TRAIN-01 recipe and control (only `source_commit` migrates from the bound train01-11 pin),
seed allocation domain and reused exhaustive audit, the 8,388,608-row preparation cap,
support/quality gates, proxy schedule, 8 h base / 16 h effective runaway guard, resume declaration,
qualified transport tree. No CUDA timing or capacity qualification is claimed by CPU preflight.
