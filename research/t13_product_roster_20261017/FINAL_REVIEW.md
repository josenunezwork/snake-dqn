# Independent final T13 evidence review — PASS

Completed 2026-10-17 after the full run; descriptive integrity review, not a gate.

The read-only reviewer independently reconstructed all eight roster/scope point
tables and every paired bootstrap estimate/95% interval directly from the compressed
raw records, without importing the analyzer. It used resampled sums, inverse-CDF
binary search for quantiles, and sampled per-episode maxima. All values matched.

Verified evidence:

- 100 unique roster/seed episodes, 25 per roster, all H5000; 6,000,000 decisions.
- Compressed record objects equal all original JSON records. All raw receipt digests
  and exit codes match; all result-manifest hashes and copied run metadata match.
- All 165 bound runtime source/config/pin hashes unchanged from the run intent.
- Initial world, Python RNG and NumPy RNG fingerprints match for each paired seed.
- Exact served identities and veto scopes are stable; mixed B8 dispatch is hero-only.
- Per-snake and pooled arena counters/histograms reconcile, including kill accounting.
- All 156 rendered paired metric/CI entries match, including direction and boost scaling.
- The report's viewer interpretation is supported and retains the legacy hero-only
  veto, whole-arena reversal, mixed opponent/veto confound, descriptive-interval
  scope, and twelve-snake override of the six-snake YAML.
- Elapsed 27.01885 minutes; summed worker wall time 1.73669 hours.

The final test-only configuration-restoration fixture was inspected. It keeps the
Watch12 global config inside the tests and does not alter measured runtime source.
The final focused suite passed 9 tests. All measurement workers and their coordinator
exited normally; no gameplay was repeated during this audit. No tmux sessions or
unowned processes were operated on.

Bound summary SHA256: `7aadfa6574a9393ee6efd3baba8fb2893a2bd5c3b202528c263d7e9294ef46d0`.

Bound compressed raw-record SHA256: `d49d669285ab2ae10b73f7ed898036080763ef9c6929c3be792540e7dfe6b69d`.
