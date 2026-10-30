# Pre-data mechanical clarification — 2026-10-30

The registration is unchanged. Before any pilot data exist, make rounding and
operational details explicit:

- The 1,024-world split has exactly 256 assessment parents. Select each fourth
  world within a mix using one-based rank; this yields 255. Fill the remaining
  one from the last unselected world in global roster order. No outcomes enter.
- The qualified preference wrapper preserves the primary reserve when deriving
  its fallback. Actual effective A100 reserve is therefore 24 units, conservatively
  above the registration's 16-unit forecast. Timeout-derived unit caps remain
  H100144/A10096. This is an admission accounting detail, not extra science.
- Collection remains H5000 terminal-hero, without replacing dead worlds. The
  ~1M decision-support bar counts alive hero decisions rather than dead padding.
- The sparse archive stores ordered body/food coordinate shards, equivalent to
  compact owner/order maps reconstructed only for the current minibatch.
