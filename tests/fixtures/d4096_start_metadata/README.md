Copied metadata only from
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d4096-old-equiv-02-pack-01/`
on 2026-10-25 for the R-A hash-convention regression:

- `d4096-control.json.gz` and `d4096-admitted.json.gz` losslessly preserve the
  original JSON bytes (gzip with mtime 0). These contain allocation/control
  metadata; no collected label, trajectory, statistic, or result payload.
- The seven `attempt-*.start.json` files are unchanged copies from
  `d4096-state-mac-01/`.
- The admitted allocation's compact JSON SHA-256 is
  `74b1c5db8cd0c082311cf2c02666418847855298762174e0b9f721762cbff229`.
- The admitted control's compact JSON SHA-256 is
  `b1182f800a23dd98117133d2051be4773ac9d5d65e01f9961e3a3ad39a563584`.

Tests first verify the unchanged metadata against these real receipt hashes,
then relocate authority paths into a temporary directory and update only the
affected control hash. No test reads the production artifact directory.
