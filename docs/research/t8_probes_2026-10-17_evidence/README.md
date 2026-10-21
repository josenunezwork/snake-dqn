# T8 raw-raster and arrival-race probe evidence (2026-10-17)

Development diagnostics only. This covers no policy training, gate rescoring, candidate selection
or serving change. The probes reuse T5's bank (19,976 states, 342 worlds) and folds.

**Cited by:** [t8_probes_2026-10-17.md](../t8_probes_2026-10-17.md).

**What it supports:**

1. Probe 1. The raw-raster CNN reached a D300 AUC of 0.741950. That is below the ≥0.85
   signal-recovery bar and the ~0.77 frozen-fuse benchmark.
2. Probe 3. Arrival-race inputs add a modest gain whose CI excludes zero (GBM +0.022983 over the
   60 scalars; AUC 0.749106). The gain is too small to authorize training.
3. The T1 late-LAD check. The recorded best surviving action has the larger raw Voronoi area in
   11/46 = 23.91% of cases.

| File | Contents |
| --- | --- |
| `cnn.json` | CNN probe (294,353 params, MPS, seed 20261017): per-fold training and AUC results. |
| `cnn-audit.json` | Numerical audit of the CNN and arrival predictions; `PASS`. |
| `cnn-source-equivalence.json` | Executed CNN commit `08d49a1a` vs tracked source: formatting-only change, AST equal; `PASS`. |
| `arrival-fit.json` | Arrival-race fits (108-wide feature design) and T1 results. |
| `arrival-kernel-parity.json` | Compiled vs float32 reference parity on 46 restored T1 states; `PASS`. |
| `bank-kernel-parity.json` | Bank feature kernel parity (505 exact states, full bank 19,976); `PASS`. |
| `restoration.json` | Byte-verified restoration of 19,976 states / 342 worlds. |
| `t1_check.json` | T1 late-LAD rows: 46 late LADs, 11 with larger Voronoi area (share 0.239). |
| `audit.json` | Combined numerical audit of the CNN and arrival probes; `PASS`. |
| `verification.json` | Focused pytest, Black, flake8, `git diff --check`, independent review. |
| `artifact-manifest.json` | Execution commits, plus SHA-256 of the raw artifacts, source, report and evidence. |
| `environment.json` | Versions; CNN on MPS, other work on CPU; at most 2 concurrent processes. |
