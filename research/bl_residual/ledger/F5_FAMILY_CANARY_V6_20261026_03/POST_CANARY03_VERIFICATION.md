# Canary-03 offline verification

Output published first at `0c76c088e641bf0a314ade32d773df290cb2e290`; clean local==origin gate passed.

| Check | Result | Evidence |
|---|---|---|
| Native v6 environment | PASS | Linux 6.6.159+, #1 SMP Thu Oct 22 03:07:54 UTC 2026; glibc 2.39; exact before/after; sealed reference comparison |
| CUDA parity | PASS on 2 exercised fixtures | cuda:0 fp32 finite 1x6 Q; rtol=atol 2e-4; exact masked argmax; live recorded, source checked offline |
| Producer/scanner resume | PASS CANARY_ONLY | export 672 -> verified ledger [672] -> skip 672 -> export 673; actual scanner revalidated both |
| Transfer / staging | PASS | 146.616334s / 172.460472s, independent 300/300 s bars |
| Complete publication / exports | PASS | 106 files, 110619361 bytes; 15 exported payloads, 109181750 bytes; exact roundtrip, hashes, sizes, sidecars, markers |
| Runtime release | PASS saved evidence | stop_verified; completed stop/sessions/usage; assignments 0/rate 0, cleanup error null; no fresh provider query |
| Full timing / calibration | NOT EXECUTED | transport-only; no optimizer, full source workloads, or Mac companion measurement |

Operational telemetry only; unique fixtures=2. Loop counter transiently reports 3 because skip 672 counts as a visit.

| Phase | Seconds | Unique fixtures/s |
|---|---:|---:|
| admission | 0.110 | n/a |
| fixture-load | 0.456 | n/a |
| control-source-checks | 2.278 | n/a |
| environment-capture | 65.858 | n/a |
| environment-compare | 16.265 | n/a |
| initial-boundary | 153.781 | n/a |
| cuda-load | 0.171 | n/a |
| cuda-parity | 117.875 | 0.00848356 |
| fixture-complete | 31.747 | n/a |
| cuda-parity | 93.222 | 0.01072708 |
| final-boundary | 153.464 | n/a |
| export | 0.808 | n/a |

672 inner sample 24.927973s (feature 24.548578, CUDA 0.364777, CPU 0.011033); 673 inner sample 0.025813s (feature 0.007373, CUDA 0.009178, CPU 0.006038). These inner clocks exclude boundary capture/scanner/export costs. Calibration seconds: unmeasured.

Whole qualification 636.035s, 2/636.035=0.00314448 fixtures/s; broker wall 1335.656052s (22.260934min). This wall time occupies 18.55% of 120 min and 9.28% of 240 min. Neither timeout is qualified for the full twelve-workload run: source timings and calibration remain missing. Use observed fixed overhead for planning, without extrapolating tiny parity throughput to H300/H5000/Watch/calibration. Raw-attestation envelope work remains independent and untouched.
