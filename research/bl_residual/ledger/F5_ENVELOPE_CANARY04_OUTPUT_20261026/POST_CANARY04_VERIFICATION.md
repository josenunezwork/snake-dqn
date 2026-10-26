# Canary-04 offline verification

Complete output published FIRST at `e4dc238a55a27304e69d5d99d9a20bea76db0ec4`; local==origin verified before report.

| Check | Result | Evidence |
|---|---|---|
| Native v6 + kernel | PASS | bl-residual-f5-environment-receipt/v6; kernel 6.6.159+; exact raw boundaries and frozen reference |
| Real CUDA parity | PASS exercised fixtures 672/673 | fp32 cuda:0 finite 1x6 Q; atol=rtol2e-4; exact masked argmax; live recorded checks, frozen source verified offline |
| Actual producer/scanner | PASS CANARY_ONLY | export672 → ledger672 → scanner skip672 → export673; both units independently reopened |
| Transfer / staging | PASS | 169.497080s / 197.206293s; independent 300/300s limits |
| Publication / exports | PASS | 113 original files / 110644273 bytes reconstructed exactly; 15 exports / 109181742 payload bytes; hashes, sizes, SHA sidecars and markers |
| Terminal / release | PASS saved receipt | 1307.429406s; succeeded exit0; stop_verified; stop/sessions/usage complete; assignments0/rate0, cleanup null |
| Operational telemetry | PASS recorded | 25 records; whole qualification 634.251s; unique fixtures2; counter transient3 includes skipped672 |
| Full timing / calibration | NOT EXECUTED | Twelve workloads, coupled C/W and Mac companion remain unmeasured; base120/effective240 is an engineering allowance |

| Phase | Seconds |
|---|---:|
| startup | 0.000 |
| admission | 0.131 |
| fixture-load | 0.459 |
| control-source-checks | 2.319 |
| environment-capture | 65.711 |
| environment-compare | 16.115 |
| initial-boundary | 153.130 |
| cuda-load | 0.179 |
| cuda-parity | 117.551 |
| fixture-complete | 31.775 |
| cuda-parity | 92.755 |
| fixture-complete | 0.000 |
| final-boundary | 153.317 |
| export | 0.809 |

First fixture parity/boundary/export 117.551s; scanner/skip 31.775s; second fixture 92.755s. Inner sample clocks are separately recorded in JSON and exclude boundaries/scanner/export. No extrapolation of tiny fixture throughput to full timing or calibration. Saved receipt-backed release verification only; no fresh provider query. No dispatch, Colab/provider call, label/outcome opening or checkpoint deserialization.
