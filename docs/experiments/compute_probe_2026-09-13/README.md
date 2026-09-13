# CPU/MPS PQN compute probe — September 13, 2026

Status: **BOUNDED_STABILITY_TRIAL_ONLY**. On the current M5 Pro Mac, MPS completed the frozen short whole-update workload 2.361x and 2.534x faster than CPU in two matched seeds. This is throughput evidence for a further bounded MPS stability trial. It does not establish policy quality, long-training reliability, cross-device numerical equivalence, or a reason to change production device defaults.

## Frozen probe

The probe used source a19b13ca3d7d3983fe7c7f083f3e1f395084a220 and launched four serial roles in ABBA order: CPU seed 2026091391, MPS seed 2026091391, MPS seed 2026091392, then CPU seed 2026091392. Within a seed, CPU and MPS share the device-normalized configuration and initial network tensor hash. Native floating-point updates and final weights were not expected to be cross-device identical.

Each role ran two excluded warmups followed by eight timed whole updates. MPS synchronization occurred inside every measured update boundary. The workload was corrected-v3/raster31v3 with E16/S6/T16, one hero, batch size 256, one exact padded SGD epoch, beta zero, Watch pre-move decisions, and per-environment autoreset.

The fixed opponent was the historical native stateful RandomSafeSimdPolicy. It intentionally differs from the stateless canonical advice anchor used in the scripted-advice study. This probe did not train or compare an advice policy.

## Throughput result

| Seed | CPU mean update | MPS mean update | CPU useful steps/s | MPS useful steps/s | MPS/CPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2026091391 | 0.8621 s | 0.3652 s | 277.2369 | 654.4202 | 2.361x |
| 2026091392 | 0.8721 s | 0.3442 s | 293.5421 | 743.8152 | 2.534x |

Both pairs had matching normalized configuration and initial-tensor hashes, finite telemetry, eight timed updates, and complete supervisors. The result meets the protocol's narrow condition for one separately frozen MPS stability trial.

## Resource and integrity evidence

The host is an Apple M5 Pro Mac with 18 CPU cores, 20 GPU cores, 64 GiB unified memory, and macOS 27. Every role was bounded by one numerical job, two CPU threads, a 30-second child limit, 4 GiB child RSS, 8 GiB MPS driver memory, 12 GiB available system memory, and a 20-second heartbeat limit.

The independent standard-library audit passed 37 checks with no failures or pending items. Across 111 host samples, CPU utilization was 0.0–13.6%, GPU utilization 0–35%, available memory 26.89–28.55 GiB, and swap 6.599 GiB. No thermal or performance warning was recorded. Host GPU use and CPU use are aggregate observations, so they cannot be attributed solely to a probe child. The highest recorded MPS driver allocation was about 1.08 GiB, below the 8 GiB cap.

The probe script SHA-256 is 4b7ee462dc87855c9490f08b9de3587c5988bd4b19901b0a422ea58866bf868c; the frozen protocol SHA-256 is 48af030c204fe0118863c70fa1975548d1169432ba5cabee031628d45facff5b.

## Limits and evidence

This sample has eight timed updates per device/seed after warmup. It does not measure long-run memory behavior, MPS recovery after faults, evaluation speed, end-to-end training time, or model quality. A later MPS run must use a new frozen protocol, one numerical job at a time, and the same or stricter limits. No checkpoint or configuration default changes follow from this probe.

- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/protocol.json>)
- [Independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/audit.md>) and [machine-readable audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/audit.json>)
- [Probe source](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/probe.py>) and [closed resource log](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/resources.jsonl>)
- [CPU seed 2026091391 terminal](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/runs/cpu-seed2026091391/terminal.json>) and [MPS counterpart](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/compute-probe/runs/mps-seed2026091391/terminal.json>)

