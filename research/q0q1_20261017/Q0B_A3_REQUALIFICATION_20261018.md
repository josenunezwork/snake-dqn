# Q0b A3 preserved A100 packet and L4 replacement

Outcome: **QUALIFIED under A3**, using preserved A100 r6 measurements only.
This is retrospective operational hardware admission, not a gameplay efficacy
result, Q1 dispatch, strict pass, or release. No GPU work was rerun. No queue job
was canceled or enqueued by this implementation.

Implementation and L4 source commit:
`5573713a34edf7fe1413f4acfa62872a8e5ae2d1` (published on origin/main).
Registration A3 SHA-256:
`0b056540089473549493fc1d706796c912739e97d095bcc67843fa6c29309461`.
The owner selected **"Amend, raise total"**; the complete quote and the disclosed
post-r6 tolerance are in the appended A3. Original registration bytes are a prefix
of the amended file. Earlier admissions and manifests remain unchanged.

## Lag rule and limits

`policy_lag` is an integer published-version difference. The pinned recipe
publishes every `round(25/2)=12` finalized optimizer updates. A3 tolerates mean
increase **strictly less than 1/12 of a publication**, equivalent to one finalized
update, with **no p95 increase** and the unchanged ingestion-age p95 ceiling of
110% of matched control. Nonfinite/missing/negative inputs and every other
identity, counter, parity, exit and cap failure remain fail-closed.

The mature/n1 difference is 0.00248035985255266 publications, or approximately
0.029764318230632 finalized updates; p95 remains 1.0 and age p95 improves.
This tolerance was written after r6's zero-tolerance check failed. Fractional
means do not create fractional weight snapshots; they also do not establish
identical acting-weight histories. No efficacy endpoint was changed.

## Immutable evidence and actual result

Drive root on the Mac:
`/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab`.
All paths below are relative to it.

| Evidence | Path / SHA-256 |
|---|---|
| Original archive, unchanged | `q0q1-20261017/q0b/capacity-a100-20261018r6/result.tar.gz` |
| Original SHA-256 | `d052e91f467f6b42bd4be6b01baaffd8effba306a89dd318acc68d2c948c77ea` |
| New A3 archive | `q0q1-20261017/q0b/capacity-a100-20261018r6-a3/result.tar.gz` |
| New SHA-256 | `f46f85ab593fca4c770c1c36d5ee58bfa9273497251e58d5ad3f058d5789792a` |
| Original claim | `queue/claimed/q0b-capacity-a100-20261018r6/g0.291b7018260c-c3915164906b4032af56cf8f2c1d17b4.1555b50b6783474894f6a1e5ee667fdf.json` |
| Original terminal receipt | `queue/failed/q0b-capacity-a100-20261018r6/g0.1555b50b6783474894f6a1e5ee667fdf.json` |

The new archive has its own SHA sidecar. It preserves the original measurement,
original wrapper failure, runtime parity and T22/T24 fixture bytes, and embeds
hash-bound queue evidence and the A3 qualified receipt. The queue state for r6
remains failed; no done record was forged. Generator extraction and
`_qualified_receipt(..., 'A100')` accepted the actual published new archive.

| Inclusive result | Value |
|---|---:|
| claim-to-terminal seconds | 2435.551744699478 |
| GPU hours | 0.6765421513054106 |
| units at 5.3/hour | 3.585673401918676 |
| separately billed CPU hours | 0 |
| conservative Q1 attempt overhead seconds | 510.731003128476 |
| restart/n1 decisions/s | 2380.6587993910343 |
| mature/n1 decisions/s | 2430.777822164222 |
| restart/n3 decisions/s | 13508.133308649843 |
| mature/n3 decisions/s | 13585.468090254257 |

The inclusive duration includes claim arbitration, run setup, parity, measurement,
export and cleanup. The skipped wrapper calculation is reconstructed with all
outside-measurement time as an upper bound on prelude overhead, plus four parity
worlds, maximum observed cell overhead, and a 60-second Q1 export reserve. Actual
past export is charged once; it is not charged another artificial reserve.

Read-only accounting currently includes 11 attempts: **67.17911639787754 units**,
12.669356343415048 GPU hours and 0.06305555555555556 CPU hours. Two historical
closure estimates remain conservative bounds, disclosed by the accounting tool.
Pending L4 has no terminal charge yet, so this is not the final A3 cap. Its current
formula would yield 202.17911639787752 combined units and 167.17911639787752 usable
units, retaining the original 35-unit reserve. Final accounting must include L4 r7.
The unchanged aggregate runtime caps also still apply; raising units does not
waive those caps, and a later full Q1 forecast can still be refused.

## Preflight and operator commands

The new L4 manifest is
`jobs_approved/q0b-capacity-l4-20261018r7.json`, validated using
`research.colab_queue.core.validate_job`. It pins the published implementation
commit above and uses a fresh output directory. The measured A100 launcher units
were 516.431, 502.374, 497.433 and 497.693 seconds (mean 503.483), with 3/6 actors.
The runtime has 12 vCPUs; each cell has 120s warmup and two >=180s scored windows.
The estimate remains 50 minutes, timeout 100 minutes; the stricter inclusive
packet cap remains 3600s. A100 is a disclosed proxy for L4's timing, not an L4
qualification. No additional CPU-bound loop or simulator source changed.

The real Mac `q0b_measure` entrypoint exercised 70 oracle tests on real staged B8,
demo and parity inputs, then returned INCOMPLETE at the CUDA/environment fixture
guard. The final invocation used the L4 r7 manifest environment, current project
venv and default registered oracle argv (70 passed in 5.30s; total 6.733s). Its
exact argv is preserved at
`/var/folders/h2/r_nk05n12vg4v5f_rxyf65v40000gn/T/q0b-a3-final-preflight-4j499dv5/command.json`. Repeating its output directory was refused. This is an honest Mac
preflight; it does not manufacture CUDA qualification. Recovery itself ran for
real against the preserved CUDA packet and published the archive above.
Targeted qualification/accounting/generator/integration/queue tests: 121 passed.
The final Q0/Q1 package also passed 220 tests with four workers. An earlier
whole-repository non-slow run overlapped construction of the new test module and
reported 17 failures, 7,631 passes and 26 skips. Eight were in that module; the
finished module subsequently passed in both serial and parallel package runs.
Remaining failures are outside the changed implementation: actor message-key
parity expectations, missing frontend build, serving fallback expectations,
queue process control under concurrent load and native NumPy crashes. The broad
suite is not green; see `/tmp/q0q1-a3-full-tests.log` for the complete evidence.
Eight non-A3 failures reproduced on the pre-A3 `ad33f390` checkout at
`/tmp/snake-dqn-baseline-ad33.S3prMJ` (8 failed in 12.32s with `--tb=no`). They
cover M6 actor-message keys (3), T21 actor continuation (2), T4 actor-message keys,
T15's absent frontend build and served-checkpoint fallback expectations. The
queue cancellation test passed focused reruns on both current and baseline
checkouts (1 passed each); its full-run PermissionError was transient.
No training, simulator or serving source was changed to work around those failures.
Black, isort and flake8 passed for the research package and changed tests; the
repository `make ... lint` passed. The final operator handoff distinguishes these targeted gates from the broad run.

See [README.md](README.md#a3-operator-sequence-2026-10-18) for the complete
post-L4 accounting, Q1 materialization and dependency-ordered enqueue sequence.
The immediate operator commands are:

```bash
set -euo pipefail
cd /Users/josenunez/Projects/ml/snake-dqn-q0q1-a3
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
"$PY" research/colab_queue/q.py --drive-root "$DRIVE" cancel q0b-capacity-l4-20261018r6 --reason 'Superseded by A3 pinned L4 r7; no r6 GPU work'
"$PY" research/colab_queue/q.py --drive-root "$DRIVE" enqueue /Users/josenunez/Projects/ml/snake-dqn-q0q1-a3/research/q0q1_20261017/jobs_approved/q0b-capacity-l4-20261018r7.json
```

The L4 runtime still needs its Drive mount. No Q1 manifests have been generated
from absent L4 evidence. No tmux sessions, released checkpoints, serving pins,
worker notebook QUEUE_REF values, original receipts or older admissions changed.
