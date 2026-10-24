# V8 split main sizing: unchanged-pin preparation is blocked

Status: **INCOMPLETE preparation; no admissible main dispatch prepared.**
The passed canary, its operational PASS/release, the nine-cell scientific
allocation and the ruling remain unchanged. This record does not amend the
hash-bound intent, supply a main result, or authorize a larger envelope.

The requested mechanical repair cannot be completed while preserving all three
of the runtime pin, exact uploaded inputs, and qualified acceptance checks.
The main estimate is unsupported, and the fallback resume path requires an
input change that the accepted canary does not cover. Do not bypass either
refusal or dispatch a knowingly stranded first chunk.

## Frozen evidence

- Package: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-metadata-20261024-02`.
- Runtime: `669405e2bef629f7f69bc951cf49486d8b0a87bf`.
- Clean qualified broker: `/Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex`.
- Authority: `JUDGMENT_V8_ROUTE_20261023.md`, SHA-256
  `5358934afc504cbee94a74622dea0534c831de5e8d7d8f471d5ad4177871a108`.
- Effective packet envelope: **72 minutes / 12 units**; canary portion
  **24 minutes / 4 units**, main portion **48 minutes / 8 units**.
  The ruling requires an admitted main estimate <=20 minutes and a timeout
  at least twice that estimate. The single owner doubling already applies.

The accepted canary receipt records PASS and `stop_verified=true`. Its native
parity and per-cell independent reducer checks pass before the packer's sizing
refusal. These are saved receipt facts, not a new live session query.

## Timing calculation

| Measurement or calculation | Seconds |
| --- | ---: |
| Canary whole receipt | 511.20726829208434 |
| Canary benchmark | 39.40054554800008 |
| Whole receipt minus benchmark | 471.80672274408425 |
| Sum over canary arms/rosters of 2 x plain + event wall | 7.288841009999487 |
| Benchmark residual outside those intervals | 32.111704538000595 |
| Existing forecast: outside + 63 x benchmark | 2954.0410922680894 |

The existing forecast is **49.23401820446816 minutes**, exceeding both the
20-minute estimate ceiling and the 48-minute main wall cap. This is an
extrapolation, not a measured nine-cell runtime.

The factor 63 is the main/canary tick-pass ratio of 3 multiplied by the
E64/E256/E1024 shape-factor sum of 1+4+16. Applying it to the entire benchmark
also multiplies fixed setup and CPU-oracle work. However, reducing all residual
work to fixed cell overhead is not justified either: `v8_split_bench._tick`
ends its wall timer before `_digest`, which copies/hashes complete persistent
inventories after every warmup, plain pass and event pass. Both world count
and repeat count affect part of the residual. `_proof` and `_oracles` also
include population-wide checks/transfers.

The optimistic residual x3 scenario gives 1027.338820 seconds
(17.122314 minutes). Residual x21 gives 1605.3495016720644 seconds
(26.75582502786774 minutes), and still omits repeat scaling for part of the
digest work. Neither is a measured bound or a replacement admitted estimate.
The canary did not partition these costs. It therefore cannot justify silently
replacing the existing refusal with a <=20-minute estimate.

## Why the unchanged-pin split cannot be prepared

The pinned producer already publishes atomic cell payload/SHA/completion trios
and iterates frozen-E1024, mixed-E1024, scripted-E1024, then the same roster
order at E256 and E64, with three warmups and three measured repeats. Its CLI
has no cell-selection or unit-limit argument. Its existing deadline can leave
completed trios plus INCOMPLETE failure evidence, but cannot promise an exact
cell boundary or that both initial E1024 cells finish.

The pinned shell always unpacks `metadata.tar` and loads the main ledger from
that uploaded input tree. The current archive contains an empty main ledger.
A real continuation must upload a new verified ledger and its payloads/fixtures,
which changes the archive SHA. Leaving the archive unchanged would recompute
verified cells and violate resume-by-default.

The qualified `research.colab_cli.canary.check` requires equal input inventory
count, destination, transport and SHA, not merely equal or smaller upload sizes.
An offline same-size archive substitution with its manifest SHA correctly
updated is rejected by the actual frozen `supervisor.validate` with:

```text
canary does not cover real upload bytes x file count/destinations
```

The unchanged-input control passes that same validator. No acceptance check,
receipt or source identity was mocked for either control. The input substitution
is a synthetic negative control, not fabricated resume evidence or a proposed
runtime input.

A fresh canary covering resumed inputs is additional qualification, rather than
reuse of the accepted proof. On the observed whole-canary time, its doubled
17.04024227640281-minute forecast also exceeds the remaining
15.479878861798595 minutes of the 24-minute canary portion. No extra canary,
larger cap, waiver or new runtime pin is authorized by this status record.

## Offline verification and handoff

Evidence directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-main-sizing-20261024-01`.
`offline_verify.py` runs only against the clean frozen broker and writes a
create-only `OFFLINE_VERIFICATION.json` plus SHA/completion sidecars. It rechecks
qualification, rederives the real canary proof, reproduces the original sizing
refusal, runs unchanged/changed-input controls through the frozen validator,
inspects the empty archived ledger, calculates timing scenarios, and verifies
every original package file remains byte-identical. No dispatch manifest is
published.

Reproduce the evidence in a fresh output directory by copying the driver there
(it deliberately refuses to overwrite its existing report), then running from
the frozen broker with its project venv, single numerical threads and nice 15.
The recorded first driver invocation caught the validator's refusal under the
wrong exception type; correcting the diagnostic driver to catch the supervisor's
`Refused` class did not change any product code, input or acceptance check.

Focused unchanged contract tests:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-gpu-v8
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -o addopts= -p no:xdist tests/test_colab_cli_canary.py research/gpu_sim_spike/test_v8_split_packet.py research/gpu_sim_spike/test_v8_split_contract.py
```

Result: **91 passed in 0.64 seconds**, exit 0 through the macOS OpenMP isolation
wrapper. No numerical training, CUDA timing or broad project suite was run.

**Main dispatch commands: none under the unchanged constraints.** A future
implementation needs qualified support for bounded resumed input coverage and/or
a newly pinned harness with measured cost partitioning and an explicit unit
boundary, followed by a cap-respecting admission plan. Those changes exceed this
mechanical request; this report does not implement or approve them.

No live jobs/tmux sessions, runtime checkout files, checkpoints, serving pins,
hash-bound intents, original manifests/receipts, transport or qualification
registry were changed. Only this sibling status record is added to the repository.
