# R30 timing upload-cap repair: probe-02 offline handoff

Prepared for independent predispatch acceptance; **G5/main HOLD**. No Colab call or live dispatch.
Astra ruling SHA256 `de404ed2ec595ed60f506ca403e142ae0ee7e90666b2a6c49b64c457c103a51b`
was read fully and verified against its sidecar. See the sibling
[prospective amendment](../../research/bl_round30/TIMING_UPLOAD_CAP_NONATTEMPT_AMENDMENT_20261025.md).

Runtime source pin: `773ef710fb468dfb4fb58b1ef342b9237946f7b2`, published on origin/main.
Clean starting point: `8260c8fe03b5f06dd31ad9a532a4e74e88b97cbd`; worktree
`/Users/josenunez/Projects/ml/snake-dqn-r30post`, branch `sol-r30-post-f4-20261025`.

The timing builder sets shared transport `max_export_bytes=67108864` (64 MiB).
Producer publication, exact file/sidecar admission and native receipt checks separately
retain the 524288-byte timing.json cap. The forecast now consumes the qualified native
receipt's integer `size` field. The source/control and every private input size/hash
are checked offline using the supervisor-normalized manifest and actual pinned source
preparation. The qualified transport implementation and scientific workload are unchanged.

Probe-01 UPLOAD_CAP is NON-ATTEMPT; no additional attempt allowance. All 55 files under
its refused packet, including local staging/output and receipt, remain byte-identical,
recorded in REFUSAL_PRESERVATION.json. Existing accounting records and operator evidence
were not modified. The original 40-minute liability remains pending reconciliation.

Retained: exact opaque af82651c bundle, full frozen cache and fit/audit plans, 2,000-batch
objective, continuous updates 0–30, closures 0/10/30, certificate, timing-family probe,
base/effective 15/30-minute envelope, deadline reserves, 150/300-second transfer/staging
bars, no_automatic_retry, native receipt and verified release requirements.

Verification used nice 15, threads=1, ./venv/bin/python, serial pytest -n 0:

- Timing producer/packet/forecast: **58 passed in 10.51s**.
- Dispatch/source/packet regressions: **108 passed in 72.97s** (seven packet tests overlap).
- Black --workers 1, isort and flake8 --jobs 1 over six changed Python files: PASS.
- git diff --check and unchanged qualified transport/scientific consumer checks: PASS.
- Actual offline staged/private preflight: **12 files PASS**, source archive 1,209,036 bytes,
  bundle 36,189,672 bytes; every file <=67,108,864 bytes.
- Actual dispatch --dry-run: **PASS**, zero allocations; live accounting NOT_CHECKED.

Exact commands and evidence are in VERIFICATION.json and DISPATCH_DRY_RUN.json.
No scientific units, private bundle contents, D4096 or t34-round0-study-01/ were opened.
No tmux sessions, released checkpoints, serving pins, qualified transport files or old
hash-bound evidence were modified. There is no live timing result or release proof.

Packet root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-02`.
Job ID: `r30-1-timing-calibration-probe-02-773ef710`.
Output/state: packet-relative `runtime-output` / `supervisor-state`, both unused.

| Payload | SHA256 |
|---|---|
| G5_TIMING_UPLOAD_CAP_PACKET.json | `62405e8567ea993cd0cf23cf66ad672ca6e25016781d7fc00e4c9f4124e1c90e` |
| job.json | `1fc4428f725b5667badd63c2e5ab15330fc4e6cea620553cd4dcc1efa7c24a8f` |
| inputs/timing-config.json | `c842e7314ad93cfeee16f4a94f1884fbccf28c4ed13f915f62945bc6fc1312d5` |
| UPLOAD_PREFLIGHT.json | `a7a22839b97c41fbb2c5854ca2c8eee0978f8757ddb45804382935b338dae6b8` |
| DISPATCH_DRY_RUN.json | `c10eed57ebe89e6b851b446c2432d94b7661e8a7d36540786f41aafae29f8093` |

Actual dry-run output:

```text
PASS_DRY_RUN r30-timing-calibration-probe-02 allocations=0 live_accounting=NOT_CHECKED
```

Exact operator command after independent predispatch ACCEPT, **not executed**:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30post
PATH="$PWD/venv/bin:$PATH" tools/ops/dispatch.sh r30-timing-calibration-probe-02 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-02/job.json \
  --broker "$PWD" --cwd "$PWD" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-02/supervisor-state
```

Append `--dry-run` for offline validation. This preparation does not authorize dispatch;
the independent predispatch acceptance and operator's live account check remain required.
