# F0 whole-session cap repair — pack-05

Source pin: `816b366912eb41a499a8c64161b1b16e3ef90a39`, published on
`data-f0-20261023` and `origin/main`. Worktree:
`/Users/josenunez/Projects/ml/snake-dqn-f0`. Later receipt/documentation commits
do not change executable source. Pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-pack-05`.

The old family probe `bl-f0-family-probe-f9bbae8c` was mechanically refused
because setup plus its independent 270-second runtime exceeded its five-minute
whole-session cap. Pack-04, its log, refusal receipt, original INTENT.md and
FAMILY_PROBE_EXECUTION.md are preserved. The owner's explicit resize authority
and complete arithmetic are in [SESSION_CAPS_AMENDMENT.md](SESSION_CAPS_AMENDMENT.md).
No Colab dispatch/allocation was performed by this repair.

## Replacement caps

| Stage | Base seconds: setup + work + export + margin | Base → effective minutes | Base → effective units | Base → effective reserve units |
| --- | --- | ---: | ---: | ---: |
| Excluded family probe | 180 + 270 + 120 + 30 = 600 | 10 → 20 | 0.02 → 0.04 | 0.004 → 0.008 |
| Full CPU canary | 180 + 3150 + 120 + 150 = 3600 | 60 → 120 | 0.10 → 0.20 | 0.020 → 0.040 |
| Series | | **70 → 140** | **0.12 → 0.24** | **0.024 → 0.048** |

Owner multiplier applies exactly once to caps/reserves. At 0.08 u/hour,
full stage wall costs are 0.026667/0.16 units; with reserves each fits its cap.
Setup planning uses the old log's approximately 180-second observation; work
values retain prior planning allowances and are not measured throughput.
The runtime now subtracts measured setup from the qualified watchdog's remaining
session window and reserves 300 seconds (probe) / 540 seconds (science) for
export and margin. It exports `session-budget.json`. Its absolute monotonic
collection deadline also charges ledger verification. There is no independent
`F0_RUNTIME_SECONDS` default.

CPU only, W=nproc, one numerical thread per worker, account cap three. The
scientific allocation remains 12 frozen + 12 mixed + 8 scripted H5000 worlds
and four Watch6 H14400 sessions. No retry is automatic; original plus at most
two manual mechanical completion attempts share stage/series replacement caps.
Previous repair-attempt wall/units must be deducted before any completion manifest.
The old refusal remains charged to campaign history; it is never erased.

## Verification and independent receipt

- CPU, one numerical thread, single process:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist research/bl_labels tests/test_colab_cli_guard.py tests/test_colab_cli_warmup.py`
  — **109 passed in 8.14 seconds**.
- `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/bl_labels`
  — PASS; 11 files unchanged.
- `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_labels`
  and `.../venv/bin/python -m flake8 research/bl_labels` — PASS.
- `bash -n research/bl_labels/colab.sh` and `git diff --check` — PASS.
- Pack preparation ran the real clean declared-dependency-only Python 3.12.15
  preflight: **7 runtime roots, 673 import statements, 223 local modules**;
  user/system site packages disabled. Fresh namespace/seed audit and empty
  verified ledger are included; all six checkpoint identities remain pinned.
- Qualified supervisor validation of the probe — PASS, preserved in
  `supervisor-validation.json`. Science template remains unadmitted until the
  actual same-pin family receipt is bound. No fabricated live receipt is used.
- Tiny real runtime entrypoint, four H3 populations, one worker: **COMPLETE in
  1.49 seconds**, no scientific `--life` files. Artifact:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/session-cap-smoke-05/output`.
  Modeled setup 180.0318 s plus 100 s pre-watchdog setup/guard, qualified window
  1100 s, reserve 300 s gave 619.9682 s collection. This proves the budget seam,
  not Mac timing/yield or a live Colab family PASS. Initial harness incorrectly
  expected no diagnostic session files; the corrected no-scientific-life check
  passes and is preserved in `session-cap-smoke-check.json`.
- Independent Codex-V receipt: [CODEX_V_SESSION_CAPS.json](CODEX_V_SESSION_CAPS.json),
  also copied immutably into pack-05.

Qualified CLI tree remains `d0d92bdb572b770284a1fea6152417ffa5f06bea`, qualified
commit `a36a821de98f7ef48b730ab454fa2870b0962a5e`, acceptance
`acceptance-1792740334`; registry SHA256
`4983cab57b26d9df9879ada094a84471eedbbb6c85599b1cdea26d86013f70f0`.
No transport implementation edits or requalification were needed.

| Artifact | SHA256 |
| --- | --- |
| `f0-family-probe.json` | `19c27d4eda4961a9bfcc1d8eb10061b6f56aaa1dd4fa9c1fa9367cb802ce2d2a` |
| `f0-cpu-canary-template.json` | `8a962520b3cd67149a82cfe0f5ee27394b033a8c8369bccbc50d6cf82f14e8d0` |
| `clean-import-preflight.json` | `91bf5b3e28b44912e77ce0b8c3721daba5a4ebd854659d1c0b0cff7198477a8d` |
| `SEED_AUDIT.json` | `11b00b91d5191447cc2e6f6521bf2810cbe09ecb2429711d0248021d36dfad20` |
| `control.json` | `9ee31223d989b8fc023efa54ac01837c6332b28e91ba64196e526fd253d879bf` |
| `verified-ledger.json` | `0e8023c24c6300e3a9db8d85b9b4852911b3bcc8489afbddf762d037b9efe542` |
| `DISPATCH.txt` | `7c3ca8e36f94aed3a2b74b8323a2be73d62c450f427d9619a8ee9d6626b02d6a` |
| `supervisor-validation.json` | `6541d95df713a43db858f237124d362523cf5c7c51c23bf0a6bda7d3bce0fce0` |

## Exact operator sequence

The designated broker was clean at `0c29d96cbc10de7332e94248a6d3ba2827eae644`
when checked. The commands below fast-forward it before dispatch. No broker
checkout, live session or daemon was changed by this repair.

```bash
BROKER=/Users/josenunez/Projects/ml/snake-dqn-acctfix
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-pack-05
OUT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-output-05
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
COLAB=/Users/josenunez/.venvs/colab-cli/bin/colab
cd "$SOURCE"
git -C "$BROKER" fetch origin
git -C "$BROKER" merge --ff-only origin/main
"$PY" "$BROKER/tools/ops/qualification.py" check --repo "$BROKER" --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$BROKER/tools/ops/dispatch.sh" bl-f0-family-probe-816b3669 "$PACK/f0-family-probe.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-state-05-probe --dispatch-txt "$PACK/DISPATCH.txt"

# Run the following only after actual probe PASS, verified exports and backend release.
"$PY" -m research.bl_labels.manifests bind --template "$PACK/f0-cpu-canary-template.json" --receipt "${OUT}-probe/receipt.json" --out "$PACK/f0-cpu-canary.json"
"$PY" -m research.colab_cli.supervisor --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-state-05-science --colab "$COLAB" validate "$PACK/f0-cpu-canary.json"
"$BROKER/tools/ops/dispatch.sh" bl-f0-cpu-canary-816b3669 "$PACK/f0-cpu-canary.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-state-05-science --dispatch-txt "$PACK/DISPATCH.txt"
```

## Remaining gates and protected state

Actual source-family probe PASS/verified exports/backend release, receipt-bound
science canary, Mac worker-hour yield ruler and 64-root Mac replay audit remain
operator steps. A prepared/validated pack proves none of those live outcomes.
Digest/bare-audit mismatch remains INVALID; completed zero-rescue remains
INSUFFICIENT; fewer than eight rescue-bearing lives per measured Mac-worker-hour
equivalent remains KILL; missing units/cap expiry remains INCOMPLETE.

No Colab dispatch, provider mutation, unit purchase, serving change, Mac timing
run or strict gate occurred. No tmux session, released checkpoint, serving pin,
qualified CLI implementation, frozen intent or old hash-bound artifact changed.
