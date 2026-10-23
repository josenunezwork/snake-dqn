# F0 delivery — exact labels and bounded CPU canary pack

Implementation and offline family-probe admission pass. No Colab allocation or
dispatch was performed. The live probe, full canary, Mac yield ruler and
64-root Mac replay audit remain operator steps.

Source pin: `f9bbae8c02f2cdcee235b5415eb5d9eec8a6fe8b`.
Worktree: `/Users/josenunez/Projects/ml/snake-dqn-f0`.
Branch: `data-f0-20261023`. Source is published on the branch and `origin/main`.
Later delivery documentation/receipt commits do not change executable source.

## What is built

`factory.py` uses the existing B2 driver/snapshot restoration, T3 gate constructor,
exact clone/full digest and exact-stop rule. It collects bare B8 Watch6 lives and
the prescribed S-gate worlds; restores the last 300 decisions after death;
reads bare truth from the natural tape; simulates reacting alternatives; and
exports complete ordinary mass labels, selected bare audits and H1440 probes.
Snapshot memory is bounded. Every descendant carries parent-world provenance.

`exports.py` provides atomic immutable per-life/session payloads, SHA/completion
sidecars, portable verified ledgers and outcome-blind partial aggregation.
Resume skips exactly verified life labels and skips a session only when all its
expected life payloads are verified. `replay_audit.py` provides the independent
64-root recorded-action Mac audit. `runtime.py`, `colab.sh` and `manifests.py`
provide the source-pinned CPU transport path, including the mandatory excluded
source-family probe and real-receipt binding.

## Evidence

- Final source package plus reused snapshot/digest seams:
  `python -m pytest -q -p no:xdist research/bl_labels tests/test_t34_shadow.py research/bl_served_search/test_b2_engine.py`
  — **51 passed**, 16.36 seconds.
- Package plus newly qualified CLI guard/warmup integration:
  `python -m pytest -q -p no:xdist research/bl_labels tests/test_colab_cli_guard.py tests/test_colab_cli_warmup.py`
  — **99 passed**, 9.26 seconds before the final operational probe addition.
- Earlier supervisor/schema/transfer regression baseline:
  `python -m pytest -q -p no:xdist tests/test_colab_cli_supervisor.py tests/test_colab_cli_schema_failures.py tests/test_colab_cli_transfers.py`
  — **107 passed**, 533.62 seconds, before merging the upstream requalified CLI update.
- Existing tied-rescue serial oracle:
  `python -m pytest -q -p no:xdist tests/test_t34_shadow_batch_oracle.py::test_h300_doom_search_exports_match_serial_with_tied_rescue`
  — **1 passed**, 101.95 seconds.
- Black (100 columns), isort (Black profile), flake8, `bash -n colab.sh`,
  and `git diff --check` pass. All pytest commands use the primary venv,
  CPU and one numerical thread, with xdist disabled.
- Tiny real Mac entrypoint: all four populations at three transitions each
  completed in `mac-smoke-20261023-a/output-c/runtime-result.json`.
  A separate real excluded probe completed, published no life-label files,
  and was rejected by scientific aggregation. These are execution proofs,
  with no Mac timing/yield claim.
- Final clean Python 3.12.15 preflight passes: seven explicit runtime roots,
  673 import statements, 223 local modules, system/user site packages disabled.
  Final qualified supervisor validation of `f0-family-probe.json` passes.

The immutable pack is:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-pack-04`.

| Artifact | SHA256 |
| --- | --- |
| `f0-family-probe.json` | `494dcdae4c2adcaf028a8856548875d3771055ed15080c13070dd59923b1086c` |
| `f0-cpu-canary-template.json` | `88abefede8f97188b61b2c077ce411a0e5cb396cb53f5d2678403e22bc707fb6` |
| `clean-import-preflight.json` | `6634c60bb24b0c094b73fb91de1c2a6e18c418285b3422ac00688014d0096331` |
| `SEED_AUDIT.json` | `7daba9ba0530af7db6833adbfc68a9571751c951c504b1138d5c7d2823d29a14` |
| `control.json` | `91cf2a29f43fe011c76bfc08882e57add6c3894a82863fb7330c20942126fc2b` |
| `verified-ledger.json` | `8bd4eb27c004fcffb5a8edabbf5e8d3edb31d2be66a0459111b02580a0c85fc2b` |
| `DISPATCH_v2.txt` | `c3f580b02cae96f838605cc5fe95962424b79a4739d7e2571bddb59ad4aaad40` |

The qualified CLI registry binds tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`,
qualified commit `a36a821de98f7ef48b730ab454fa2870b0962a5e`, acceptance
`acceptance-1792740334`. Registry SHA256 is
`4983cab57b26d9df9879ada094a84471eedbbb6c85599b1cdea26d86013f70f0`.
The requalified CLI update arrived through the required `origin/main` merge;
F0 did not edit its implementation or run requalification.

## Claude's exact operator sequence

The designated broker was clean at `e9322ae02925ffd14809148d78d7466db18ef85c`,
an ancestor of `origin/main`, when inspected. It needs the operator's fast-forward
before dispatch. It was not changed by F0. Use the corrected `DISPATCH_v2.txt`;
the original handoff file is retained as immutable history.

```bash
BROKER=/Users/josenunez/Projects/ml/snake-dqn-acctfix
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-pack-04
OUT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-output-04
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
COLAB=/Users/josenunez/.venvs/colab-cli/bin/colab
cd "$SOURCE"
git -C "$BROKER" fetch origin
git -C "$BROKER" merge --ff-only origin/main
"$PY" "$BROKER/tools/ops/qualification.py" check --repo "$BROKER" \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json

"$BROKER/tools/ops/dispatch.sh" bl-f0-family-probe-f9bbae8c \
  "$PACK/f0-family-probe.json" --broker "$BROKER" --cwd "$SOURCE" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-state-04-probe \
  --dispatch-txt "$PACK/DISPATCH_v2.txt"
```

After the probe's actual PASS receipt, exports and backend release are verified:

```bash
"$PY" -m research.bl_labels.manifests bind \
  --template "$PACK/f0-cpu-canary-template.json" \
  --receipt "${OUT}-probe/receipt.json" --out "$PACK/f0-cpu-canary.json"
"$PY" -m research.colab_cli.supervisor \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-state-04-science \
  --colab "$COLAB" validate "$PACK/f0-cpu-canary.json"
"$BROKER/tools/ops/dispatch.sh" bl-f0-cpu-canary-f9bbae8c \
  "$PACK/f0-cpu-canary.json" --broker "$BROKER" --cwd "$SOURCE" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/canary-state-04-science \
  --dispatch-txt "$PACK/DISPATCH_v2.txt"
```

## Frozen caps and decisions

CPU only, zero GPUs, W=nproc, one numerical thread per worker, account cap three.
The unchanged scientific allocation is 12 frozen + 12 mixed + 8 scripted
H5000 worlds, plus four Watch6 H14400 sessions.

| Stage | Effective units | Reserve units | Effective minutes |
| --- | ---: | ---: | ---: |
| Excluded family probe | 0.02 | 0.004 | 5 |
| Full F0 canary | 0.18 | 0.036 | 55 |
| Total series | **0.20** | **0.040** | **60** |

Base total is 0.1 units / 30 minutes / 0.02 reserve; doubled once. Measured
accounting rate is 0.08 u/hour. Setup, failed attempts and exports count; first
cap wins. Up to two manually dispatched outcome-blind completion attempts must
fit the original series envelope. There is no automatic retry or extension.

Any digest or bare-audit mismatch makes the tranche **INVALID**. A complete
zero-rescue canary is **INSUFFICIENT**, without extension. Fewer than **8
rescue-bearing lives per measured Mac-worker-hour equivalent** kills the
hindsight claim. Missing units or cap expiry is INCOMPLETE. H1440 reversals
above ten percent derive tranche-wide delay labels. Local smoke success,
transport proof and prepared manifests do not establish any of those outcomes.

No serving change, strict gate, Mac yield qualification, 64-root cross-platform
qualification, unit purchase, direct provider call or Colab dispatch occurred.
No live tmux sessions/daemons, released checkpoints or serving pins were changed.
