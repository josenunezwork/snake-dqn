# F1 digest repair — pack-09

**Packaging repair verified; live admission blocked by remaining frozen probe
envelopes.** Source pin `2af48cd0b98a2063f46871cf4d320a5541b789af`, published on
`data-f0-20261023` and `origin/main`. Qualified Colab CLI tree remains
`d0d92bdb572b770284a1fea6152417ffa5f06bea`. This sibling corrects the preparation
claims in [F1_HANDOFF.md](F1_HANDOFF.md); old pack-08 and receipts remain immutable.

## Exact comparison and root cause

`f1_ruler.main` successfully calls `load_bank` and `bank_digest` before entering
the INVALID handler. The shared printed logical bank SHA is
`a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d`.
The next check is `run -> load_ledger -> exports._verify` on
`inputs/f1-verified-ledger.json`. `_verify` computes SHA256 of the payload bytes
and compares the entire sidecar string to `<digest>  <payload basename>\n`.

Pack-08's CPU ledger payload SHA is
`9448f8ed92eb6469a766bad770b287214551bcc05ff556e92389c506a56b7d2c`;
its copied sidecar names `f1-cpu-verified-ledger.json`, while the runtime path
requires `f1-verified-ledger.json`. L4's payload SHA is
`200610b46100025e2ed98d32280bbc7923d15df450a23b82f029c3ac2e370ea9`;
its copied sidecar names `f1-l4-verified-ledger.json`. Both payload hashes are
correct. The basename comparison fails identically on both platforms.

There is no runtime comparison against an original F0 session export digest at
this failure. The error's F0 prefix comes from the shared export verifier. F0
session hashes are provenance embedded by `build_bank`; runtime replay has not
started when the ledger verifier fails. Rebuilding from all 31 real canary-05
completed session exports reproduces pack-08's raw bank bytes exactly: 64 roots,
11 selected parents. XZ transport digest remains
`f626e2510d1c2f7620d2463c3eece089f9d11e72b3c37d8c873f0b68b1f2b995`.
No partial bank, path hash, temporary-file identity or platform nondeterminism
caused these INVALIDs.

The fix stages ledgers at `<pack>/<platform>/f1-verified-ledger.json` and rejects
any triplet upload that changes its payload basename. Content and filename
verification remain strict.

## Verification and artifacts

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0`.
Fresh pack `f1-pack-09`, outputs reserved under `f1-output-09`, and state names
`f1-state-09-{cpu,l4}-probe`. No Colab dispatch was performed.

- Real artifact offline regression: reconstructs the bank from the 31 real
  canary-05 session triplets, stages both actual failed probe input inventories,
  reproduces the ledger failure, and verifies both corrected destinations.
- Focused package plus qualified-family guard selection: **145 passed in 8.92s**,
  single process with `-p no:xdist`; final changed test file rerun **12 passed in
  1.50s**. Exact environment/commands below. Black 100, isort, flake8, Bash syntax
  and `git diff --check` pass.
- Expanded selection: 145 passed, one existing dispatch warmup fixture failure.
  Independent reviewer reproduced it from archived baseline `8126cbcb`: fake CLI
  requires the supervisor sitecustomize hook during the earlier accounting
  `sessions` call. The transport implementation and fixture were not changed.
- Fresh Python 3.12.15 declared-only sparse import preflight: PASS, 8 roots,
  679 statements, 225 local modules.
- All four exact-manifest full upload qualifications PASS; largest fallback
  forecast 149.738273s <150s, including source-fetch bound 269.738273s <300s.
- Mac actual `f1_ruler run --probe --workers 1` used manifest-staged inputs and
  qualified-session budget fixture: COMPLETE in **13.962617167038843s**, all three
  one-root schedules, five verified export triplets. Both platform ledgers pass
  at their runtime destinations. Evidence: `f1-functional-13/entrypoint-proof.json`,
  `destination-checks.json`, `output/runtime-result.json`, and `execution.log`.
  This is functional evidence, not a live family PASS or Mac timing ruler.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu F1_REAL_ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist research/bl_labels tests/test_colab_cli_canary.py
```

The create-only Mac entrypoint script and exact argv are retained under
`f1-functional-13`; repeating it requires a fresh artifact directory.
Independent review receipt:
[CODEX_V_F1_DIGEST_REPAIR_20261023.json](CODEX_V_F1_DIGEST_REPAIR_20261023.json).

## Live admission and exact commands

The original stage caps are shared by all attempts. `charge_failed_probe`
verifies the old stopped receipt, manifest hash, exported INVALID payload and
inventory hash, then conservatively charges the entire supervisor lifetime.
It preserves the original forecasts, reserves and series limits; repeated or
already-charged attempts refuse rather than reset the envelope.

| Probe | Failed lifetime s | Charged u | Whole s remaining | Units remaining including reserve |
| --- | ---: | ---: | ---: | ---: |
| CPU | 187.42267954204 | 0.004164948434267555 | 1012 | 0.03583505156573245 |
| L4 | 347.0063398340135 | 0.14844160092899467 | 252 | 0.21155839907100532 |

Actual supervisor validation refuses both replacement probes with
`timeout must include at least 2x measured estimate`; main templates refuse
the missing family PASS. `f1-pack-09/validation.json` records all four refusals.
The dispatch instruction file therefore carries an explicit **HOLD**. An
independent Fable prospective ruling is needed for headroom and the old
INVALID/fault census before a newly frozen live package can be admitted.
The old receipts include failed CLI calls: CPU one upload, L4 one upload and
one exec. Existing operational rules exclude both platforms; the analyzer also
refuses old INVALID attempts. These failures cannot be silently omitted after
fresh success. No caps, forecasts, rules or placement outcomes were amended.

The exact reserved dispatch commands are below; **they currently refuse** and
are not instructions to bypass the HOLD. After a ruling, prepare new immutable
manifests and dispatch instructions rather than editing this pack in place.

```bash
BROKER=/Users/josenunez/Projects/ml/snake-dqn-acctfix
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-pack-09
"$BROKER/tools/ops/dispatch.sh" bl-f1-cpu-probe-2af48cd0 "$PACK/f1-cpu-probe.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-state-09-cpu-probe --dispatch-txt "$PACK/DISPATCH.txt"
"$BROKER/tools/ops/dispatch.sh" bl-f1-l4-probe-2af48cd0 "$PACK/f1-l4-probe.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-state-09-l4-probe --dispatch-txt "$PACK/DISPATCH.txt"
```

No tmux sessions, released checkpoints, serving pins, existing receipts,
qualified CLI code or legacy queue workers were modified. The 64-root F0 Mac
audit, F0 completion/yield admission and serving release remain independent.
