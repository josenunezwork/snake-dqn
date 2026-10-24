# F4 D256 device-declaration repair

Frozen F4 source: `c2c82ff240a28247dd9b18608831eecea6d94205` at
`/Users/josenunez/Projects/ml/snake-dqn-f4-d256-qualified-03.noindex`.
Published on main with the complete qualified CLI subtree
`d0d92bdb572b770284a1fea6152417ffa5f06bea`; main's operational transport is preserved.
The intermediate pin `1629fb25` is superseded and must not be used.

The successor runbook is `RUN_D256_NEXT_DEVICE_FIX_20261024.sh` in
`/Users/josenunez/Projects/ml/snake-dqn-f4fix/research/bl_residual/`.
Keep `RUN_D256_NEXT.sh`, `D256_NEXT_HANDOFF_20261023.md`, the failed
`d256-pack-01`, its original preflight and its ops log unchanged.

The registered fit is L4/fp32. Both prepared manifests now declare `cuda:0`
at top level and in `env.F4_DEVICE`. The shell retains its required-variable
check and passes the plain manifest-bound `"$F4_DEVICE"` argument understood
by the qualified parser. The fit CLI accepts `cuda:0`. FitRecipe, registration,
allocation, labels, masks, seeds, updates, caps and resume rules are unchanged.
Registration SHA256 remains
`66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65`;
FitRecipe SHA256 remains
`49d12f486bffa455adcedbc318e88f1524f5c502894d0c1ace47daa84b9d3a5a`.

## Completed offline evidence

- Targeted package/canary/schema/qualification suite: **101 passed in 3.52s**.
- Black, isort, flake8, Bash syntax and diff checks passed.
- Real fit CLI entrypoint with `--preflight-main --device cuda:0` passed on
  the failed pack's existing inputs. This metadata-only mode skips training
  partitions and performs no fit.
- Real no-fit preflight entrypoint on the final pin: **63 passed in 1.89s**.
  Receipt: `big-lever/exp/f4/d256-preflight-02.json`, SHA256
  `b844c4c626c840d50021721922a3cf7d72ebb182b4c5de88f2de339e5e29205b`.
  It is already complete; do not repeat the preflight phase into this receipt.
- Offline admission copies: `big-lever/exp/f4/d256-device-validation-03/`.
  The script `validate_offline.py` calls
  `research.colab_cli.supervisor.validate(manifest, ROOT)` for **both** emitted
  manifests, using the final clean qualified broker. Canary: **PASS**.
  Main template: **EXPECTED_HOLD**, exact reason
  `dispatch requires a SHA-bound family canary receipt`. Both resolve to CUDA;
  no proof was fabricated or guard bypassed. The live canary is still required
  before `seal-main` can create and validate production `main.json`.
- `validation.json` SHA256:
  `ea2356afc8904b563968366bbf53fa6ec2dd024d893f05ad3253df135c4029ec`.
  All **16 input inventory entries** are identical to the failed pack, and
  supervisor validation rehashed every local input. Only device declarations,
  commit/job identity and fresh output namespaces differ in the admission copies.
  These are validation copies; production `d256-pack-02` remains unmaterialized.

All artifact paths above are relative to
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.
No Colab allocation, dispatch, fit, reserve look, gameplay analysis, timing-arm
mutation, tmux mutation, serving change or release was performed.

## Remaining operator commands

Run one line at a time and wait for terminal evidence before the next dependent
phase. Every Colab phase must verify runtime release. The runbook retains the
P1 timing-arm refusal. The fresh production pack is `d256-pack-02`, with new
prepare/canary/fit IDs and state directories. The existing judgment, factory
ledger and registered input identities remain in force.

```bash
NEXT=/Users/josenunez/Projects/ml/snake-dqn-f4fix/research/bl_residual/RUN_D256_NEXT_DEVICE_FIX_20261024.sh
bash "$NEXT" prepare
bash "$NEXT" canary
bash "$NEXT" seal-main
bash "$NEXT" fit
bash "$NEXT" publish-fit
bash "$NEXT" stage-d1024
bash "$NEXT" speed
bash "$NEXT" publish-speed
bash "$NEXT" analyze-speed
bash "$NEXT" record-speed
bash "$NEXT" d1024-check
bash "$NEXT" d1024
```

If and only if the sealed T-EQ result is FAIL, replace the last two commands with:

```bash
bash "$NEXT" d1024-check-old
bash "$NEXT" d1024-old
```

All remaining rules and phase meanings are those of
`D256_NEXT_HANDOFF_20261023.md`. Joint R-A remains after both D256 and D1024
checkpoints; this repair opens no scientific look or release authority.
