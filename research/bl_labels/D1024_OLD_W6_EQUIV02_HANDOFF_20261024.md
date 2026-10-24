# D1024 OLD W6: approved equiv-02 source rebinding

Status: **PREPARED_CHECKED_NOT_DISPATCHED**. Owner chat on 2026-10-24 relayed
Claude's YES to a fresh package binding repaired speed pin
`111c47818a097c57451386ef4f8a7009616f8f77`, preserving the original namespace,
allocation, seeds, caps, calibration and OLD W6 collector. No collection was launched.

The original `bl-labels-f0-speed/d1024-d4096-pack-01` remains immutable; all 24
files were inventoried before preparation and verified unchanged afterwards.
The fresh package is:

```text
/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01
```

Only `source_commit` and `mac_gate.source_commit` changed in the control, from
`bf8ac97dc66d1b4515e4c5510db3ad5886fa695d` to `111c4781...`. New control SHA256:
`9952387d0f493c7b0b62f1a210b1ed7e859031640c0022d9d43e200c51d337f7`.
The namespace still contains `bf8ac97d`; it must not be regenerated from the new pin.
All 2,160 materialized parents, seeds, allocation indices, gate plans and folds
are unchanged. The seed audit and empty verified ledger triplets are byte-identical.
The cap remains 144,000 seconds (40 hours), six workers, same calibration source
`61900af28932e47e22e332840fc4406559adb062` and runner `2a0821a9...`.

Reason: the sealed published round-27 equiv-02 T-EQ FAIL binds `111c4781`, while
the original control binds `bf8ac97d`. The registered fallback uses the entire
OLD scientific import closure at `61900af2`, W6. This does not change the T-EQ
FAIL or its allowlist and does not authorize NEW/W12 collection.
Publication: `research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES_02.json`, SHA256
`2d2e5ea4486e4cb8eb0c501c09b26906e9222f41aae99482e1c7c4afdf003541`;
verified byte-identical to origin/main before preparation.

The fresh HOLD manifest reflects this explicit sealed FAIL/OLD fallback. Package
`d1024-rebinding.json` and its identical committed ledger copy
`ledger/D1024_OLD_W6_EQUIV02_BINDING_20261024.json` bind all inputs and original files.
Its initial operational-pin recommendation is superseded by the create-only sibling
`d1024-operational-erratum.json`, also committed as
`ledger/D1024_OLD_W6_OPERATIONAL_20261024.json`.

Operational pin is **`587ff73ecdb9ebb25a104c3e45450ceb851c0358`**:

- `3c5fc403` predates `111c4781`; the admission requires an operational descendant
  of the sealed speed pin, so the former pin cannot accept the rebound package.
- At `111c4781`, the real OLD-root check raised `AttributeError` because
  `mac_gates.LEGACY_FACTORY_COMMIT` does not exist. `587ff73e` corrects that one
  reference to `frozen_factory.OLD_PIN` and adds four regression cases. It includes
  both `111c4781` and the `3c5fc403` guards. Science remains frozen at `111c4781`;
  the actual collector remains frozen OLD `61900af2`.

Verification: 122 targeted existing tests and four new guard cases pass; Black,
isort, flake8, shell syntax and diff checks pass. The real launcher at the clean
detached published operational pin reports
`CHECKED, stage=d1024, workers=6, width_reasons=[], deadline_seconds=144000`.
An additional offline admission/revalidation and real `mac_driver --check` pass,
confirming OLD collector selection and 2,160 pending parents. That verification-only
admitted control is outside the production package, in sibling
`d1024-old-w6-equiv-02-verification-01`; the production admitted-control, state and
output paths remain absent. No worker or lease was started by these checks.
One proof-harness assertion was corrected from the returned tuple length to its
jobs-list length; no production attempt or data was involved.

Independent Codex-V light review receipt:
`CODEX_V_D1024_REBIND_20261024.json` (with SHA256 and completion sidecars).

Use the following in Bash. These are operator commands only. Run the second
dispatch only after the first terminates with CHECKED and EXIT=0 and its lock
is released. A dispatched check is asynchronous. Names were fresh at preparation;
never reuse a consumed dispatcher name. Runtime host, priority, timing, lease
and cumulative-accounting guards still apply at actual admission.

```bash
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f4fix
OPS_PIN=587ff73ecdb9ebb25a104c3e45450ceb851c0358
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
D1024_ENV=(
  PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
  PYTHONDONTWRITEBYTECODE=1 SNAKE_DQN_DEVICE=cpu
  F0_COMMIT=111c47818a097c57451386ef4f8a7009616f8f77
  F1_COMMIT=111c47818a097c57451386ef4f8a7009616f8f77
  PACK="$ART/bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01"
  EQ="$ART/bl-labels-f0-speed/equiv-02/receipt.json"
  CAL="$ART/bl-labels-f0/w12-admission-01/receipt.json"
  OLD_FACTORY_ROOT="$ART/bl-labels-f0-speed/equiv-pack-02/old-reference"
  F4_INPUTS="$ART/big-lever/exp/f4/d1024-metadata-staging-03/inputs"
  F4_CONTROL="$ART/big-lever/exp/f4/d1024-metadata-staging-03/inputs/control.json"
  B8=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
  FRP3=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth
)

"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-d1024-check-old-eq02-rebind-01 "$OPS_PIN" -- \
  /usr/bin/env "${D1024_ENV[@]}" /bin/bash research/bl_labels/RUN_D1024.sh --check

# After the check log ends in CHECKED and EXIT=0, and its lock is released:
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-d1024-old-eq02-rebind-01 "$OPS_PIN" -- \
  /usr/bin/env "${D1024_ENV[@]}" /bin/bash research/bl_labels/RUN_D1024.sh
```

Expected check log:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/mac-bl-f0-d1024-check-old-eq02-rebind-01-587ff73ecdb9.log`.
No dispatch, tmux/session mutation, checkpoint/serving-pin change, or factory,
held-out, reserve or gameplay outcome look was performed. D4096 remains HOLD.
