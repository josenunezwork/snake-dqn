# Mac F0 production and equivalence — 2026-10-23

Authority: round-24 `JUDGMENT_PLACEMENT_MAC.md`, SHA256
`738a367bec2c3399ed8bd9383811d7fa643109a8aaba384a775f1c028a7285ea`,
including its appended C1 clarification (separate stall-fix scope), plus round-24
and round-26 judgments. This sibling supersedes the preserved old Mac
W2/quiet-gap execution wording. Frozen intents, old receipts and serving stay unchanged.
This is implementation/preparation evidence; the full Mac ruler, Canary-B, replay audit,
registered F0 look and D256 collection remain Claude's operator steps.

Prepared package: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/mac-pack-02`.
The controls record their frozen executable `source_commit`; later receipt/doc commits
can leave executable bytes unchanged. Admission checks those bytes against that pin.
Canary-B keeps the canonical 36 namespace/ID/seed/roster/horizon allocation exactly;
it has fresh Mac output/state and an empty verified ledger. F1 logical bank SHA256:
`a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d`.

**Digest provenance:** full_digest binds session/platform-sensitive policy/runtime carry
and Python/NumPy/Torch global RNG/checkpoint context; the separate `state_digest` is a
labelled SHA256 of the established simulator-only environment digest, retaining its
bodies/food/events/masks/rewards/environment RNG and excluding that context. Full restore
identity remains mandatory. Mac W1/W3/W6 must match full and label-row hashes, including
selected H1440 results. Byte identity with x86 is not required. A CPU/L4 neutral-digest
comparison needs newly exported neutral digests before F1-bis; it was not performed here.

## Exact Claude commands

The package's immutable scripts contain all synchronous Python commands. Launch each
through tools/ops so it runs from a clean detached pinned worktree and has its own log.
The factory and guard command each hold the shared research.mac_lease; mac_run owns
only the worktree reservation. Do not wrap the scripts in a second Mac lease.

```bash
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/mac-pack-02
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PIN=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$PACK/control.json")
cd "$PACK"
bash -n RUN_ME.sh RUN_D256.sh
shasum -a 256 -c RUN_ME.sh.sha256 RUN_D256.sh.sha256
cd "$SOURCE"
"$SOURCE/tools/ops/mac_run.sh" --repo "$SOURCE" bl-f0-mac-me-01 "$PIN" -- /bin/bash "$PACK/RUN_ME.sh"
```

`RUN_ME.sh` sets F0_COMMIT/F1_COMMIT to the pin, CPU and every numerical thread=1,
then performs these operations in order:

1. Guarded F1: all 64 frozen roots at W1/W3/W6, identical digest, state_digest and
   label_sha256 per root. Expected about30min; operational cap1h.
2. `mac_gates admit-canary`, binding the actual passing F1 receipt into a new control.
3. Factory `--check`, then the exact36 Canary-B allocations, W6/nice10, cumulative
   cap3h. Setup, source/ledger verification, recovery and drain are charged. The final
   ten-minute drain reserve is inside the cap. Full36/36 is required.
4. `mac_gates ledger`: publish the outcome-blind verified ledger before the look.
   Only complete immutable triplets enter it; interrupted originals stay unchanged.
5. Guarded `replay-canary-a`: the bank's64 hash-selected x86 recorded-action roots,
   on the Mac. Budget up to1h, actual duration unmeasured. This runs outside Canary-B's
   denominator. Unexplained divergence means X86_LABELS_NOT_MIXABLE; it does not gate
   Mac-only production. No x86 labels enter D256 here.
6. `look-canary`: the single registered F0 aggregate after the complete published
   ledger. A canonical global claim prevents re-looks via new files or package roots.
   Denominator=W6×cumulative full-attempt runtime wall; the exact durable attempt
   chain is verified so no earlier zero-leaf attempt can disappear. Zero rescues is
   INSUFFICIENT; rate<8 is KILL; equality at8 passes the rate bar.

The script stops on any failed ruler/guard or incomplete canary. It never automatically
retries. A timing arm stops admissions within1min and yields at the next completed life;
555s drain plus bounded cleanup fits10min. Verified lives skip label computation on
resume; partial parents replay deterministic discovery. This is completed-unit resume,
not full-state trajectory continuation. A crash without a terminal receipt is HOLD.
One mechanical completion shares Canary-B's original3h cap: fresh output, SAME
canary-state-mac-01, original admitted control, union ledger over both attempts, and
both `--attempt` arguments to the sole look. No completion follows scientific failure.

After the script exits, inspect its terminal receipt and `f0-look/decision.json`.
D256 stays HOLD until F1/F0 PASS and the actual F4 approval JSON exist. Claude creates
that exact five-field judgment.json only after accepting F4 Codex-V, following
research/bl_residual/F4_HANDOFF.md. Its reserved input path is
`$PACK/inputs/judgment.json`; this package creates no approval. Then:

```bash
APPROVAL="$PACK/inputs/judgment.json"
"$SOURCE/tools/ops/mac_run.sh" --repo "$SOURCE" bl-f0-mac-d256-01 "$PIN" -- /bin/bash "$PACK/RUN_D256.sh" "$APPROVAL"
```

`RUN_D256.sh` admits only the exact F4 judgment/registration/recipe, clean frozen seed
audit and passing Mac receipts, then checks and launches W6 at the fixed16h operational
pause. Its finite pre-data window is864 parents:96 Watch6,288 frozen,288 mixed,192
scripted, preserving4:32 and12:12:8 parent ratios. Namespace-pinned parent IDs and
parent-index×1,000,000+life-index give stable append-only life indices. F4 selects192
fit lives (folds1–4) and64 reserve lives (first16 Watch6 then48 other, fold0, cap4 per
reserve parent). No refill or resize follows insufficient support. The fixed window
runs until complete or the cap; there is no unregistered intermediate support look.
Selected support is forecast at6–9h, not measured here; the full window can take longer,
up to16h. A cap pauses on its ledger for the judge. D1024/D4096 and every fit, reserve,
funding or gameplay step remain behind F4 R6; their manifests are outside this task.

## Guards and limits

W12 is configurable only after explicit owner approval and a verified calibration
receipt: schema bl-labels-f0-w12-calibration/v1, duration_seconds≥1800,
0<w12_w6_wall_ratio≤1.30, thermal_not_ok=0, slowdown_trips=0,
platform=darwin-arm64 and code_sha=the frozen F0_COMMIT. Until then W6.
Guards require AC/lid open, nominal parsed pmset therm, RSS≤4GiB/worker and40GiB total,
free disk≥50GiB, and slowdown≤30% against the first-hour baseline. Three thermal trips
in24h reduce subsequent production admission width by3; Canary-B remains W6.
The sealed rolling6h rescue-rate guard pauses below half Canary-B's rate without
printing rescue counts or relabelling results. Progress heartbeats contain operations only.

Timing/strict/serving preempts through
/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed. Wait for factory
terminal receipt and lease release before starting timing work. Remove only an arm
you own. Never kill daemons/tmux or use process-name patterns. Cleanup targets only
children/groups this implementation created and verifies quiescence before lease exit.
No Colab dispatch, transport stall-fix work, fit, strict/serving gate or release occurred.
The closed Round0 bank and snake-dqn-l2mac were not opened or modified.

## Prepared pin and verification

Executable pin: `61900af28932e47e22e332840fc4406559adb062`. It is published on main.
The native entrypoint proof used merged commit `94210cf61e86e2fc8c7762d0b48ff7a46006afc1`,
whose executable bytes are identical. The full package selection passed126 tests
with `-p no:xdist`, one numerical thread and the real immutable artifact fixture;
the final priority-setting fixture rerun passed8 tests. Black/isort/flake8 pass.
Native final-pin proof: factory `--check`0.142s, real excluded H3 factory2.556s,
guarded repeated one-root F1 probe15.298s; no production/yield claim. Evidence is
`bl-labels-f0/mac-functional-02/entrypoint-proof.json`.

Pack01's entire D256 allocation was rejected before data because seeds collided
with historical training-exclusion inventories. Pack02 uses a fresh namespace
from the equivalent published source pin; its full seed audit passed.
`SEED_ALLOCATION_ERRATUM.json` preserves the rejected allocation's receipt hash,
new pin and executable equivalence. No world was swapped after an outcome.
The full ruler, Canary-B, replay audit, F0 look and D256 remain NOT RUN.
