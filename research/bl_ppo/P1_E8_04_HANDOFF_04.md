# P1 E8 amendment 04 — inventory bound, canary admitted

Status: **READY_FOR_CLAUDE_CHECK; CANARY_ADMITTED_NOT_DISPATCHED**.
Independent supplement `P1_E8_04_CODEX_V_REVIEW_04.json`: PASS, zero unresolved
findings; `V3-INVENTORY` cleared. Source remains gated on a genuine family canary.

Authority: external `JUDGMENT_P1_E8_04_B.md` §§5–7 and owner's 2026-10-22 request.
Claude's inventory receipt SHA256 is
`7a182e7369745c423fd3c53b74cad98becc963c71eef973b998d533d8d2b7e14`.
Its inspection SHA is
`4f08a82ecb99f814cf7af4bf0384f5eb25180cb35d3141e5e9d6624172938ace`;
CAMPAIGN_LOG records the inventory at `39f22d8f59d9df2aa34ea977556c9b16518a2621`.
The 43-row receipt identifies no new executed collection family requiring C3 to reopen.

Fresh immutable sibling root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-04/final`.
Both control coverage blocks, shared input bundle and allocation index03 bind the
receipt path and SHA. Receipt payload/sidecars are also uploaded inside the shared
bundle. Existing -03 bytes, indexes, reviews, audit directories and all numerical
receipts remain unchanged. Pack04 uses new namespace/allocation identities and
empty ledgers; offsets remain canary3010000/source3011000 with131466 burned roots.

Published indexes: `P1_E8_04_ALLOCATION_V3_INDEX_03.json` and
`P1_E8_04_DISPATCH_PACKS_04.json`, each with SHA sidecar and completion marker.
The pack index binds every final top-level payload, the supplement and debit files.

Executable manifest pin remains `5d877123ce8734537ef1f6924d5543be50dacdf7`.
All five bound files equal replay pin `35104156c44cefd59f1113a976400821e6029a47`
and current source. No replay, fit, first-step fixture, allocation audit or timing
run was repeated. Qualification B and C2-prime PASS remain unchanged; the original
control/arm reward-scale mismatch is carried to the separate post-PASS assay.

The reviewed templates retain false top-level dispatch fuses. Their two embedded
runtime controls are provisioned true so the source and canary can share identical
uploaded inputs. After independent PASS, `canary-admitted.json` sets top-level
scientific admission and transport.dispatch_ready true and binds the supplement.
`source-template.json`/`manifest.json` remain false. The local-only additive helper
`e8_04_source_admission.py` invokes existing exact-source family admission and
publishes `source-dispatch.json` true only after a genuine current canary receipt,
verified family exports/release, source review and complete debit reconciliation.
It does not change any pinned scientific or transport file.

All eight historical terminal receipts are charged conservatively (maximum of own
elapsed-rate charge and gross account debit, rounded up per attempt): **7.46u**.
The initial path reserves **22.94u** including canary2.00/source9.62/replacement3.86.
The helper rechecks all attempts including later staging and final sibling outputs,
then adds the real current canary before source admission. The older six-receipt
debit is retained solely for the existing admission function; it is supplemented
by the all-attempt accounting. `WARNING_UNEXPLAINED_DEBIT` remains carried.
These receipts do not replace fresh live usage reconciliation.

Both actual upload preflights PASS at **148.30540393181695s** (<150s), including
the admitted canary preflight. Shared archive SHA256:
`7f7900b0cde5c77ea60fe92cade9ffc6c4e702c67e1494ac6963f0f9ffff901e`.
Transport tree remains `553af3f692515b2bf66173356fe4d099acc96942`, checked against
the current external qualification registry and external PASS acceptance.
L4 only, cuda:0, rate1.54u/h, tolerance0.0. Canary estimate15/timeout30min,
cap2.00u/reserve0.40; source75/300min, cap9.62u/reserve1.92. Replacement3.86u
reserved, unadmitted; frozen25.50u envelope. No second doubling.

Verification, all light and single process with OMP/OpenBLAS/MKL/Numba threads1:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist research/bl_ppo/test_e8_04_source_admission.py research/bl_ppo/test_e8_04_preparation.py research/bl_ppo/test_p1_family_canary.py research/bl_ppo/test_p1_collection_allocation.py
```

**15 passed in1.79s**. New helper/tests: black100, isort black profile and flake8
PASS. Actual helper CLI refuses the absent canary receipt before creating any
source-admission output. Genuine-evidence family preflight passes with the new
supplement; admitted canary supervisor validation and upload preflight pass.
No broad suite or new runtime/transport qualification is claimed. Concurrent
origin/main313d27eb imported unrelated A1 changes; P1/transport bytes unchanged.

Claude must check the supplement against artifacts, then fresh usage/sessions with
zero other assignments, calibrated rate<=1.54u/h, current conservative debit with
the warning, qualification proof==current==manifest pin, and X5/K-1/M1 windows.
No live provider or calendar eligibility claim is made here. The owner requested
light preparation while Mac timing waits; no timing run was started.

Exact operator commands (also in the immutable pack's DISPATCH.txt):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-04/final
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ACCEPTANCE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/canary-staging-requalification-20261022-b/requalify-gljfvzqa/acceptance-1792710644/acceptance-result.json
REVIEW=/Users/josenunez/Projects/ml/snake-dqn-p1snap/research/bl_ppo/P1_E8_04_CODEX_V_REVIEW_04.json
tools/ops/dispatch.sh p1-y3-06-canary-20261022 "$PACK/canary-admitted.json" --broker /Users/josenunez/Projects/ml/snake-dqn-p1snap --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt "$PACK/DISPATCH.txt"
# After passing terminal canary + verified exports + stop_verified + disappearance; refresh live checks.
"$PY" -m research.bl_ppo.e8_04_source_admission --root "$PACK" --receipt "$PACK/canary-output/receipt.json" --acceptance "$ACCEPTANCE" --review "$REVIEW"
tools/ops/dispatch.sh p1-y3-06-source-20261022 "$PACK/source-dispatch.json" --broker /Users/josenunez/Projects/ml/snake-dqn-p1snap --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt "$PACK/DISPATCH.txt"
```

Scientific refusal final; timeout/cap INCOMPLETE, no completion attempt. One
outcome-blind mechanical retry only under the existing verified-stop/no-look
rule and charged+9.62+3.86<=25.50. Generic resume maximum2 is not new funding.
No dispatch, allocation, provider call, spend, tmux modification, serving change,
checkpoint change or protected-study access. `snake-dqn-l2mac` untouched.
