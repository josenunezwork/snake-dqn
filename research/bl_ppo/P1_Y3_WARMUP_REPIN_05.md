# P1 Y3 source-latebody-05 warm-up transport repin

Prepared and locally validated on 2026-10-22; **NOT DISPATCHED**.
This sibling handoff supersedes only the operational pack pointers in
`P1_Y3_HANDOFF_05.md`. Original packs, registrations, reviews and receipts remain immutable.

The initial required `git pull origin main` fast-forwarded the clean wp3 worktree
to `96e9b7769dbf0d4ce428dadd0c9320988fc864ca`. Both regenerated manifests pin that
published main commit. Its qualified transport tree is
`213f25be1b6a9de5b42a02c387eb86304e3731fe`. The later concurrent DENSITY-2 handoff
commit `16d09536926f0e184b23f13ec5b31103701a9fc4` was integrated before publication;
it changes neither P1 nor transport executable bytes.

New immutable pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-warmup-repin-20261022`.
The sibling `P1_Y3_WARMUP_REPIN_05.json` binds its actual evidence and commands.

Only commit, qualified transport, acceptance bindings and sibling paths changed.
The 31-entry reviewed study source identity, science, seeds, rules, ledger and
reconciled debit are preserved. Canary and source share identical uploaded input
inventories. The owner 2x rule remains frozen:

| Lane | Forecast min | Timeout min | Cap u | Reserve u |
| --- | ---: | ---: | ---: | ---: |
| Family canary | 15 | 30 | 2.00 | 0.40 |
| Y3 source | 75 | 300 | 9.62 | 1.92 |
| Replacement reservation | 30 | 120 | 3.86 | 0.78 |

Total envelope remains 25.50 u, prior conservative debit 4.67 u, rate ceiling
1.54 u/h with zero tolerance and no other assignments. Replacement remains
reserved, not admitted. Both resume blocks retain the verified empty ledger and
maximum two outcome-blind completion retries. No retry was launched.

Local verification checks the real generated payloads, sidecars, qualification
registry and final-only acceptance. The canary validates; the source template
refuses without its real SHA-bound family-canary receipt. The actual
`tools/ops/dispatch.py` admission prefix passed using an isolated temporary ops
root, the real qualification checks and `launch.sh validate`. The allocation lock
was replaced with an abort before entry, and `start` was forbidden, so the test
cannot allocate. The bound script and validation log are in the new pack.
Independent read-only review PASS has no unresolved findings; real family
`preflight()` passes with the materialized review. Both upload preflights pass.
Detailed regression commands and results are in the JSON index.

Use `DISPATCH-warmup.txt`, which has fresh wrapper names. The generator's original
`DISPATCH.txt` is retained as preparation evidence. Check fresh empty account
inventory and the calibrated rate before each actual dispatch. Execute in order:

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-wp3
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-warmup-repin-20261022
ACCEPTANCE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/warmup-requalification-20261022/requalify-v9t3s_ai/acceptance-1792703228/acceptance-result.json
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-warmup-canary-20261022 "$PACK/family-canary/canary-manifest.json" --broker "$REPO" --cwd "$REPO" --dispatch-txt "$PACK/DISPATCH-warmup.txt"

# Continue only after the real passing canary receipt, verified exports and release.
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_family_canary admit --root "$PACK/family-canary" --receipt "$PACK/family-canary/canary-output/receipt.json" --acceptance "$ACCEPTANCE" --review "$PACK/independent-artifact-review.json"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-warmup-source-20261022 "$PACK/family-canary/source-admitted.json" --broker "$REPO" --cwd "$REPO" --dispatch-txt "$PACK/DISPATCH-warmup.txt"
```

The source-admitted file and both output directories intentionally remain absent.
No provider call, Colab allocation, dispatch or live L4/scientific result occurred.
No tmux session, checkpoint, serving pin or pre-existing hash-bound artifact was
modified. The explicitly prohibited study directory was not accessed.
