# P1 Y3 source-latebody-05 kernel-probe and watchdog transport repin

Prepared on 2026-10-22; **NOT DISPATCHED**. This sibling supersedes only the
operational pack pointers in `P1_Y3_WARMUP_REPIN_05.md`. All previous packs,
registrations, reviews and receipts remain immutable.

The required initial `git pull origin main` fast-forwarded the clean wp3 worktree
to `0e7bbd307df6cbec8d0ed1aa34c188680d47a271`. Both manifests pin that published
main commit, including its canary. Qualified transport tree:
`e8e207d24472decb1eb586119c53ff1e5707edf0`, per the current qualification registry.
The later concurrent advice-only main commit
`48f776b20e853ecf653d3045db9775eb09bc0540` was integrated before publication;
it changes no P1 or transport executable bytes. The acceptance binding is `acceptance-1792707441`, SHA-256
`fc7d6a907a8fb314caf7e796d72585f09c3a0816ca60d51c577991001c440214`.

New immutable prepared pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-kernel-watchdog-repin-20261022`.
The sibling `P1_Y3_KERNEL_WATCHDOG_REPIN_05.json` binds actual evidence hashes.
Only executable commit, qualified transport, acceptance bindings and fresh sibling
paths changed. Science, seeds, rules, reviewed study source identity, ledger,
reconciled debit and resume declarations are preserved. No new research decision
or budget amendment is introduced.

| Lane | Forecast min | Timeout min | Cap u | Reserve u |
| --- | ---: | ---: | ---: | ---: |
| Family canary | 15 | 30 | 2.00 | 0.40 |
| Y3 source | 75 | 300 | 9.62 | 1.92 |
| Replacement reservation | 30 | 120 | 3.86 | 0.78 |

Owner 2x rule remains frozen, including base/effective envelopes; total 25.50 u,
prior conservative debit 4.67 u, rate ceiling 1.54 u/h with zero tolerance and
no other assignments. Replacement remains reserved, not admitted. Both resume
blocks retain the verified empty ledger and maximum two outcome-blind completion
retries. No retry was launched.

Verification compares actual controls, budgets, requirements, exports, resume,
ledger and debit to the latest previous pack. Actual `tools/ops/dispatch.py`
admission runs with the real qualification checks and `launch.sh validate`, in
an isolated temporary ops root. A forced abort before entering the allocation
lock and a forbidden `start` prevent allocation. The source template correctly
refuses without its genuine SHA-bound canary receipt. Review, final family
preflight, source/canary upload preflights and regression evidence are bound by
the JSON index. Preparation is not live canary success.

Use the fresh wrapper names and `DISPATCH-kernel-watchdog.txt`; the generator's
original `DISPATCH.txt` remains preparation evidence. Require fresh empty account
inventory and the registered calibrated rate before each actual dispatch:

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-wp3
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-kernel-watchdog-repin-20261022
ACCEPTANCE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/warmup3-requalification-20261022/requalify-80auhwsd/acceptance-1792707441/acceptance-result.json
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-kernel-watchdog-canary-20261022 "$PACK/family-canary/canary-manifest.json" --broker "$REPO" --cwd "$REPO" --dispatch-txt "$PACK/DISPATCH-kernel-watchdog.txt"

# Continue only after the real passing canary receipt, verified exports and release.
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_family_canary admit --root "$PACK/family-canary" --receipt "$PACK/family-canary/canary-output/receipt.json" --acceptance "$ACCEPTANCE" --review "$PACK/independent-artifact-review.json"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-kernel-watchdog-source-20261022 "$PACK/family-canary/source-admitted.json" --broker "$REPO" --cwd "$REPO" --dispatch-txt "$PACK/DISPATCH-kernel-watchdog.txt"
```

The source-admitted file and both output directories intentionally remain absent.
No provider call, Colab allocation, dispatch or live scientific result occurred.
No tmux session, released checkpoint, serving pin or pre-existing hash-bound file
was modified. The prohibited study directory was never opened.
