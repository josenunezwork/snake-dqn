# P1 amendment 06 source-repair handoff 10 — 2026-10-23

Verdict: **SOFTWARE IMPLEMENTED AND VERIFIED; HOLD_TRANSPORT_CONFLICT; NOT DISPATCHABLE.**
The registered critic PASS is preserved. No production actor/refit/gate, Colab
allocation, replacement or qualification was run. Y3 is still not PASS.

Executable pin: **ec5ef9e1156a6b226ce3e6e905823d97051202d3** on
`p1-snapbound-20261022`, published on `origin/main` and the remote branch.
Colab CLI tree: **d0d92bdb572b770284a1fea6152417ffa5f06bea**, unchanged.
Authority: `JUDGMENT_P1_FINALISATION.md`, SHA-256
`fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc`.
Implementation sibling: `P1_Y3_SOURCE_REPAIR_AMENDMENT_06.md`.

## What is complete against §3

| Ruling line | Implemented evidence |
| --- | --- |
| 1: empty ledger, fresh namespace, same allocation | Empty uploaded source/canary ledgers; source namespace `bl_p1_y3_pack04_repair06_c3a06a5f64f4`, allocation `source_latebody_06_e8_04_pack04`. Attempt IDs derive from fresh output roots. |
| 2: hash-verified complete origin set | Exact dbdc10c6 origin and 9b1f309a status; failed receipt aaa9f106 and audit 738c4a5b; `_snapshot_catalog` closure plus explicit late-body row/index evidence catalog, all sidecars and receipt/audit checks. |
| 3: cross-pin identity | `cross-pin-diff.json` records real 5d877123-to-pin diff/stat and per-file hashes; runtime verifies the shipped closure without assuming Git history exists remotely. No src/ or algorithm modules changed. |
| 4: skip origin/refit/gate | Imported source bypasses `_run_one(origin)`; finite ValueError guards cover refit collection, solve, gate and swallowed entry; counters and critic subtree must remain equal. |
| 5: cadence plus suffix | Existing operational validator unchanged. Real software fixture reaches genuine scaled 900-s actor capture plus >=3 updates, >=1 episode and >=1000 actions. No live 900-s claim is made. |
| 6: receipt | Source producer copies verbatim critic subtree/digest, records IMPORTED_VERIFIED_ORIGIN, superseded receipt/audit and before/after frozen counters. Software fixture verifies it. No production COMPLETE_SOURCE_CONTINUATION exists yet. |
| 7: prefix witness | Fixed 49152-action chain witness, immutable namespace/allocation witness ledger and source receipt; mismatch blocks replacement without voiding source. Fixture scales the count and checks exact chain equality. No live prefix match exists yet. |

Replacement and qualification are **NOT MATERIALIZED**. Their fresh price-only
held records fix the invoices and dependencies; their inherited source input
bundle is deliberately not a runnable next-stage pack. A real receipt-bound
replacement sibling must be built after COMPLETE_SOURCE_CONTINUATION, prefix
MATCH, source exports/release, Claude's line check and debit reconciliation;
its assignment must differ and both replays must PASS. A real 1M sibling needs
full Y3 PASS, Claude/Codex-V, a new namespace/allocation, retry=0, original origin,
target 100M, requested pause 1000000 / accepted boundary 1000064 and cadence1800.
`Y3_PASS_BOUNDARY_ONLY` cannot fund qualification. The complete-job finalizer
requires measured final transfer/release and retains the <=46h AND <=120u kill.

## Exact artifacts and preflight result

Final immutable pack: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02`.
The earlier `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023` remains immutable and
held; it predates the attempt-identity correction. Never dispatch that sibling.

`mac-validate-only-preflight.json`: **PASS_VALIDATE_ONLY** on real data, repair
**2.174397s**, canary
**2.193352s**. Both use the real runtime module,
verify B8 and the actual origin closure, stop before `_execute`, create no output
directory and execute zero actor/refit/gate calls. Exact upload bytes:
**595,321,881**. The full subprocess argument arrays,
control hashes, output and times are in that create-only receipt.

`upload-preflight.json`: **HOLD_QUALIFIED_UPLOAD_LIMIT**. The real pinned source
preparer and file hasher ran at actual bytes; local wall time was16.594098s.
Normal upload forecast13515.595s exceeds the qualified150s preflight budget;
fallback forecast **13545.411953s** also conflicts
with the unchanged300s initial staging deadline and canary transfer proof.
This is historical-rate forecasting, not a measured live upload.

`measured-forecast.json`: **HOLD_JUDGE_REQUIRED**. Complete repair forecast
**263.703749min**, doubled
**527.407498min**, exceeds200min. The6.16u/200min
invoice is unchanged. Ruling §3 says twice the measured preflight forecast over
200min returns to a judge before dispatch. Do not bypass qualification, alter
the transport pin, invent a hosted URL, or clear these immutable fuses.
An admitted HTTPS path or newly judged/requalified transport solution is missing.

The current qualified family proof also binds exact input hashes. A fresh retry
namespace changes control/bundle bytes, so reusing the first canary after such a
change needs a reviewed resolution; the extra2u retry-canary line is funded only
when transport identity changes. No automatic retry workflow is claimed proved.

## Prices and ledger

All caps/timeouts below already apply2x exactly once; rate1.54u/h, zero tolerance,
zero other assignments, `device=cuda:0`, L4 only, first cap wins.

| Phase | Effective cap/reserve u | Timeout min | Estimate min | Mechanical completion retries |
| --- | ---: | ---: | ---: | ---: |
| Fresh exact-pin canary | 2.00 / .40 | 30 | 15 | No generic retry; changed-transport retry canary only |
| Imported repair | 6.16 / 1.03 | 200 | 50 | 1 |
| Replacement, both replays | 3.86 / .78 | 120 | 30 | 1 |
| Qualification, separate invoice | 4.00 / .80 | 120 | 40 | 0 |

Y3 envelope34.50u, charged10.11u, happy path22.13u, worst case34.15u;
.35u rounding slack funds no attempt. `prepared-envelope-ledger.json` is a
prepared ledger for Claude, not a published live debit reconciliation.
Carry WARNING_UNEXPLAINED_DEBIT and max(own,gross) conservative cent rounding;
no refund. There was zero new spend. Only verified SUPERVISOR/transport failures,
verified stop, null cleanup error and missing work permit the single fresh
completion. CHILD_NONZERO_EXIT/finaliser failure, cap/timeout, scientific failure
and a second failure stop exactly as ruled; no raised cap or hidden retry.

## Verification

All pytest commands used the primary venv, thread caps1 and `-p no:xdist`.
`verification.json` contains exact commands and complete selection counts.

- 153 passed /69.04s: import, runtime, qualification, pack/admission and dispatch selections.
- 5 passed /148.47s: entire existing Y3 resume-entrypoint suite, including exact
 pause never seeding a suffix, both replays and interrupted source recovery.
- 72 passed /1.23s: transport guard/schema plus admission tests.
- 10 passed /.81s after the helper-only fresh-namespace correction.
- Independent Codex-V reran16 import/admission tests successfully and inspected
 the real preflight and artifacts. Its signed-off payload is the sibling
 `P1_E8_04_SOURCE_REPAIR_CODEX_V_10.json`; its verdict remains HOLD.
- Black100, isort Black profile, flake8 and `git diff --check` passed on all12
 changed Python files. Source, refit, learner, model, collector and late_body
 science bytes are preserved; the run.py changes are import guards/identity and
 tape witness only, in addition to the already-ratified suffix guard.

## Ordered operator forms — BLOCKED, do not execute

The user-required wrapper runs its adjacent policy file. `--broker` alone cannot
update it. After a later transport ruling and fresh reviewed packs, first do:

```bash
git -C /Users/josenunez/Projects/ml/snake-dqn-acctfix fetch origin
git -C /Users/josenunez/Projects/ml/snake-dqn-acctfix merge --ff-only origin/main
```

The current wrapper checkout was clean at992ddc49 and was an ancestor of main;
root did not modify it. It needs the narrow ruling-SHA-bound retry1/0 exception.
The following actual held forms are recorded in dependency order and intentionally
refuse before operational side effects. They are not admitted dispatch commands:

```bash
/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh p1-repair10-canary /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/canary-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/DISPATCH.txt

/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh p1-repair10-repair /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/repair-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/DISPATCH.txt

/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh p1-repair10-replacement /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/replacement-held-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/DISPATCH.txt

/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh p1-repair10-qualification /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/qualification-held-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-02/DISPATCH.txt
```

A later admissible path is Claude line check+campaign log+published ledger, exact
proof==current==manifest qualified tree, fresh usage/sessions and no other
assignments at <=1.54u/h, canary terminal/export/release, repair, then freshly
materialized replacement, then full-Y3-gated qualification. Codex never dispatches.

No tmux jobs, released checkpoints, serving pins, closed study, l2mac worktree,
prior hash-bound documents/receipts or existing untracked canary pack were changed.
B8 s3 remains served; release still belongs to the owner.
