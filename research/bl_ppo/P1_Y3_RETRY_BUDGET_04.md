# P1 Y3 mechanical retry — budget stop, 2026-10-22

Outcome: **BLOCKED_BUDGET_JUDGE_REQUIRED**. The remaining envelope cannot
fund one complete retry at the unchanged source and replacement forecasts.
Funds remain; the invoice is insufficient, rather than literally zero.
No new attempt or canary manifest was created, and nothing was dispatched.
An independent judge must rule before retry preparation continues.

The requested initial `git pull origin main` fast-forwarded this worktree to
`6e4465ded60269e705ca7c1a77798c33d9315cfe`. The previous admitted source remains
immutable at executable pin `8fd1e5b43f526c89f8ecbc5d3e9a75bfbc1331af`.
Its terminal receipt is INCOMPLETE with `stop_verified=true`, no cleanup error,
and failure class SUPERVISOR. The operator log contains worker EOFError,
cli_guard kernel HTTP 404, and EXIT=1. This is mechanical failure evidence,
not a Y3 result or a successful continuation.

## Receipt reconciliation

Ran the real `tools/ops/spend.py` over all four real supervisor receipts under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1`.
Preparation/producer receipts and fake-provider local test receipts are excluded.
Each real receipt has verified stop. Own spend is exactly elapsed seconds times
`accounting.own_rate_units_per_hour` divided by 3600. The old 2u/h prices are
retained, even though calibration observed 1.54u/h. Provider aggregate balance
changes are not substituted for own spend.

| Receipt relative to `bl-p1/y3/` | Elapsed seconds | Own rate u/h | Own units |
|---|---:|---:|---:|
| `source-02/colab-output/receipt.json` | 2914.989636 | 2.00 | 1.619438687 |
| `source-latebody-02/colab-output/receipt.json` | 467.852995 | 2.00 | 0.259918330 |
| `source-latebody-03-canary-01/canary-output/receipt.json` | 580.994416 | 1.54 | 0.248536500 |
| `source-latebody-03-canary-01/source-output/receipt.json` | 2973.108546 | 1.54 | 1.271829767 |
| **Total** | | | **3.399723284** |

Against 8u, own-spend remainder is **4.600276716u**. The unchanged source
reservation is 4.81u (75min estimate, 150min timeout, .96u reserve), plus
1.29u for replacement (20/40min, .26u reserve), both at maximum 1.54u/h and
zero tolerance. Required total **6.10u** exceeds that remainder by
**1.499723284u**, before any new canary.

The family canary had separate <=1u authorization. Even excluding its actual
.248536500u debit from Y3 gives only **4.848813216u** remaining, a
**1.251186784u** shortfall for source plus replacement. It therefore cannot
change the stop decision. `P1_Y3_ADMISSION_ERRATUM_03.md` preserves both
forecasts and forbids shortening them to squeeze the invoice.

The frozen historical conservative debit is not refunded or amended here.
Its policy takes the larger of own spend and gross account decrement, rounded
up to cents. Retaining that policy gives Y3 charges 1.62 + .28 + 2.52 =
**4.42u**, leaving **3.58u** when the separately funded canary is excluded.
Charging its rounded .25u here too gives 4.67u and 3.33u remaining. The failed
source's 2.52u gross decrement includes concurrent account activity; it is
not claimed as that source's own spend. These are separate accounting views,
and every view fails the required invoice. A judge must resolve any prospective
funding change; this record authorizes no cap increase or reserve consumption.

## Canary and dispatch boundary

Existing canary evidence cannot be reused against current main. A direct local
`research.colab_cli.canary.check` of the old admitted manifest refuses with
`ambiguous job device: explicitly declare cpu or cuda[:index]`. Independently,
the executable transport inventories differ in `research/colab_cli/canary.py`.
The old pin's transport tree is `8747f3067bc27bb5073b11e28f1f1b218baff751`;
current main and the qualification registry use
`40985ca47b3c1f82cdab2ee77849daf937e71865`. The current family gate requires
the proof, current broker, and manifest pin to share the transport identity.
An old broker would not satisfy the wrapper's qualification check. No device
override, old-broker bypass, source repin, or fresh live canary was attempted.
Funding and transport compatibility need resolution before a new preparation.

The create-only artifact root is:

`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-04-budget-stop-01/`

It contains `spend.txt`, `reconciliation.json`, and `DISPATCH.txt`, each with
verified SHA256 and completion sidecars. The committed sibling JSON index binds
them. **Manifest: none. Dispatch command: none.** DISPATCH contains genuine
BLOCK lines for the insufficient invoice and incompatible canary. It cannot
be mistaken for a dispatch-ready attempt. The previous manifest, receipts,
bundles, and dispatch instructions were preserved.

Validation: `tests/test_ops_spend.py` **10 passed**; independent read-only
review confirmed the invoice and transport mismatch; all new sidecars and
receipt hashes were rechecked. No product or transport code changed, so no
training smoke or broad suite was needed. No Colab command, allocation,
dispatch, runtime stop, serving change, or judge ruling occurred. The forbidden
study directory was never opened.
