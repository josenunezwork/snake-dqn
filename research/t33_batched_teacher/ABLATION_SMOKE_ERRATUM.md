# Initial ablation smoke invalidation

The initial `ABLATION_SMOKE.json` and `ABLATION_BUDGET_SMOKE.json` are retained
unchanged but INVALID as evidence of the value-leaf comparison. In the initial
`reference.py`, an inner callback named `leaf` shadowed the leaf-mode string.
Consequently the rows named `*-value` actually used masked-max Q.

The callback is renamed `leaf_selector`, and a regression test supplies Q
values whose dueling mean and masked maximum differ. The corrected real-budget
smoke is written separately as `ABLATION_BUDGET_SMOKE_V2.json`. No full
calibration has run and no recipe was selected from either initial receipt.
