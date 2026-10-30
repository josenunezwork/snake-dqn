# T34 doom AUC smoke

Prepared supervised representation diagnostic. Read `REGISTRATION.md` before use.
The frozen smaller spatial U-Net and frozen-B8-feature MLP each receive 3,000
updates on the existing C/T train worlds. Both predict per-action H300 death risk.
Assessment labels stay on the Mac; the Colab worker exports final logits only.
The primary result is the minimum of late-game per-action and B8-action AUC,
with a parent-world 95% bootstrap CI: GO >= .85, GRAY [.75,.85), KILL < .75.

`data_mac.py` verifies the original bank, all eight outcome shards and canonical
B8 checkpoint hashes. It reconstructs head-first sparse full-arena observations
and freezes standardized B8 post-fusion features. Anchors lack full snapshots and
H300 labels and are excluded. The train target uses selection suffixes; the Mac
assessment uses independent scoring suffixes, retaining their binary outcomes.

`prepare build` seals compressed tensor inputs, a Mac-only assessment archive,
the split audit, empty ledger, clean-import proof, H100-first canary and A100
canary manifests, and a pending main manifest. It performs no fitting. The actual
Mac `run --mode preflight --device cpu` uses three tiny real train batches and
assessment inference without scoring. The GPU family canary uses synthetic data
at full board size, three batch-32 optimizer steps per arm, and exactly the main
input inventory. It does not fit the real bank. All timings are recorded.

`prepare admit` accepts genuine same-pin H100 and A100 family receipts, verifies
them through the unchanged supervisor, and rejects a measured fit projection
above the frozen 45-minute worker envelope. Only then does it publish `fit.json`.
Whole-session base/effective timeouts are 60/120 minutes, separate from useful
compute. `fit.pending.json` intentionally fails supervisor admission until those
receipts exist. A dry-run never produces a canary receipt.

`evidence.py` publishes immutable full-state snapshots with SHA/completion-last
sidecars. `resume.py` builds an outcome-blind union from verified stopped receipts,
selects only the latest eligible snapshot per unfinished model, and prepares a
fresh completion packet with a bound ledger. It requires the full original-through-
current attempt history, a verified mechanical failure and missing work, and
allows at most two completions. New same-input canaries are required for completion
admission; the runtime refuses cross-hardware/library/source training resume.

`aggregate.py` alone scores held-out labels on the Mac after receipt-verified final
predictions from both arms exist. It reserves the study's fixed scoring-look file
before reading labels; changing the output directory cannot take a second look.
It refuses duplicate units and foreign identities and reports paired bootstrap
differences against the B8-feature baseline. GO is a learnability screen, without
gameplay or serving implications.

Use `tools/ops/dispatch.sh --dry-run` on the canary packet before operator launch.
No dispatch or merge belongs to implementation. Exact prepared paths, commands,
test results and outstanding independent review are retained in
`../snake-dqn-artifacts/t34-sol/doom-auc-smoke/SUMMARY.md` (relative to project
parent). The source transport is the unchanged qualified supervisor. Its genuine
canary must establish staging and execution timing; local projections do not
qualify a runtime or prove GPU throughput.
