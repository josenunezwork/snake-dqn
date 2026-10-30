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

Both canary and main fitting record a diagnostic lowest-doom legal action from
the first training logits of each arm. Negated detached logits are legal-masked
and reduced with `Tensor.argmax(dim=-1)` on the model device; ties choose the
first legal index. The decision is printed and retained in the prediction export.
It never feeds fitting, sampling, assessment scores or endpoint selection. This
is a real six-action CUDA decision attributable to `run.py`, as required by the
unchanged `research/colab_cli/canary_observer.py`; availability checks and CUDA
training alone do not satisfy that observer. Fresh runtime pins require fresh
same-pin/input canaries and admission. Consumed job/output identities and old
failure receipts cannot qualify the changed pin. The A100-only clarification
continues to apply when the fresh matching H100 probe records allocation 503.

`prepare admit` accepts genuine same-pin H100 and A100 family receipts, verifies
them through the unchanged supervisor, and rejects a measured fit projection
above the frozen 45-minute worker envelope. Under
`REGISTRATION_OPERATIONAL_CLARIFICATION_02.md`, `--a100-only` instead requires a
manifest-bound H100 allocation-unavailable receipt and a genuine stopped A100
proof. It emits an A100-only `fit.json` in a fresh `--admission-dir`, keeping
the old runtime pin and upload inventory. H100 unresolved allocation records
remain unresolved; the broker's execute-time checks still apply.
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
gameplay or serving implications. Both admission and aggregate now require
`--prelook`, produced by `python -m research.doom_auc_smoke.prelook --packet ...
--bank ... --output ...`. This reads bank metadata only, counts L>=600 support
per assessment world, and binds the pre-look amendment and root-kind joins.
The census cannot verify positive/negative class support without labels.

`REGISTRATION_PRELOOK_AMENDMENT_01.md` adds descriptive ordinary/death-lookback
AUC splits at the same Mac look. The root sample is death-lookback-enriched:
GO is an UPPER BOUND learnability screen, not natural-state danger AUC (nor a
mathematical upper confidence bound). Interpretation also carries the 630,182
parameter / 2,048-root overfitting limitation: 46.875 training-set equivalents
and final update only. These disclosures do not alter any primary bar.

Use `tools/ops/dispatch.sh --dry-run` on the canary packet before operator launch.
No dispatch or merge belongs to implementation. Exact prepared paths, commands,
test results and outstanding independent review are retained in
`../snake-dqn-artifacts/t34-sol/doom-auc-smoke/SUMMARY.md` (relative to project
parent). The source transport is the unchanged qualified supervisor. Its genuine
canary must establish staging and execution timing; local projections do not
qualify a runtime or prove GPU throughput.
