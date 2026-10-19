# Spec: S1 input delivery without per-job Contents uploads

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT. Owner approved this direction
("deliver S1 inputs via a fast hash-verified fetch, drop anything the VM doesn't need, settle
the stranded reservation without spending N3's rerun, re-run final-only acceptance, re-dispatch").

## Evidence (dispatch-02, 07:46-07:54 PT)

Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-source-repair-20261019/production/training/s1-mature-n3-223213833/receipt.json`
(`stop_verified: true`, ~496 s L4). Error: `TransferError(... league.tar.gz ... total upload time cap reached)`.
Per-job Contents uploads: B8_s3_final.pth 6.5 MB in 18 s (365 KB/s); demo.tar.gz 3.9 MB in 293 s
(13 KB/s); then the 300 s `UPLOAD_TOTAL_TIMEOUT_SECONDS` ran out on league.tar.gz (21 MB).
Inputs per job (`.../inputs/`): B8_s3_final.pth 6.5 MB, demo.tar.gz 3.9 MB, league.tar.gz 21 MB,
mac-parity-reference.json 28 MB, parity-bank.json 2.2 MB, q0b-l4.json 52 KB, seed bank,
cli-acceptance-result.json. Ten jobs. Contents upload throughput is 13-365 KB/s and unstable.

The runtime already fetches source from the PUBLIC GitHub repo
`josenunezwork/snake-dqn` (`research/colab_cli/source.py`).

## Required outcome

1. **Drop what the VM does not need.** Audit every S1 input against what the remote trainer
   actually reads. Anything only the Mac uses (likely `mac-parity-reference.json`) must not be
   shipped. Record the audit.
2. **Fast, hash-verified delivery for the rest.** Each input is fetched on the VM from a fast
   source and verified against the manifest's pinned SHA-256 before use; mismatch or absence
   refuses before training. Pick the mechanism after reading the code. Hard constraint: the repo
   is public — **do not publish any artifact that is not already public** (e.g. bytes already
   committed in the public repo at a commit are fine; deterministic regeneration on the VM from
   public source + pinned seeds, verified by hash, is fine). If some required input can only be
   delivered by newly publishing it or via a new private channel, stop and say which file and
   why, with sizes, instead of choosing for the owner. Contents upload may remain only for small
   files (<= ~100 KB) within the existing caps.
3. **Settle the stranded reservation.** `ledger-01.json` (in
   `.../admission-run9-20261019/dispatch/`) has outstanding `s1-mature-n3-223213833` from an
   allocated attempt that failed in input staging before training started (no trainer
   process, no metrics, no efficacy look). Settle it through the ledger API with the receipt,
   charging its actual reconciled spend, classified as a pre-training staging failure that
   does NOT consume N3's one operational rerun; the same job id/seed runs next. Cite the
   prereg lines and the evidence (no trainer start, no exports) in the record. If the ledger
   cannot express this safely, add a narrowly scoped method with tests.
4. Any change to `research/colab_cli` is allowed for this, but keep every safety property and
   gate; note that it invalidates the run-9 acceptance binding (Claude will re-run final-only
   acceptance after review). Keep the change minimal and covered by Fake-Colab tests,
   including a test that a hash mismatch refuses before training.
5. Re-prepare the S1 plan offline for the new published commit; continuation on the same ledger.
   Add a preflight that estimates per-job staging time from bytes and refuses if staging could
   exceed its cap.
6. All `tests/test_colab_cli*.py` + `tests/test_s1_*.py` green; Black/isort/flake8.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch
`t31-colab-safety-20261018`); commit and push when green. No live Colab, no dispatch, no tmux.
Do not change the pre-registration's science. Final message: the delivery mechanism, the input
audit, the settlement record, test counts, the exact next commands (settle if not done,
acceptance, continue-dispatch), and anything unverified or needing an owner decision.
