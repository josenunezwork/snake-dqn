# Spec: S1 inputs via a public GitHub release + staging-failure settlement

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT.
**Owner decision (08:20 PT): publishing the S1 inputs publicly is fine.** Supersedes the
stop condition in `2026-10-19_s1_input_delivery.md`; your audit there stands (all 7 inputs
required on the VM, including `mac-parity-reference.json` for the train.py:335 parity gate).

## Delivery

Claude publishes (you must NOT publish anything) a release on `josenunezwork/snake-dqn`, tag
`s1-inputs-20261019`, with these assets, byte-identical to
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/` (+ seed bank):
`B8_s3_final.pth`, `demo.tar.gz`, `league.tar.gz`, `mac-parity-reference.json`,
`parity-bank.json`, `q0b-l4.json`, `SEED_BANK.json`.
URL form: `https://github.com/josenunezwork/snake-dqn/releases/download/s1-inputs-20261019/<name>`.

1. Manifests carry each large input as a URL + pinned SHA-256 + size (no Contents upload).
   The VM fetches each with bounded timeouts/retries (idempotent GET to a temp file), verifies
   size and SHA-256, atomically renames into the inputs dir, and refuses before training on any
   mismatch/absence. Small evidence files (`cli-acceptance-result.json`, <=100 KB) may still go
   via Contents within existing caps.
2. Mac preflight before reservation: HEAD/GET-range each URL (or full download to a temp dir)
   and verify size + SHA-256 against the pinned values; refuse if any asset is missing.
3. Keep every supervisor safety property and gate. Changes under `research/colab_cli` are
   expected and invalidate the run-9 acceptance binding; Claude re-runs final-only acceptance
   after review. Fake-Colab tests: good fetch, hash mismatch refuses pre-training, network
   timeout refuses cleanly with stop verified.

## Settlement (ledger-01)

`.../admission-run9-20261019/dispatch/ledger-01.json` has outstanding `s1-mature-n3-223213833`
(receipt `.../admission-source-repair-20261019/production/training/s1-mature-n3-223213833/receipt.json`:
INCOMPLETE, stop_verified, 496.2 s L4, no exports, input-upload timeout, no trainer start).
Add a narrowly scoped ledger method that settles it with actual reconciled spend, classified as a
pre-training staging failure that does NOT consume N3's operational rerun (owner-approved), with
evidence that no trainer process started and no exports exist. The same job id/seed runs next.
Tests for the method and for refusal when the receipt shows trainer start or exports.
Leave a ready-to-run settle script/command (Claude runs it), like reconcile-02.py.

## Then

Re-prepare the S1 plan offline for the new published commit (same ledger continuation), add the
staging-time preflight from the previous spec (refuse if staging could exceed its cap), and write
the exact next commands: settle, final-only acceptance, continue-dispatch.
All `tests/test_colab_cli*.py` + `tests/test_s1_*.py` green; Black/isort/flake8.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch
`t31-colab-safety-20261018`); commit and push when green. No live Colab, no dispatch, no tmux,
no publishing. If the release assets are not reachable yet when you test, use fixtures and let
Claude run the live Mac preflight.
