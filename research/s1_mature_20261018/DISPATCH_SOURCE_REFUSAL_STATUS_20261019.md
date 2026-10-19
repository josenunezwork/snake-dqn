# S1 source refusal repair — 2026-10-19

Status: implementation prepared for offline verification and branch publication.
No S1 dispatch or live Colab account command is authorized in this implementation task.
The refreshed machine-readable status will be written outside the repository at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-source-repair-20261019/status.json`.

The unused committed `S1_SEED_AUDIT.json` symlink is removed. The existing
`SEED_AUDIT.json` remains the authoritative regular file. The committed tree audit
found no other symlink or non-blob selected by the supervisor's fallback defaults.
S1 admission builds every prepared manifest's fallback using
`research.colab_cli.source.prepare`; dispatch repeats those exact local builds before
ledger initialization, account access, or reservation. Manual operational continuation
checks the retry manifest as well. The Colab transport bytes are unchanged.

## Original ledger and pre-allocation reconciliation

The original `admission-run9-20261019/dispatch/ledger-01.json` is preserved. The
available evidence has the before snapshot, logged source refusal and empty job state
directory, but no fresh post-failure zero-assignment snapshot. The implementation task
must not collect one because the user prohibited live Colab commands. Consequently
real reconciliation remains an operator action, after a fresh account snapshot.

`Ledger.reconcile_preallocation_refusal` validates the immutable refusal review,
original manifest, logged source error, historical before snapshot, fresh zero-account
snapshot and empty supervisor state directory. It restores the original job and bound
to `planned`, records the evidence in an event, and preserves acceptance debit, study
start, caps, all liabilities, and both arms' operational replacement allowances.

Preregistration lines 123–129 limit the one operational replacement to a **run lost**
to disconnect, transfer failure, or runtime reclaim before verified final export.
This proven pre-allocation source refusal never started such a run and is recorded
separately, without a `retry-*` job or a `retry_of` entry. No scientific rule changes.

## Refreshed plan and explicit continuation

Prepare into `admission-source-repair-20261019` after the repair commit is published to
`origin/t31-colab-safety-20261018`. Use a new production output root: the old first-job
output already contains the supervisor's staged manifest and must not be reused.
The original seed bank, world bank, audit, pricing, inputs, job IDs, seeds, recipes,
liabilities and acceptance remain frozen.

`run --continue-preallocation --previous-plan ORIGINAL --plan REFRESHED` accepts the
same ledger only after its reservation has been reconciled back to `planned`. It
compares both plans and all ten manifests, allowing only source commit and fresh
output destinations to change. It refuses settled scientific jobs, outstanding jobs,
changed science, changed liabilities, or an unrelated refusal record. It then follows
the ordinary serialized primary-job path without reinitializing the ledger.

The refreshed external directory will contain `reconcile-02.py` (capture a fresh
snapshot, bind the refusal review, and invoke only the ledger reconciliation API) and
`continue-02.cmd` (the exact operator continuation command). Neither is executed by
this task. These preserve the original ledger and use new snapshots/state paths.

## Limits

Offline archive construction proves packaging and inventory acceptance, not successful
runtime execution. The existing default sparse source selection omits the
`research/q0q1_20261017` and `research/redesign_m3c_20261008` packages imported by the
S1 runtime. A GitHub outage followed by fallback execution remains unqualified; this
repair does not change the frozen manifest source-selection contract or its measured
transport qualification. No live runtime, allocation, release or dispatch is verified.
