# S1 mature collection study

This is a new study, separate from Q1. Its registration remains
`DRAFT_PENDING_OWNER_REVIEW` until Claude approves the complete package.
The owner's two-variant budget authorization is recorded verbatim in
`OWNER_AUTHORIZATION.json`; that file does not bypass registration, acceptance,
or complete-budget admission.

**Current admission: REFUSED.** S1 exports only terminal online/target weights
and decision evidence, with a 64 MiB aggregate artifact cap per attempt. No
mid-run checkpoint, replay transfer, or exact resume is part of this study.
See `FINAL_EXPORT_STATUS_20261018.md` for the current design, measurements and
conditional forecast. The trainer's full-state barrier correction is preserved
for other consumers. Earlier barrier/capacity/admission receipts remain
historical evidence; they do not qualify this final-only adapter.
No S1 data or Colab allocation was produced by this change.

Both arms fine-tune the same released bare B8 initializer with fresh Adam and
empty replay. N3 retains the Q1 static-pool collection package; N1 uses Q1 D's
task-aligned terminal-hero collection. Both use the mature schedule from decision
zero. Five paired training seed blocks each produce two 8M finals. One Mac look
evaluates both finals and shared B8 on 640 worlds, for 1,920 policy episodes.

The preregistration fixes the endpoint, statistical rules, safeguards, history,
and fixed-N exception. A development GO funds only preparation of a separately
authorized fresh Mac strict gate. It does not change serving.

## Package boundaries

- `bank.py`, `WORLD_BANK.json`, `SEED_AUDIT.json`: shared allocation, training
  namespaces, disjoint operational fixtures, and historical collision coverage.
- `manifests.py`, `train.sh`, `train.py`: ten serialized L4 manifests, the real
  M3-C trainer adapter, progress, and immutable final-artifact transport.
- `evaluate.py`, `analyze.py`: single-thread Mac worker pool, resumable hashed
  world units, fixed-order merge, final-look statistics and decisions.
- `ledger.py`: durable study-wide reservations and receipt reconciliation.
- `admission.py`, `run.py`, `launch.sh`, `rerun.py`: complete-plan admission,
  serialized execution and explicit manual operational replacement after all
  prerequisites pass.

The protected `research/colab_cli` and `research/q0q1_20261017` implementations
are consumed as read-only dependencies. Q1's historical ledger and A2/A3/A5
evidence remain preserved; S1 does not amend or reset them.

## Budget rules

| Dimension | Nominal | Usable after 20% reserve |
| --- | ---: | ---: |
| Compute units | 60 | 48 |
| GPU allocation hours | 25 | 20 |
| Mac worker hours | 24 | 19.2 |
| Elapsed hours | 48 | 38.4 |

CLI acceptance is included, with at most four units. Charge allocation request
through confirmed backend disappearance, including idle, installation, failures,
exports, and cleanup. A job's full liability is reserved before the supervisor
can request it. Unknown intervals, missing receipts, stale account evidence,
changed source/input hashes, and an incomplete remaining-study forecast block
further dispatch. First cap wins. One expressly prepared from-scratch replacement
per recipe arm is allowed for a verified operational loss before final export
and before any efficacy look. Both attempts charge the same envelope. No
automatic retries, extensions, seed replacements, cap raises or reserve borrowing.
The twelve worst-case attempt slots plus acceptance reserve 22.6 GPU-hours,
above 20 usable; admission correctly refuses that hard upper bound.

The owner-supplied account context is 1,174.37 units, zero usage rate, and zero
active assignments. It is historical context: the launcher requires fresh
read-only supervisor account snapshots before admission and every allocation.

## Operational handoff

See `FINAL_EXPORT_STATUS_20261018.md` for current verification and admission
dependencies;
`IMPLEMENTATION_RECEIPT.md` retains the original operator sequence. No Colab
allocation is part of package preparation. Existing run artifacts are create-only
and must never be deleted to make a rejected command pass.
