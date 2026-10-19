# S1 mature collection study

This study is separate from Q1. Its prospective registration is
`APPROVED_S1_MATURE_20261018`; the original DRAFT header remains historical.
The owner's two-variant budget authorization is recorded verbatim in
`OWNER_AUTHORIZATION.json`. Dispatch retains the independent review, published
source, fresh account evidence and ledger guards.

**Current offline pricing admission: PASS.** See
`ADMISSION_STATUS_ACCEPTANCE_RUN9.json`, which supersedes earlier admission
status snapshots. Final-only CLI acceptance run 9 is hash-bound by absolute path,
with all required gates observed and every current CLI Python/shell source hash
verified. Its 64 MiB L4 export was verified on the Mac at a conservative
2,389,970 bytes/s. Each S1 attempt reserves its enforced 64 MiB aggregate cap,
so its export liability is at most 28.0793751 seconds. This is a conservative
cap-based bound, not a production bundle-size measurement.

The complete primary reservation is 38.3333333 units / 19.1666667 GPU-hours /
15.8656729 Mac worker-hours / 27.0995031 elapsed hours, within usable caps of
48 / 20 / 19.2 / 38.4 respectively. Optional reruns remain conditional and are
reserved atomically at explicit launch; an unfunded rerun records the affected
arm INCOMPLETE. Earlier barrier, capacity and admission receipts are preserved.
No S1 dispatch is part of offline preparation.

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
Admission reserves ten primary worst-case attempts plus acceptance: 19.1667
GPU-hours, within 20 usable. Each optional rerun is reserved atomically at its
explicit launch against spent, outstanding and all remaining primary/evaluation
liabilities. An unfunded rerun records that arm INCOMPLETE and launches nothing.
The earlier twelve-slot 22.6-hour refusal is historical. Admission refuses if the bound final-only L4 acceptance evidence is absent,
changed, stale or unqualified.

The owner-supplied account context is 1,174.37 units, zero usage rate, and zero
active assignments. It is historical context: the launcher requires fresh
read-only supervisor account snapshots before admission and every allocation.

## Operational handoff

The offline preparation sequence is below. `FINAL_EXPORT_STATUS_20261018.md`
and `IMPLEMENTATION_RECEIPT.md` retain the historical measurements and operator
sequence; the run-9 status supersedes their missing-transfer blocker. No Colab
allocation is part of package preparation. Existing run artifacts are create-only
and must never be deleted to make a rejected command pass.


Prepare locally after committing and publishing the clean source. The explicit
remote-tracking reference defaults to `origin/main`; use the published safety
branch for this acceptance binding. Preparation records that reference and later
plan verification requires the same exact clean published source. The pricing
receipt supplies `cli-acceptance-result.json` as an additional hash-bound input
to every training manifest; the scientific input inventory is unchanged.
All output paths below must be fresh. The seed path is the frozen inventory's
existing input, which is only read.

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.s1_mature_20261018.admission prepare \
  --published-ref origin/t31-colab-safety-20261018 \
  --seed-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_BANK.json \
  --world-bank research/s1_mature_20261018/WORLD_BANK.json \
  --seed-audit research/s1_mature_20261018/SEED_AUDIT.json \
  --pricing research/s1_mature_20261018/ADMISSION_STATUS_ACCEPTANCE_RUN9.json \
  --artifact-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-run9-20261019/production \
  --manifest-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-run9-20261019/jobs \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-run9-20261019/plan.json
```

This command creates only a plan and manifests. Independent acceptance review,
fresh account evidence and full ledger admission remain dispatch requirements.
