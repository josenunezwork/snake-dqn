# TRAIN-01 c2: released accounting-guard stop, 2026-10-29

Implementation decision: the exact native host accounting-rate guard is an
outcome-blind mechanical stop under TRAIN01.md's original-plus-two-manual-
completion rule. A prepared c2 resumes the latest verified iteration 0336 and
ends at the unchanged iteration 0384. This is a tooling admission extension over
accepted readiness-6 (`da85a426`), not a claim that that older scanner admitted
this receipt. Independent Argon review and packet qualification are retained in
the external task directory below. No training or provider operation is authorized
by this note; the requested live command remains unrun.

## What actually stopped r6

Retained native receipt: `sandbox-a-train01-20261028-15-c1-i0064-r6`, generation
`3b090bbe2b1a4ca493edb4f18ce75fcc`, runtime `dccf7e25`, elapsed
75,429.385666291 s, `INCOMPLETE`, `failure_class: ACCOUNTING`, `terminal: null`,
`stop_verified: true`, `cleanup_error: null`. The exact error is:

```
AccountingError('provider usage rate drift is not explained by accounted active assignments')
```

`research/colab_cli/accounting.py:AccountTracker.check` raises this error when
the instantaneous provider usage rate cannot be reconciled to the observed own
rate plus accounted other assignments. This check is independent of trainer
metrics. The supervisor then follows its normal stop/verify/receipt path.
It was not a priced-cap stop: the last successful cross-check charged
167.10211134351997 units against a 369-unit guard. The frozen wall cap was also
not reached.

The 344 `UNEXPLAINED_ACCOUNT_DEBIT` alerts are warning-only balance cross-checks,
not that fatal rate check. Every alert has a negative discrepancy. The first,
at log line 140 / elapsed 1,714.793238125043 s, reports 2.05 debited units versus
2.524556711684091 expected. The last reports 148.33 versus 150.25601297946747
expected. These warnings do not establish excess billing or causation by the
operator's much later failed allocations.

The last persisted usage sample, poll 2068 at elapsed 75,195.95010458399 s,
reports rate 5.3 u/h and one assignment. Earlier successful samples include
properly accounted concurrent assignments; other integrated charge totals
39.5508642143855 units and had stopped increasing by poll 1592. The account debit
955.56 - 806.79 = 148.77 units covers the account's whole interval, including
those other assignments. It is not r6-only spend. The zero-rate/zero-assignment
`usage_after` is post-cleanup evidence.

The triggering usage, sessions and active-record values are not retained:
`Broker.execute` invokes `accounting.check` **before** writing usage/spend files,
and `last_crosscheck` remains the preceding successful check when it raises.
CLI call timings record that the reads succeeded, not their returned values.
Consequently, these artifacts cannot attribute the final mismatch to failed
allocation requests, a provider reporting artifact, a genuine rate change or
a broker membership transition. No causal diagnosis beyond the exact guard is
claimed. The supplied context about GET-only reconciliation is not a provider
billing statement. A billing/root-cause investigation needs independent retained
provider request/usage responses or provider-side records; reconstructed values
must not be inserted into r6 evidence.

The last retained trainer tail reaches iteration 350, including completed 349.
The latest completed exported full-state snapshot is still 0336. Unexported work
337 onward may be replayed from 0336 under the registered snapshot recovery
rule; no previously verified exported unit is recomputed into the result set.

## Narrow admission change

Only the exact native error above, with `failure_class: ACCOUNTING` and
`terminal: null`, is newly admitted. The ordinary receipt still must match
job/commit/manifest identities, be `INCOMPLETE`, verify stop and clean cleanup,
and pass every retained output/export/poll and full supervisor-log fault scan.
The error is not accepted as a trainer diagnostic, an alternative exception
wrapper, a different receipt class or an appended/compound error. Assignment
disappearance, runtime self-release without a terminal, arbitrary accounting
errors, unknown exceptions and scientific/stall faults remain refused. The
receipt-less host-loss path is not used or broadened. The qualified transport
and all accounting guards remain unchanged.

`capture_supervisor_goldens.py --scenario rate-drift` captures the actual
unchanged qualified Broker against the subprocess fake: after native calibration,
only the fake provider rate changes; its sessions remain fixed. Native
`AccountTracker.check` produces the receipt; tests do not manufacture it.
Healthy/fault twins verify exact-envelope admission and scientific-fault
precedence in receipt, progress, poll, snapshot header and supervisor log.

## Hardware and frozen science

Use A100 as before. The frozen sandbox preflight explicitly requires
`requirements.gpu == A100`, and the runtime fixes `TORCH_CUDA_ARCH_LIST=8.0`.
The accepted c2 builder emits `standing-a100-development-sandbox-admission/v1`
and the A100 priced envelope. H100-first preference support on main does not
override these TRAIN-01 constraints or qualify this runtime on H100. Borrowing
that support would change the sealed admission/runtime surface; the reported
fallback-deadline bug supplies no reason to do so here. No dispatch changes from
the concurrent GPU-preference repair are imported.

The control recipe, actor rows, seed offset/allocation, snapshot/proxy cadence,
monitor/nomination banks, frozen inputs, endpoint, retry maximum, transport,
training kernels and serving state are unchanged. c2 is completion index 2 of 2;
there is no c3 or automatic completion retry. Envelope calculations use the
accepted builder's measured A100 rates and pre-data 2x guards.

## Retained evidence

Task directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-c2/`.
`R6_DIAGNOSIS.json` records the exact arithmetic, latest unit identities and full
scan counts. `SOURCE_INVENTORY.json` binds the entire stopped r6 root and log by
path/size/SHA-256. Replays use a byte-verified private copy; original evidence is
read-only. Summary, test commands/logs, Argon review and prepared packet result
are retained there. Inspect `SUMMARY.md` for qualification and exact local pins;
none of this declares a scientific outcome, promotion or serving release.
