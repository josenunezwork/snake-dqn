# Overnight 2026-10-18: what ran, what waits for the owner

**Runtime:** one Colab A100-SXM4-40GB High-RAM (12 vCPU, ~5.3 units/h), owner-granted all night; queue worker
`3854efbc34ec-...` started 00:36 PDT with `IDLE_MINUTES=100000` (verified in the running process), so it does not
self-release. A leftover CPU High-RAM worker (`d946df279cf1-...`) was idle and keeps the default 10-min release.

## Ran
- `overnight-host-contracts-a100` (Astra-advised row 3, Sol-built launcher `research/q0q1_20261017/run_host_contracts.sh`,
  commit 2f966e8): **PASS** - 75/75 tests (Q1 trainer, collector oracles, qualification, Q0 audits, T22/T24 parity)
  on the A100's x86 host in 38 s, 0 skips. Receipt: Drive `t16_colab/overnight-20261018/host-contracts-a100/attempt-1/`.
  Qualifies host contracts only; not CUDA, not Q1 admission.

## Blocked (by design) until owner review
- `q0-audit-t23-mass-a100` failed in 5 s: the Q0/Q1 launcher hard-refuses while
  `research/q0q1_20261017/PREREGISTRATION.md` is `DRAFT_PENDING_OWNER_REVIEW` (the draft covers Q0 too).
  `q0-audit-b8-init-a100` cancelled for the same reason. The guard was not bypassed.

## Owner decisions needed
1. Approve (or amend) the Q0/Q1 pre-registration. After approval: re-enqueue the Q0 audits, then the Q0b
   qualification packets, then Q1 (20 L4 training jobs + 100 CPU eval shards).
2. Budget: the draft caps Q0 at 15 units assuming L4/CPU prices; running the audits on this A100 would cost
   ~30+ units. Either run Q0 on CPU/L4 runtimes or amend the cap.
3. Overnight A100 debit is allocation/idle cost (~5.3 units/h) with only the 40 s check as work.

Advice: `docs/research/advice/2026-10-18_overnight_a100.md` (Astra panel: audits + optional host check only).
