# R30 Mac qualification and operator handoff — 2026-10-24

Outcome: **MAC_PREFLIGHT_PASS / CONSUMER_CANARY_PREPARED**. No dispatch,
allocation, real-label fit, validation look, gameplay or release occurred.
Scientific R30-1 remains HOLD pending the separately required corpus and native
canary admission; R30-2 retains every F4 closure and diagnosis prerequisite.

The binding R30 ruling is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/JUDGMENT.md`,
SHA256 `f69c44863a8331028ade0dff20ee69eb9dc5959a92e8480a7329cb39d2b58d5c`.
The interim concurrency ruling SHA256 is
`ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd`.

## Exact source and broker

- Integration worktree: `/Users/josenunez/Projects/ml/snake-dqn-r30`, branch
  `t34-r30-20261023`. Changes and dispatch-leaf ancestry are published to main.
- **Use the clean dispatch worktree**
  `/Users/josenunez/Projects/ml/snake-dqn-r30-l4-20261024`, branch
  `t34-r30-l4-20261024`, pinned at
  `72b6fcf0583c216f3185b9c5c31ae9abcb861a3f`.
- Its `research/colab_cli` tree is
  `d0d92bdb572b770284a1fea6152417ffa5f06bea`, exactly matching the operational
  qualification registry. Main has newer transport bytes that do not match this registry;
  this task does not qualify them. Ancestry-only merges retain the dispatch pin
  on main without rolling back main's transport. Do not fast-forward the dispatch
  worktree to main or replace its pin with a later documentation commit.

## What changed and what ran

`research/bl_round30/factory_check.py` adds an R30-only ordinary-production route.
It preserves the live lease and checks exact driver/start/control identities,
the approved D1024 operational pin, its own immutable fresh W6 first-hour baseline,
a current successful process observation, nice10 numerical workers and absence of
timing/terminal signals. It retains the common serial lock through owned cleanup,
120-second deadline, CPU/thread limits and nice>=15. The running factory and the
shared launcher were not patched.

`local_check.py` binds the executed dependency families and explicit test nodes.
The original float64 `sum == 1` fixture now permits absolute roundoff `1e-14`,
with zero relative tolerance. The independent objective/gradient comparison and
all scientific thresholds are unchanged.

`job.py` now reloads the actual immutable synthetic fused endpoint and performs
a real target-device six-action fp32 argmax with advisory-mask/normal fallback.
The qualified CUDA observer can attribute that decision to the R30 package.
CPU tests prove the reload/call/mask/tie path; they do not prove native CUDA.
Both canary types now declare the lawful 20-minute `short-canary` exemption,
because canary units cannot enter scientific resume ledgers. Scientific mains
retain resume-by-default, including the exact verified-unit skips and maximum
of two mechanical completion attempts.

Preserved attempts under the artifact root:

1. `sol-verification/nice15-check-20261024-01`: refused before pytest because
   the previously observed factory PID had exited. Its lease had disappeared.
2. `qualified-check-20261024-01`: 62 passed / 1 failed, 3.28 s; exact float64
   fixture equality failed, while independent objective/gradient equivalence passed.
3. `qualified-check-20261024-02`: 63 passed, 2.84 s, source `ad163339`. Its prepared
   canaries were superseded after independent review found the missing CUDA
   action-selection call. Sibling `STATUS_20261024.json` files mark them
   **SUPERSEDED_DO_NOT_DISPATCH**; original artifacts remain unchanged.
4. `qualified-check-20261024-03`: 65 passed, 2.87 s, source `ce73d168`.
   Its canary declarations were superseded by the explicit short-canary exemption.
5. **`qualified-check-20261024-04`: 66 passed, 2.74 s, final source `72b6fcf0`.**
   The existing quiet-window launcher held the shared serial lock and Mac lease.
   Caller and numerical children ran at nice15; the CPU-only plugin enforced
   all six numerical pool limits and Torch intra/inter-op threads at one.
   No xdist, process pool or scientific job was used. The new ordinary-factory
   route was tested with synthetic fixtures, not exercised against a live factory.

The exact final preflight command, run from the dispatch worktree, was:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_round30.local_check \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/sol-verification/qualified-check-20261024-04
```

Black checked 19 package/test files with one worker; isort, flake8 (one job),
Bash syntax and `git diff --check` passed. Supervisor `validate` passed for both
fresh canaries, and the exact R30-1 wrapper below passed with `--dry-run`:
`PASS_DRY_RUN ... allocations=0 live_accounting=NOT_CHECKED`.
The sparse runtime import closure was reviewed. A fresh dependency-only venv
install/build and native CUDA canary were not run.

## Immutable evidence

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30`.
Every JSON below has SHA and completion sidecars.

| Evidence | Relative path | SHA256 |
|---|---|---|
| Final Mac PASS | `sol-verification/qualified-check-20261024-04/preflight.json` | `477e06a0605ae46fd7352883dfdf8eb16bbee225b3c4b583d311e407ddc75db2` |
| R30-1 canary | `qualified-72b6fcf0-r301/job.json` | `f6843a66806e63d180f0317e65c09fdda418ea9bb8c32814f05d24ab72ee664b` |
| R30-2 synthetic template | `qualified-72b6fcf0-r302/job.json` | `8afff36d9574a0eeb1cfd8e8900da7b959400e594b33161d9d483b439d1f3ce1` |

`sol-verification/QUALIFICATION_72b6fcf0_20261024.json` records package validation,
source identity, prior dispositions and the remaining scientific HOLD. The final
independent Codex-V light report is transcribed in the sibling repository JSON
`round30_codex_v_light_20261024.json`, with artifact/hash references.

## Exact R30-1 L4 operator command — prepared, not executed

This command runs the **synthetic consumer rehearsal**, not the scientific
optimizer diagnosis. Revalidate with `--dry-run` if desired; remove only that
flag to perform the operator-owned dispatch. The wrapper performs fresh locked
accounting, preserves existing assignment priority and enforces the account cap.

```bash
/Users/josenunez/Projects/ml/snake-dqn-r30-l4-20261024/tools/ops/dispatch.sh \
  r30-1-rehearsal-72b6fcf0-01 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-72b6fcf0-r301/job.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-r30-l4-20261024 \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-r30-l4-20261024 \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/broker-r301-rehearsal-72b6fcf0-01
```

| Scope | Effective cap, first limit wins |
|---|---|
| This synthetic L4 canary | **20 min / 2/3 unit**, rate **1.54 u/h**, reserve **1/9 unit**, estimate **5 min**, max export **128 MiB**, lease **180 s** |
| Entire R30-1, including setup/canaries/failed and completion attempts | **144 min / 4.8 units** (base 72 min / 2.4 units) |
| Conditional R30-2 | **288 min / 9.6 units** |
| Combined cloud series | **432 min / 14.4 units**, including reserve **72 min / 2.4 units** |
| Mac work | R30-0 **.25 aggregate core-hour**; total light work **.75 core-hour** |

No automatic completion dispatch. At most two outcome-blind completion attempts
remain subject to mechanical-failure, verified-stop and missing-unit evidence;
all prior liabilities remain in the same stage/combined envelope. Rates, estimates,
N and retries are not doubled.

## Scientific main remains HOLD

No eligible reusable corpus, independent metadata approval or unencumbered
reservation was supplied/admitted in this task. No protected F4, retention,
strict, probe, natural-archive or gameplay payload was opened. An authorized
training partition must first pass `prepare`, then the exact-input same-pin
native L4 canary must pass with verified exports, terminal receipt and runtime
release before a scientific main manifest can be generated/admitted. Synthetic
consumer evidence cannot replace that exact-input proof.

The existing README's admitted-data preparation sequence still applies, using
this dispatch worktree, the exact `72b6fcf0` pin and final preflight receipt.
There is intentionally no executable scientific-main command for an absent corpus.
Its manifest must subtract all measured earlier charges and the reserved native
canary liability. With no earlier spend and one native canary the maximum main
ceiling is 124 min / 62/15 units. If this separate synthetic rehearsal and that
native canary each consumed their full envelopes, it would be at most
104 min / 52/15 units. Never budget both canaries outside R30-1.

R30-2's fit/look still waits for F4 terminal disposition, its closure fence,
resolved optimizer diagnosis, complete rescue support and admitted natural
ordinary archive. No serving change or release is authorized.
