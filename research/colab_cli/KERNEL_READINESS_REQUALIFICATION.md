# Kernel creation URL/readiness correction — 2026-10-22

Status: **NOT QUALIFIED; fresh Claude final-only requalification required.**
This supersedes the execution handoff in `KERNEL_AMBIGUITY_REQUALIFICATION.md`.
The original document, failed acceptance, receipts and previous P1 packs remain
immutable. Offline review does not qualify the changed transport.

## Failed acceptance and diagnosis

Evidence root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-ambiguity-requalification-20261022/requalify-7mpzkr3k/acceptance-1792727428`.
The acceptance tested `38a12a915d357ac0412090bfb22acfe881833333` and ended
INCOMPLETE in CPU readiness; L4 was not reached. CPU `receipt.json` shows one
failed exec lasting **0.306285292 seconds**, no transfers/exports, no terminal
workload, verified stop, no cleanup error and zero remaining assignments.
`warmup-measurement.json` records **50 attempts / 300.007870583 seconds**.

The final exception retained in `last_probe_error` is
`GuardError: kernel creation does not match tracked runtime identity`.
This is a pre-POST local safety refusal, not evidence that the kernel was busy.
The installed CLI passes its runtime URL unchanged to
`KernelHttpManager.start_kernel`, which constructs
`f"{self.server_url}/api/kernels"`. A trailing slash therefore produces
`//api/kernels`, while the new guard compared only against the stripped,
single-slash construction. The old tests used a trailing-slash state URL but
manually supplied a single-slash request and missed the actual boundary.
An independent offline reproduction with the installed Jupyter manager and
StateStore confirmed refusal before any network POST on the old source.

The original URL and individual GET responses were not retained, and cleanup
removed the session row. The trailing-slash trigger is a deterministic installed
client reproduction, not a recovered original URL. The terminal guard refusal
and its incorrect retry classification are directly recorded in the evidence.

The exact Rich traceback includes the neighboring source line that raises
`tracked kernel reconnect required; kernel already exists`. The classifier
matched that source excerpt as though it were the terminal exception. After
the pre-POST refusal, warmup kept polling without another exec until reporting
the misleading busy/unknown deadline. The recorded error has no timeout marker;
the neighboring reconnect text is the observed false transient signal.

## Correction and offline evidence

- Accept only the two exact creation URL constructions derived from the saved
  runtime URL: the installed manager's verbatim append and our canonical route.
  Endpoint, notebook UUID, config/session and proxy-token checks still apply.
- Classify the terminal `GuardError:` line independently of Rich source excerpts.
  Permanent guard refusals fail immediately. A real saved-kernel reconnect signal
  remains recoverable only after independent GET proves the exact saved ID idle.
- Preserve the durable saved-ID fence and the process-local uncertain-POST fence.
  Unknown identities, foreign paths, malformed responses and unowned ambiguous
  inventories still refuse; no extra allocation, deletion, cap or retry is added.

`tests/fixtures/colab_kernel_identity_refusal.json` copies the exact CPU receipt
and warmup evidence. The regression replays its CLI failure through production
call/RPC/warmup code using a synthetic clock, verifies fail-fast behavior, and
keeps uncertainty set. URL construction and Rich reconnect recovery cover both
trailing-slash and non-trailing-slash runtimes. Before the fix, the new selection
had **5 failed, 2 passed**; the focused final suite has **81 passed in 31.32s**:

```sh
cd /Users/josenunez/Projects/ml/snake-dqn-kernelamb
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
  tests/test_colab_cli_guard.py tests/test_colab_cli_warmup.py \
  tests/test_colab_cli_dispatch_warmup.py tests/test_colab_cli_latency_policy.py \
  tests/test_ops_qualification.py \
  tests/test_colab_cli_supervisor.py::test_warmup_failure_stops_runtime_without_uploading_or_launching
```

Independent Codex-V receipt (create-only, outside the qualified Git tree):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-readiness-20261022/codex-v-review.json`.

## Claude requalification commands

Use the final published branch SHA, clean and unchanged between plan and execute.
Use a **new** plan; the previous failed plan is consumed. Preserve `final-only`,
all eleven required gates and the existing **4.0-unit total / 0.8 reserve**.

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-kernelamb
cd "$REPO"
git fetch origin
git status --short
git rev-parse HEAD origin/supervisor-kernel-ambiguity-20261022
# Claude prepares a fresh reviewed map from current account inventory/rates.
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" \
  --state-parent /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-readiness-requalification-20261022
# Inspect the generated plan, manifests, source inventory and budget.
PLAN=/absolute/path/printed/by/plan/ops-review.json
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

Require CPU and L4 PASS, exact source identity, all eleven observed gates,
verified exports and backend disappearance. Review saved kernel identity and
the first successful Contents upload. The registry updates only on PASS.
No live acceptance or production dispatch was run by Codex.

## P1 retry pin

The additive P1 pin index is
`research/bl_ppo/P1_E8_04_KERNEL_READINESS_RETRY_PIN_05.json`.
It binds a fresh held sibling at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-kernel-readiness-repin-20261022`.
Its manifests, embedded control source commits and input archive bind the fixed
published transport commit. Its top-level dispatch fuses remain false. This is
**BLOCKED_PENDING_REQUALIFICATION**, not renewed qualification or canary admission.
Keep the old pack04 and its failure immutable. Before any science retry, Claude
must bind the new passing acceptance and independent pack review, reconcile the
latest debit and live inventory, and run a fresh source-bound family canary.
Serving/release and main remain outside this branch-only handoff.
