# S1 public-input delivery implementation and operator handoff

Prepared locally only. No live Colab commands, dispatch, release publication,
settlement of the live ledger, or tmux operations were performed. The seven immutable inputs total
62,552,352 bytes; their digests remain in `INPUTS.json`. Claude publishes them at
`https://github.com/josenunezwork/snake-dqn/releases/download/s1-inputs-20261019/`.
The seed asset is named `SEED_BANK.json` and lands on the VM as `seed-bank.json`.
All other asset names equal their VM destinations.

| Input | Runtime use |
| --- | --- |
| `B8_s3_final.pth` | Initial weights, frozen B8 opponent and runtime parity comparator |
| `demo.tar.gz` | V8 anchor/demo data consumed by the real trainer |
| `league.tar.gz` | Frozen historical opponent checkpoint inventory |
| `mac-parity-reference.json` | Required pre-training `q1_parity --verify-reference` comparison |
| `parity-bank.json` | Worlds paired with that reference |
| `q0b-l4.json` | Current L4 hardware/runtime qualification gate |
| `SEED_BANK.json` | Registered five-seed identity and namespace checks |

`train.py` reads every input above. Only the small current
`cli-acceptance-result.json` remains a Contents upload. A plan retains separate
Mac-local input bindings for final evaluation; remote manifest URL rows contain
no Mac path. The Mac downloads and hashes every release asset before reservation
and again inside supervisor admission. The runtime independently downloads to
temporary files, verifies size and SHA-256, and atomically installs them before
training. A missing/mismatched asset refuses. No token is needed for public assets.

The staging forecast includes the sparse source fallback, helper/metadata slack,
and acceptance evidence. It uses half the observed dispatch-02 Contents rate
(6,500 bytes/s), a conservative release rate (1 MiB/s with a twofold margin),
and per-request/setup slack. Mac verification lowers the release forecast rate
when the measured rate is slower. Contents and public fetch each retain their
separate 300-second cap; the sum is disclosed, and runnable admission also checks
training + staging + final export against the unchanged job timeout. This is a
forecast, not a VM-throughput measurement. Runtime deadlines remain enforced.

The staged-failure API charges reconciled spend and elapsed allocation time,
preserves the failed receipt and manifest, and restores the identical original
job id to planned. It requires owner-reviewed proof of input-staging failure,
verified stop, no trainer start, no terminal training outcome, and no exports.
It does not consume `retry-n3`. Source changes invalidate run-9 acceptance:
historical evidence remains immutable, and fresh final-only qualification is
charged separately without resetting the original study start or caps.

The offline candidate is deliberately `PENDING_FINAL_ONLY_ACCEPTANCE`; the
dispatcher refuses it. Fresh pricing and normal `admission prepare` create a
separate runnable plan after acceptance. The existing ledger is continued with
`--continue-staging --previous-plan ...`; science, seeds, input identities,
liabilities, and Mac evaluation remain frozen.

## Operator sequence (Claude only)

These commands are a handoff, not authorization or a record of execution. Run
from a shell with the published clean branch checked out. Keep each new output
directory create-only. Do not overwrite a previous plan, review, or receipt.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
S1_REPO=/Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_BASE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018
S1_NEXT="$S1_BASE/admission-public-release-20261019"
S1_LEDGER="$S1_BASE/admission-run9-20261019/dispatch/ledger-01.json"
```

First prepare/review the staging settlement using the settlement CLI described
below, then apply it explicitly. This preserves the original failure evidence.
The delivered local review is `$S1_NEXT/staging-failure-review.json`. Read-only
receipt reconciliation found 0.19000000000005457 units and 496.2047262920096
seconds (0.13783464619222488 allocation hours); no ledger mutation was performed.
The real settlement entrypoint also passed against an isolated temporary copy of
the actual ledger, leaving the live ledger unchanged. `settle-03.cmd` beside the
delivered review contains the following ready-to-run owner command.

```bash
"$S1_PY" -m research.s1_mature_20261018.settle_staging_failure \
  --ledger "$S1_LEDGER" --job-id s1-mature-n3-223213833 \
  --receipt "$S1_BASE/admission-source-repair-20261019/production/training/s1-mature-n3-223213833/receipt.json" \
  --review "$S1_NEXT/staging-failure-review.json"
```

To regenerate an offline candidate at a later clean published commit, use a new
destination rather than overwriting the delivered candidate:

```bash
"$S1_PY" -m research.s1_mature_20261018.prepare_release \
  --repo "$S1_REPO" --published-ref origin/t31-colab-safety-20261018 \
  --previous-plan "$S1_BASE/admission-source-repair-20261019/plan.json" \
  --artifact-root "$S1_NEXT/candidate/production" \
  --manifest-dir "$S1_NEXT/candidate/jobs" --out "$S1_NEXT/candidate/plan.json"
```

After Claude publishes the release assets, check their real bytes without any
account access:

```bash
"$S1_PY" -m research.s1_mature_20261018.admission preflight-inputs \
  --repo "$S1_REPO" --plan "$S1_NEXT/candidate/plan.json" \
  --out "$S1_NEXT/release-input-preflight.json"
```

Re-run final-only acceptance against the new commit, through the existing
reviewed supervisor workflow (this is live compute and is left to Claude):

```bash
"$S1_PY" -m research.colab_cli.acceptance --profile final-only \
  --repo "$S1_REPO" --state-dir "$S1_NEXT/acceptance" \
  --colab /Users/josenunez/.local/bin/colab \
  --execute-after-review --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

The command prints its new acceptance directory. Set `S1_ACCEPTANCE` to that
directory's absolute `acceptance-result.json` path. Claude must independently
review its CPU/L4 receipts, final download, and watchdog disappearance, then
write `$S1_NEXT/acceptance-review.json` using `s1-cli-acceptance-review/v1`.
Bind all current transport sources (including `inputs.py`), preserve the original
ledger `created_at` as `study_started_at`, and charge only this additional
acceptance in `acceptance_charge`. Do not edit run-9 review or source hashes.

```bash
"$S1_PY" -m research.s1_mature_20261018.price \
  --eval-preflight "$S1_REPO/research/s1_mature_20261018/EVAL_PREFLIGHT_FINAL.json" \
  --training-preflight "$S1_REPO/research/s1_mature_20261018/TRAIN_PREFLIGHT_FINAL_ONLY.json" \
  --acceptance-result "$S1_ACCEPTANCE" --out "$S1_NEXT/pricing.json"
"$S1_PY" -m research.s1_mature_20261018.admission prepare \
  --repo "$S1_REPO" --published-ref origin/t31-colab-safety-20261018 \
  --input-delivery public-release \
  --seed-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_BANK.json \
  --world-bank "$S1_REPO/research/s1_mature_20261018/WORLD_BANK.json" \
  --seed-audit "$S1_REPO/research/s1_mature_20261018/SEED_AUDIT.json" \
  --pricing "$S1_NEXT/pricing.json" --artifact-root "$S1_NEXT/production" \
  --manifest-dir "$S1_NEXT/jobs" --out "$S1_NEXT/plan.json"
"$S1_REPO/research/s1_mature_20261018/launch.sh" \
  --continue-staging \
  --previous-plan "$S1_BASE/admission-source-repair-20261019/plan.json" \
  --plan "$S1_NEXT/plan.json" --ledger "$S1_LEDGER" \
  --acceptance-review "$S1_NEXT/acceptance-review.json" \
  --snapshots "$S1_NEXT/dispatch/snapshots" \
  --state-root "$S1_NEXT/dispatch/supervisor" \
  --artifact-root "$S1_NEXT/production" \
  --supervisor "$S1_REPO/research/colab_cli/launch.sh" \
  --colab /Users/josenunez/.local/bin/colab \
  --account-config "$S1_BASE/admission-run9-20261019/dispatch/account-01/sessions.json"
```

The continuation verifies the fresh review/pricing/manifest acceptance bindings,
all historical settlement evidence, and the complete remaining budget before
the first new reservation. Its first job is the original
`s1-mature-n3-223213833`, with a fresh output destination. No retry manifest or
replacement seed is created. If a gate refuses, retain the evidence and stop.

## Local verification

- `SNAKE_DQN_DEVICE=cpu venv/bin/python -m pytest -q -n 4 tests/test_colab_cli*.py tests/test_s1_*.py`:
  **406 passed in 546.12 seconds**. An earlier redundant serial run was stopped
  in favor of this completed final run.
- `SNAKE_DQN_DEVICE=cpu venv/bin/python -m pytest -q tests/test_colab_cli_public_inputs.py`:
  **4 passed**, including the additional hard blocked-read deadline regression.
  Together the completed runs cover all 407 current tests in the requested files.
- Focused broker release cases: **3 passed, 53 deselected**; S1 delivery/continuation
  and real child validation: **11 passed**; ledger tests: **39 passed**.
- Black (`--check --line-length 100`), isort (`--check-only --profile black`),
  and flake8 passed for all 15 changed Python files. Bash syntax passed for the
  launcher and delivered settlement command.
- Real settlement entrypoint against an isolated copy of the real ledger:
  exit 0, same job restored to planned, `retry-n3` unused, original ledger unchanged.
- CLI `--help` checks passed for settlement, candidate preparation, input preflight,
  and dispatcher continuation. Independent review reported no remaining findings.

Live asset accessibility, final-only acceptance, settlement of the live ledger,
and S1 dispatch were deliberately left to Claude's operator sequence above.
