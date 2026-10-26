# QUALIFIED_TRANSPORT_3b32d755

2026-10-26 · Astra independent transport reviewer · saved final-only acceptance review.

**QUALIFIED. No blocking defects found.** Independently inspected every required final-only observation, synthetic inputs and upload records, transfer/staging measurements, final export, and CPU/L4 release evidence. This qualifies the reviewed transport only; science package acceptance, dispatch, fit, outcome access, serving and release remain separate.

## Bound evidence

Acceptance directory: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/requal-20261026-transport/requalify-ifjq3sq3/acceptance-1793011701`. All relative evidence paths below resolve there.

- `acceptance-result.json`: PASS / qualified true / return_code 0 / final-only; independently computed SHA-256 `8356217255f0e4e8f40e27c2fbfcd4785b991d9e594ffc4eed83419063d8847f`.
- Commit `3b32d755ee3e22ed5ebf71b5f64af543b40c5ccb`; transport tree `6e8c1982b5c9ed115bfebd3145ad0720b15522fa`; whole repository tree `8c5914d5dacdec632839b78137b2cef01e3ce9bb`. Clean detached checkout. All 24 executable source hashes match both working files and pinned Git blobs, and result/final-plan inventories agree.
- `ops-review.json` SHA-256 `636b04ca84e47ec50d3b5aeccf2c42f299b9f091e63054122c12ae337e483421`, matching the accepted plan and consumed execution marker. Embedded reviewed plan equals `acceptance-plan.json`. Both frozen manifest hashes match the review.
- CPU receipt SHA-256 `aa8c632b93741ed346e96a58ee7f7cdce79c4ef61c6a6e90aca5eaf846387c17`; L4 receipt `2b45744b226486a9bbde507764f85f19017e2bbd8daf5943f61f183bba65a51b`. Both match final-plan bindings.
- Prior authority read: ACCEPT_REQUAL_PLAN_636b04ca.md and ACCEPT_TRANSPORT_REPAIR_7f5357a7.md. Execution log `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/requal-20261026-transport/execute.log` corroborates CPU terminal completion and emitted result path.

## Final observations

| Required observation | Independently inspected evidence and disposition |
|---|---|
| Account assignments and spend | Both admission baselines list no other assignments or unaccounted endpoints. Provider calibration CPU 0.08 u/h, L4 1.54 u/h, within priced 0.5/2 rates; each has three samples. |
| Fresh OAuth refresh | CPU observer records refreshed in memory and subsequent usage; no interactive auth invoked by this review. |
| Cold-kernel recovery | Final L4 observer contains both CPU/L4 arms: original and recovered IDs identical, tracked metadata unchanged, affirmative idle helper, first upload completed in one attempt. CPU-only observer's intermediate INCOMPLETE is the expected partial cumulative snapshot. |
| Silence | CPU observer records 66.314847584 s silence, explicit 120 s RPC allowance, and successful after-silence response. |
| URL without second allocation | CPU sessions before/after show the same sole endpoint and session. |
| Client loss and reconnect | Owned client PID 32587 terminated; request c684dd87beeb45ed9310abdd5bda6bce completed; same remote execution identity 7e4397be072ceba9fca0525bf3f315be, PID 1107. |
| Child failure and corrupt export | Intentional child exit 23; corrupt export rejected; failed corrupt download appears in CPU transfer ledger. |
| L4 watchdog | Renewal suppressed; checkpoint observed during grace with recorded SHA 50931f64fafcc1ff4e62580b0cb8037484302cd13b8dc4ee099e9f29b2c5f9de. Exact endpoint gpu-l4-s-ass1a1-35itnf4a1agmk absent from sessions_after {} 26.399123750 s after earliest lease expiry, below 300 s; expiry uncertainty 2.317676084 s, lease 180 s. |
| Stop/release | Both receipts stop_verified true, cleanup_error null, final assignments/rate zero. Each saved call inventory has exactly one allocation and one stop. CPU receipt PASS. L4 intentional INCOMPLETE/ACCOUNTING follows watchdog disappearance and is explicitly accepted by acceptance.py:968-1001 after clean-release verification; it is not an ignored cleanup failure. |
| Final export | L4 hardware NVIDIA L4 / capability 8.9. Exact 67,108,864-byte final-artifact.bin; local SHA independently recomputed as 0810047ca7af5fa38ccaabe665979f3b689cbe3963dd460bf94f51a0b9dc4405, matching download ledger, observer, result and local sidecar; completion marker exists. End-to-end Mac verification 30.264912875 s; lower bound 2,217,381 bytes/s. |
| Synthetic inventory and staging | Inventory SHA b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea; every local synthetic size/hash independently checked, files contain only zeros, and both CPU/L4 verified-upload rows match. No transfer exception. See measurements below. |

Final required gate IDs (12), all OBSERVED: account-assignment-spend, fresh-oauth-refresh, cold-kernel-metadata-recovery, silence-timeout, url-no-second-allocation, client-loss-reconnect, failure-and-corrupt-export, mac-loss-watchdog-unassign, backend-stop-disappearance, rate-calibration, final-artifact-download, d1024-synthetic-full-inventory-staging. Full-state busy snapshot, exact resume and checkpoint overhead are NOT_REQUIRED for this final-only profile.

The watchdog source (`acceptance_faults.py:827-954`) checks exact endpoint absence before recording success; broker fallback runs later. Saved CLI timestamps independently show final renewal exec approximately 711224.2027–711226.5197, disappearance polling ended 711430.601377, broker stop began 711431.618027. Thus disappearance precedes fallback by about 1.017 s. No subsequent lease renewal occurs. These are saved provider observations, not a fresh reviewer account query.

| Synthetic upload | Bytes | SHA-256 |
|---|---:|---|
| factory-verified-ledger.json | 2,405,362 | 26a2e61981b03569a6b3702db9ea9f256fd2565f97f9f436284c016d8c2bd714 |
| train-1024.npz | 45,397,643 | c82bb0904a2ea6879dce0cb4a3cc23531ddac7d8b3bd1d66c800cdab168fe670 |
| metadata.tar | 2,211,840 | ee49e540faa0421573230f39a8dbf7b4139bc3950621537417e0ddc38a5352b1 |

Public B8 role is separately manifest-bound: 6,562,005 bytes, SHA `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, reviewed release URL; both public-preflight files record the same size/hash. No B8 payload was opened.

| Persisted staging clock | CPU seconds | L4 seconds | Bar |
|---|---:|---:|---|
| Verified transfer | 114.575185667 | 113.512113999 | <=150 |
| Source delivery | 17.444708750 | 17.322117500 | Included in total |
| Setup | 1.731969708 | 2.299123209 | Included in total |
| Total staging | 133.752867083 | 133.134602875 | <=300 |

Measurements come from each outputs/arm/upload-measurement.json, receipt transfer ledgers, input-hashes.json, source-delivery.json and public-input-preflight.json; clocks retain the reviewed semantics. L4 measurement SHA `0d58e7b52ec69e8296ae28768f88c3beb9555ecb157c7c5d8832607113576bf5` matches final evidence. Public HTTPS fetch retains its separate runtime bound.

## Publication command and consumer checks

The existing wrapper automatically publishes after successful execution (`tools/ops/requalify.py:128-144`); it has no publish-only subcommand. The exact publishing invocation for this consumed plan was:

```sh
OPS_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops \
  /Users/josenunez/Projects/ml/snake-dqn-requal-20261026/tools/ops/requalify.sh execute \
  --plan /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/requal-20261026-transport/requalify-ifjq3sq3/acceptance-1793011701/ops-review.json \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

**Already published; do not rerun.** The saved registry `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json` already binds this acceptance ID, exact result SHA, commit and transport tree. Execution is create-only and refuses consumed outputs. The reviewer did not publish or change that registry.

Both existing consumer checks were independently run read-only from the selected checkout and exited 0:

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-requal-20261026
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
REG=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 "$PY" "$REPO/tools/ops/qualification.py" check --repo "$REPO" --record "$REG"
PYTHONDONTWRITEBYTECODE=1 "$PY" -m research.colab_cli.acceptance --repo "$REPO" \
  --verify-qualified-record "$REG" \
  --required-inventory-sha256 b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea \
  --reject-tree c148431277f4e33c3f589251df892ea158407e25
```

The first returned the reviewed tree. The second returned this exact receipt SHA, commit, tree and inventory, with transfer_exception null. No F4 exception authority is substituted here.

## Review boundaries

Root independently checked saved evidence, hashes, source and consumer checks; a separate read-only reviewer independently checked all fault arms, stop receipts and watchdog ordering and found no blocking defect. No live tests, provider/Colab calls, tmux, dispatch, repository edits or commits occurred. No label/outcome payloads, D4096 outputs or t34-round0-study-01/ were opened. No runtime checkpoint pickle was deserialized. Only this create-only ruling and its .sha256 sidecar were written under the authorized review root. This is transport qualification, with all science/canary/package and owner release rules retained.
