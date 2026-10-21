# T34 Round 0-D: operator runbook (DRAFT, re-scoped; dispatch only on a governing scientific Round 0 NO-GO)

Rules: [PREREGISTRATION_ROUND0D.md](PREREGISTRATION_ROUND0D.md). Draft registration:
[REGISTRATION_ROUND0D_DRAFT.json](REGISTRATION_ROUND0D_DRAFT.json) (status DRAFT, not ratified).
Evidence for this build: [VERIFICATION_ROUND0D.md](VERIFICATION_ROUND0D.md).

## Modules

All code is new; nothing under `research/t34_round0`, `t34_pathb`, `t34_exit_b8`, or `t34_exit_b8_fit` is modified.

| Module | Role |
|---|---|
| `controller.py` | `DenseT3Controller`: the exact T3-dense rule (`ceiling.py:103-140`) on the T34 merged-rollout engine |
| `runner.py` | Paired producer. Bare arm is Round 0's `_bare_trace`; teacher arm is dense. Writes a public label-free trace plus a sealed label export |
| `bank.py` | Fresh 240-world allocation via Round 0's `_rows`, plus excluded fixtures; whole-allocation collision audit |
| `history.py` | Round 0's create-only claim/ledger/seal machinery under `refs/tags/t34-round0d/`, plus a two-lane liveness lock |
| `dispatch.py` | One explicit chunk dispatch: Round 0 settlement, 48 h deadline, mechanical rerun, ≤ 2 in flight, forecast-derived ceiling (settled breach seals INCONCLUSIVE) |
| `replay.py` | Second-L4 cross-runtime replay of Round 0's 10% sample |
| `analyze.py` | Sole decision: Round 0's `summarize` (imported) then the owner-approved relative boost floor (`t34_round0_amend/rule.py`, pinned); §3.1 descriptive report |
| `pins.py` | SHA-256 pins of every imported Round 0 / amendment / helper file; drift refuses |
| `labels.py` | The only reader of sealed labels: dense R1 fitting, published GO only |
| `a5d.py` | A5-D fidelity (8 T3 census worlds) (tie rule) and throughput (rate from process-seconds and row-steps; may run pre-verdict) |
| `sizing.py` | A5-D receipt → frozen rate-to-action table (KILL r < 5, ≤ 120 u, ≤ 24 h), Round 0 chunk formula, derived ceiling, replay sizing, dense R1 yield rule (≤ 200 u) |
| `registration.py`, `validate.py`, `make_manifest.py` | Draft and ratify, A6 validation, supervisor-schema manifests |

## Environment

```bash
export PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export B8=/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth
export PREP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0d-prep-20261021
export E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0d-study-01
```

## 1. Preparation (Mac; already done for the draft except where noted)

```bash
$PY -m research.t34_round0d.bank            # create-only; BANK_ROUND0D.json + SEED_AUDIT_ROUND0D.json
$PY -m research.t34_round0d.registration draft
nice -n 10 $PY -m pytest -q -n 2 tests/test_t34_round0d.py
# A6: excluded H500 fixture, real entrypoint, one thread. Use a fresh path each time, after the final commit.
$PY -m research.t34_round0d.bank --preflight-inputs "$PREP/a6-XX/inputs"
nice -n 10 $PY -m research.t34_round0d.runner --preflight \
  --bank "$PREP/a6-XX/inputs/bank.json" --audit "$PREP/a6-XX/inputs/audit.json" \
  --checkpoint "$B8" --output-dir "$PREP/a6-XX/output" --runtime-id t34-round0d-a6-XX \
  --start 0 --count 1 --device cpu --threads 1 --seconds 600
$PY -m research.t34_round0d.validate --preflight-dir "$PREP/a6-XX/output" --out "$PREP/a6-XX/VERIFICATION.json"
$PY -m research.t34_round0d.a5d prepare-fidelity --out "$PREP/FIDELITY_T3.json"   # reads T3 receipts only
```

## 2. After judge approval: A5-D may run BEFORE the Round 0 verdict (excluded worlds only)

A5-D touches no Round 0 bytes and no selection world, so it may run as soon as the re-scoped bytes are approved
and a source-current A6 exists. Expected waste on a Round 0 GO is ≤ 10 u. Order: final commit, then A6 (§1),
then A5-D.

```bash
$PY -m research.t34_round0d.make_manifest a5d --commit "$(git rev-parse HEAD)" \
  --fidelity "$PREP/FIDELITY_T3.json" --output-dir "$PREP/a5d-01/job" --out "$PREP/a5d-01.json"
research/colab_cli/launch.sh --state-dir "$PREP/a5d-01-state" run "$PREP/a5d-01.json"   # 300 min, 10 u cap
$PY -m research.t34_round0d.sizing --a5d-benchmark "$PREP/a5d-01/job/exports/a5d/benchmark.json" \
  --preflight-dir "$PREP/a6-XX/output" --setup-minutes 15 --out "$PREP/A5D_RECEIPT.json"
```

The sizing command prints the frozen table action:

- **HOLD:** fidelity failed, or nothing was measured.
- **KILL:** r < 5, expected > 120 u, or wall > 24 h on 2 lanes.
- **RATIFY:** activation may proceed.

## 2b. Activation: only after the governing Round 0 verdict routes here

The governing verdict is the published `amendment-boost-floor/decision.json` with `governing: true`. It routes to
Round 0-D only on a **scientific** STOP or INCONCLUSIVE. Activation needs no new judgment. Record it in
CAMPAIGN_LOG.md, then:

```bash
$PY -m research.t34_round0d.registration ratify --operator "$T34_OPERATOR" \
  --a5d "$PREP/A5D_RECEIPT.json" --preflight-record "$PREP/a6-XX/VERIFICATION.json" \
  --round0-decision /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/amendment-boost-floor/decision.json \
  --round0-ledger-repo <Round 0 authoring checkout>
git add research/t34_round0d/REGISTRATION_ROUND0D.json && git commit -m 'Activate T34 Round 0-D' \
  && git pull --rebase && git push origin HEAD:main
git worktree add --detach /Users/josenunez/Projects/ml/snake-dqn-round0d-frozen-01 <SOURCE commit>
```

`ratify` refuses in each of these cases:

- the verdict is not governing, not sealed, or not published;
- the verdict routes elsewhere (GO, fault STOP, integrity INCONCLUSIVE, KILL);
- any import pin has drifted;
- the A6 is stale;
- the table does not say RATIFY.

## 3. Dispatch, at most two lanes, from the frozen checkout

```bash
$PY -m research.t34_round0d.make_manifest chunks --ratification <authoring>/research/t34_round0d/REGISTRATION_ROUND0D.json \
  --a5d "$PREP/A5D_RECEIPT.json" --out-dir "$E/manifests" --lanes 2 --other-assignments '{}'
$PY -m research.t34_round0d.dispatch run --manifest "$E/manifests/t34-round0d-chunk-000-attempt-01-l1.json" \
  --ratification <authoring>/research/t34_round0d/REGISTRATION_ROUND0D.json --a5d "$PREP/A5D_RECEIPT.json" \
  --state-dir "$E/supervisor-chunk-000-attempt-01"
```

After every terminal seal, commit and push the ledger from the authoring checkout
(`research/t34_round0d/ledger/t34-round0d-study-01.jsonl`) before the next admission, as in Round 0.

- Every chunk must be dispatched within 48 h of `approved_at`.
- At most 2 attempts may be in flight; the dispatcher refuses a third. Each attempt holds `inflight.lock` until
  it is sealed.
- Run each dispatcher in its own tmux session under `caffeinate -dimsu`, on AC power. Check free disk space
  first: the data volume is about 96% full, and a disk-full during a seal is KILL.
- A ceiling refusal while an attempt is in flight is transient: retry once it settles. If the ceiling is
  crossed on settled charges alone, the dispatcher seals `ceiling` INCONCLUSIVE itself; publish the ledger.
- Never kill a dispatcher. An abandoned claim is KILL.
- Mechanical rerun: `--mechanical-rerun --outcomes-unseen`, only after a MECHANICAL_FAILURE seal.

## 4. Replay (second L4) and the sole decision

```bash
$PY -m research.t34_round0d.replay mac --ratification <ratification> --chunk-dirs "$E"/chunks/chunk-*/attempt-0?/job/exports/round0d \
  --state-dir "$E/supervisor-replay" --out "$E/replay/replay.json"       # use only each chunk's latest COMPLETE attempt
# Publish the ledger, then:
$PY -m research.t34_round0d.analyze --chunks <same dirs> --ratification <ratification> \
  --replay "$E/replay/replay.json" --out "$E/analysis/decision.json"
# Publish the ledger again before reporting.
```

On GO only, `research.t34_round0d.labels.admit(registration, "dense_r1_fit")` releases the 240 sealed teacher
worlds for dense R1 fitting. The R1 sizing is in the decision's ratification (`dense_r1_plan`).
