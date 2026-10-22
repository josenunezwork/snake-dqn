# P1 source-latebody-03 family canary — validated, not dispatched

Status: **VALIDATED_NOT_DISPATCHED; accounting qualification HOLD remains.**
No Colab command, assignment or spend occurred. This sibling preserves the
original source03 manifest, receipts, control, bundle and DISPATCH bytes.

Executable pin, published to main:
`8fd1e5b43f526c89f8ecbc5d3e9a75bfbc1331af`.
Implementation: `674063c4ae809e8292604f61147392f2cfd0e746`.
Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-03-canary-01`.
The exact conditional operator commands are in that root's **DISPATCH.txt**.

## Accounting finding

The supplied `colab-cli-process-gates-acceptance/acceptance-1792660644/acceptance-result.json`
is a genuine final-only PASS, qualified=true, SHA256
`650f7bf4639f67cbfbc3395aa67d6e8afa71e4b7580a6a6a8154f3a012c6a00c`.
It does **not** discharge the source03 accounting item. Its source commit is
`e9ce99f253c301e5ddb28b4b6805c8974a7e79bd`; 21 of 22 executable transport files
match current main. The exception is `research/colab_cli/accounting.py`:

| Inventory | SHA256 |
|---|---|
| Accepted | `60ccfec58138adf9400c93ca1b8837cccd6da5b4fb2194a8ec4874f2ad33caeb` |
| Current, zero-tolerance option | `0137488c4c7e1e3f8272782e3799741503e6c871e816d9d05950390b177de784` |

The actual local `admission.require_final_only` raises:
`final-only acceptance is not bound to the current transport source`.
The CPU/L4 release receipts and 64 MiB final artifact independently rehash to
the supplied acceptance. That proves the older-source PASS, not qualification
of the changed accounting bytes. No transport code was changed for this task.
P1's generic manifest validator does not enforce this global qualification
check; use the additional `p1_family_canary preflight` chained in DISPATCH.

## Real transport and excluded workload

`p1_family_canary prepare` creates a new source template and canary. They share
the same source pin, source_paths and exact input hashes/destinations. The
original control is rebound only in new exported bytes. Its full 12-member
evidence inventory is retained, including the original review and debit.
The supplemental independent review covers the additive runtime/control-pin
change; it cannot supply funding or a scientific result.

| Uploaded input | Actual bytes | SHA256 |
|---|---:|---|
| public-inputs.json | 2,409 | `736f7b4e7730f760b8b81db00772d14abcf84c74b261a8dc7630732fdd383a10` |
| p1-inputs.tar.gz | 8,819 | `52a0ebcf6631e91f4b7d3cb17014ea45f4876f8897ec4aab08415e7e59bd1ed0` |

This is two real uploads, not padding or an invented byte/count forecast.
Public checkpoint HTTPS materialization remains the real P1 shell operation.
Upload readiness exercises the pinned sparse source builder: normal staging
forecast 67.6515 s, fallback upload plus bounded fetch 247.1724 s. The live
family gate still requires an actually observed staging time <=150 s.

The exact shell/Python CLI remains `p1_colab.sh` -> `p1_runtime y3-source`.
After unchanged production control and released-B8 checks, the explicitly
instrumented, fixed tiny operational configuration uses two worlds, one
process, 16 steps and a 32-transition boundary. It calls the real F policy,
collector, native-v8 roster, critic-only learner update, full snapshot capture
and restore APIs. Scientific P1 dimensions and gates remain unchanged.

This transport-only component loop **does not qualify** production E8,
critic readiness, the full P1 pause/durability state machine, Y3 equivalence,
or scientific outcomes. The receipt says `scientific_result=false`,
`critic_qualified=false` and `y3_source_admitted=false`. It is not a tiny
scientific PASS and does not fabricate held-out evidence.

Required producer exports are `receipt.json`, `canary/pause.json` and
`canary/pause-snapshot.pt`, each with actual `.sha256` and `.complete` sidecars.
The real runtime scanner discovers the payloads. The source template requires
its root `receipt.json`; publishing that receipt still requires the original
Y3 executor's complete snapshots and action tape.

## Price and admission sequence

Canary: L4 only, min_vcpu=12, estimate 15 min, whole-session timeout 30 min,
1.00 u cap including 0.20 u reserve, rate <=1.54 u/h with zero tolerance,
180 s lease. Time liability 0.77 u + reserve 0.20 u = **0.97 u <=1.00 u**.
Estimate includes setup, download, export and cleanup; it is not the CPU-only
component time. First wall/unit cap wins, no automatic retry.

The original 8 u Y3 invoice is fully reserved: charged 1.90 + source 4.81 +
replacement 1.29 = 8.00 u. A separate <=1 u canary funding ruling must be
recorded under the existing campaign authority before spending. Preparation
does not silently enlarge Y3 or treat the reserve as free credits.

1. Obtain genuine current-source final-only acceptance. Preserve it and its
   original evidence; provide `current-acceptance.json` at the artifact root.
   If transport code changes, reprepare at a new pin and canary root.
2. Retain `independent-review.json`. After the independent funding ruling is
   recorded in CAMPAIGN_LOG, create a new sidecar-backed `dispatch-review.json`
   carrying its exact source/transport/template identities and the genuine
   separate one-unit authorization. Those two dispatch prerequisites are not
   fabricated here. `preflight` checks them before canary allocation.
3. Confirm fresh empty account inventory; run **canary first**, using the
   preflight-chained command in DISPATCH. Inventory changes require a new
   immutable preparation. Both jobs declare `other_assignments={}`.
4. Only after the genuine broker terminal receipt is PASS with verified release,
   run `p1_family_canary admit`. It re-derives the shared family evidence from
   the saved manifest, broker transfers, measured staging, scanner exports,
   recovered bytes and attributed CUDA six-action decision. A local smoke,
   generic acceptance probe, summary or fake receipt cannot supply this proof.
5. The command creates `source-admitted.json` and `admission.json` once. Run
   supervisor validation, check fresh empty inventory again, and dispatch that
   new sibling. The original source03 manifest remains held and immutable.
   Source retains 75/150 min, cap 4.81 u including 0.96 reserve; replacement
   20/40 min, cap 1.29 u including 0.26 reserve remains conditional.

The canary broker receipt, current-source acceptance, dispatch funding review
and admitted source manifest do not exist yet. Both intended runtime output
directories remain absent. No dispatch-readiness or live evidence is claimed.

## Verification

Integrated focused selection: **83 passed in 1.77 s**:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/bl_ppo/test_p1_family_canary.py \
  research/bl_ppo/test_p1_canary_runtime.py research/bl_ppo/test_p1_runtime.py \
  research/bl_ppo/test_recovery.py tests/test_colab_cli_canary.py \
  tests/test_colab_cli_admission.py tests/test_colab_cli_accounting.py
```

Black@100, isort, flake8 on all five changed Python files, `bash -n
research/bl_ppo/p1_colab.sh`, and `git diff --check` passed. No broad suite ran.

Real CPU shell smoke copied both actual input files into a new local input
tree, used the manifest environment plus CPU device selection, and ran
`bash research/bl_ppo/p1_colab.sh`. It performed real public checkpoint
materialization and the complete canary runtime: **8.0448 s** full shell,
4.3270 s component loop, PAUSED at32, native `s2-native-v8`, restored snapshot
26,579,063 bytes. Every payload and sidecar verifies. The real
`runtime._exports` plus `schema.validate_export_inventory` accepted all three
exports. Evidence: `local-shell-smoke/verification.json`, `shell.log` and
`validation.json`. This is local smoke evidence, never the live family receipt.

Actual supervisor validation of the saved canary and upload-readiness
`--verify` pass. The source template without canary proof refuses exactly:
`dispatch requires a SHA-bound family canary receipt`. Independent review
rehashed the staged inputs, all12 extracted members, source closure, exported
bytes and sidecars and confirmed the same refusal and fresh output paths.

The compact sibling `P1_FAMILY_CANARY_PREPARED_03.json` binds the artifact
hashes, including DISPATCH. No protected package was edited; the forbidden
study directory was never opened. No tmux session, checkpoint, serving pin or
original hash-bound evidence was modified.
