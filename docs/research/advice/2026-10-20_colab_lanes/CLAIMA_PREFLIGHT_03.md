# Claim A amendment: local D0 preflight-03

**Outcome: fit path exercised; preflight incomplete; no D0 decision.** This attempt does not satisfy C2 dispatch qualification.

- Run: 2026-10-20 22:11:15–22:16:28 PDT, local CPU only, source `116be96b` plus C1 text/string edits.
- Wall time: **312.861 s (5 min 12.861 s)**; entrypoint exit 254 after an intentional SIGINT to this attempt's verified runner PID only. The driver records `FAILED` / `KeyboardInterrupt`.
- The outer watchdog allowed 1,140 s; it did not fire (`bounded_stop: false` in the supervisor receipt). The separate stop record documents the earlier operator stop after fit-path proof.
- Ran `d0.sh` in local mode through the Mac slot wrapper, `nice -n 10`, one cache worker and one fit thread; all BLAS/OpenMP/Numba caps 1. Held dev slot 3; AC, lid-open and thermal checks were clear.
- No registered rule, cell, dose, update default, threshold or batch was changed. The entrypoint exposes no preflight/fixture flag to reduce updates. The full registered cell grid was stopped during `frozen_trainable`, fold 0, after the recipe fits completed.

## Fixture and fit evidence

Fixture: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs/fixture-2`, packed by `research.t34_claim_a_loss.inputs.pack` into `fixture-2.tar.gz`.
Tarball SHA-256: `5a875d21d7f0a3473c628e7f936a5a08bc09787463cc26e3f7148b4a271ccfbe`.
Copied only complete world JSON files and their `.complete` / `.sha256` sidecars from the two authorized A5 export roots.
Both selected worlds carry decisive rows. N=2 is minimal for the requested training-fold support: with N=1, the only bearing world is held out in fold 0, leaving no training decisive rows.

| Fold | Training decisive rows | Training anchor rows | Held-out worlds |
|---|---:|---:|---:|
| 0 | 4 | 4996 | 1 |
| 1 | 8 | 2083 | 1 |
| 2 | 12 | 7079 | 0 |

The empty held-out fold makes this a fit-path fixture only. It cannot support a scientific D0 conclusion.
Replay accepted 2/2 worlds, rejected 0; cache regeneration took 11.275 s and retained 7,105 rows: 7,079 anchors, 12 decisive, 14 delay.
All four free-Q probes and all three `frozen_recipe` fits completed. The partial report records real optimizer updates:

| Recipe fold | Completed updates | Fit seconds | Seconds/update including predictions |
|---|---:|---:|---:|
| 0 | 5 | 7.570473 | 1.514095 |
| 1 | 5 | 18.300119 | 3.660024 |
| 2 | 5 | 0.307428 | 0.061486 |

Total: 15 recipe updates in 26.178020 s, or 1.745201 s/update including predictions. These small ratified-recipe batches and differing held-out workloads do **not** price full 1,024-row stream updates or CUDA execution. No stream fold completed before the stop; no full-batch seconds/update estimate is claimed.
`D0_DECISION.json` was not produced: **non-REFUSED decision reached = false**. The timing receipt's structural `registered: true` does not qualify this two-world fixture as registered evidence.

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_t34_claim_a*.py
```

**58 passed in 39.27 s.** Black, isort and flake8 passed for `decide.py` and `make_manifest.py`; `git diff --check` passed. AST comparison against `116be96b` confirms that Python edits change only a reason string and comments. The independent read-only C1 review found no remaining issue after removal of an unsupported trunk-update qualifier.

## Reproduction and immutable evidence

Build script: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs/preflight-03-build.py` (create-only fixture and tarball).
Exact local command: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs/preflight-03-command.sh`.
Supervisor: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs/preflight-03-supervise.py`; summary: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs/preflight-03-summary.json`.
These commands refuse existing run paths. A future attempt must use fresh paths; no rerun is authorized here.

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs`.

| Artifact under root | SHA-256 |
|---|---|
| `preflight-03-command.sh` | `4b3df38b6248af25b04ddf92d9011b4942cb56147c5c714b0e514a870199999c` |
| `preflight-03-supervise.py` | `b5f4a73a7b107f3ddc0f23cb20c23f62fb6c9a237d695b793dce99c7590a9170` |
| `preflight-03-fixture.json` | `70e957462361f3309793da61aeafab78439dd810b58649a8de6639ec35eafb6a` |
| `preflight-03-receipt.json` | `f034703fb398a1f3d26e5512869611ccd563c127f0dfd41dc68e57d2b17c109e` |
| `preflight-03-stop.json` | `d6f0dddc8eedcafb8547f8c750dfacbe4ef77a60437535b5c8122372b9abf8a6` |
| `preflight-03-run.log` | `ffae2c95ef627c4b01968ed24d0e6e88fed18756f0f1775375d5f1d7847bdd5d` |
| `preflight-03/out/d0_timing.json` | `33de236a1527d9e108b59b00ad19253683d873322ab08f264d10647049fde15a` |
| `preflight-03/out/cache_manifest.json` | `1c4d75cd1c50b0617e5925228226c0eadb21331f75d18a38cdb419e31febb459` |
| `preflight-03/out/d0_report.partial.json` | `007d7548ca9d3148b8c547c2083be624410efd17062d71bf5ad167aad5321c52` |

No Colab operation or main merge occurred. No Round 0 study directory was opened. Existing tmux sessions, released checkpoints, serving pins, and previously frozen evidence were untouched.
The next operator must complete the remaining C2 preflight/timing requirements before D0 dispatch; C3–C6 remain binding. This report grants no dispatch, student-selection, serving or release authority.
