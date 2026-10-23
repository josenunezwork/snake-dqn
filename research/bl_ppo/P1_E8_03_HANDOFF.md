# P1 E8 amendment 03 preparation handoff

**PROSPECTIVE DRAFT / NO DISPATCH**, 2026-10-22 PT. Branch
`p1-snapbound-20261022`; no fast-forward/merge of main. Astra review and Fable
judgment ratifying the new recipe and pricing canary/source remain required.
The independent Codex-V receipt is software review only.

The amendment is `E8_LATE_BODY_AMENDMENT_03.md`. Production opt-in uses P1
control boolean `e8_critic_convergence_amendment_03=true` with
`e8_critic_convergence_amendment_03_sha256` equal to the actual amendment digest.
The control loader rejects malformed booleans or unbound bytes, and run/snapshot
identity binds the enabled recipe. Omitted/false keeps the old one-pass recipe.
This describes the implementation; it is not a prepared execution control.
The original failed Y3 remains NOT_READY; D1 replay does not qualify P1.

Production API: `CompleteLifeRefitBank.solve_e8_critic_convergence_amendment_03`
selects natural completed FIT rows; `PPOLearner.install_exact_return_critic_solution`
performs the one float64 CPU solve, installs existing fp32 PopArt weights/moments
and clears the head's Adam moments only. The unchanged gate follows the step.

## Forecast

Frozen envelope **25.50u**, charged **7.26u**, frozen residual **18.24u**.
One canary: 15min estimate, 30min effective timeout, 2.00u effective cap,
.40u included reserve. One source: 75min estimate, 300min effective timeout,
9.62u effective cap, 1.92u included reserve. L4 ceiling **1.54u/h**, zero
tolerance. Estimates include setup/export and allow up to five minutes for
the new CPU solve within the existing source estimate. These are proposals,
not new allowances or provider timing measurements.

Combined nominal **90min / 2.31u**. Conservative combined cap liability
**11.62u**, so historic charge plus proposal is **18.88u** and leaves **6.62u**.
Holding the unchanged unadmitted replacement's **3.86u** leaves **2.76u**.
The 1M qualification remains blocked on a genuine Y3 PASS, separately priced;
no tranche, automatic retry or recovery is funded by this document.
Base/effective 2× accounting and evidence hashes are in
`P1_E8_03_FORECAST.json`. No estimate/rate/N was doubled.

## Offline replay command

Run only in a fresh output path outside protected Mac timing windows. This
loads existing hash-verified exports and the existing D1 diagnostic, installs
the production convergence solution in a disposable fp32 PopArt head, then
compares its predictions with D1. No checkpoint or old receipt is written.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.offline_critic_convergence \
  --sidecar /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-snapshot-bounds-staging-repin-20261022/family-canary/source-output/exports/origin/late-body-rows-index.json \
  --reference /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-snapshot-bounds-staging-repin-20261022/offline-critic-diagnostic-01/diagnostic.json \
  --out /absolute/fresh/path/replay.json
```

Each published local replay receipt has `.sha256` and `.complete`. It binds
the three original chunks, D1 diagnostic, amendment and actual helper code.
Numeric reproduction tolerances
are EV absolute 1e-4 and MSE relative 1e-3, fixed in the amendment before replay;
the actual gate thresholds are unchanged.

Two initial receipts are retained unchanged: `P1_E8_03_OFFLINE_REPLAY.json`
scored all 1,080 held-out rows before selecting the 162 late rows;
`P1_E8_03_OFFLINE_REPLAY_02.json` corrected the forward selection to match
the production gate (select late rows before the fp32 forward). Both reported
EV 0.5868173185655481 / MSE 307,527.84375, versus D1 EV
0.586716978151914 / MSE 307,567.66860707913. The relative MSE difference was
0.0001294832, inside 1e-3, but EV difference 0.0001003404 missed the frozen
1e-4 reproduction check by 3.404e-7. Both remain `REPLAY_MISMATCH`.

An independent builder and Codex-V algebra review then identified a PopArt
coordinate inconsistency: weights/bias were normalized using float64 fit
moments, but the existing forward de-normalizes with stored fp32 moments.
The corrected mapping uses the **actual stored** mean/scale to normalize the
same raw least-squares coefficients. This is a fit-only implementation
correction of the existing fp32 PopArt requirement in amendment section 2;
it changes no fit, features, statistic estimates, precision of the model,
gate or tolerance. A synthetic non-representable-moment regression checks
the algebra independently of held-out outcomes. The amendment bytes and
both earlier receipts remain unchanged. The final replay is retained as
`P1_E8_03_OFFLINE_REPLAY_03.json`, regardless of its tolerance result; no
further mapping variants or tolerance changes are selected from these data.

Final receipt 03: **REPRODUCED_D1**, I0 PASS, 4,120 fit rows, rank 258,
1,080 held-out rows with 162 natural late rows. Existing fp32 installed head:
EV **0.5867084270127363**, MSE **307,526.28125**, bias −104.9159927368164;
baseline MSE **561,561.6011360742**. D1 EV 0.586716978151914 / MSE
307,567.66860707913. EV absolute difference **8.55114e-6** ≤1e-4;
relative MSE difference **0.0001345634** ≤1e-3. Both unchanged numerical
checks pass. The raw least-squares fit MSE remains 896,454.5573538106 in
all three receipts, confirming the same float64 fit. Local solve/install
took **0.03028s**, total replay **0.29162s**; this is existing-export CPU
work, not a production throughput measurement. No new Colab data were read
or collected and no historical gate/admission changed.

## Final light verification

Independent Codex-V final batch: **67 passed in 1.13s**, exit 0, single
process (the repository's macOS OpenMP-isolation restart is sequential):

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -o addopts='' -p no:xdist -q research/bl_ppo/test_refit.py research/bl_ppo/test_p1_runtime.py research/bl_ppo/test_p1_manifests.py research/bl_ppo/test_offline_critic_convergence.py research/bl_ppo/test_offline_critic_diagnostic.py
```

Root final checks all passed: Black (line length 100), isort (Black profile,
line length 100), flake8 and `git diff --check`. The nine Python files checked
were `learner.py`, `p1_manifests.py`, `p1_runtime.py`, `refit.py`, `run.py`,
`test_p1_runtime.py`, `test_refit.py`, `offline_critic_convergence.py`, and
`test_offline_critic_convergence.py`, all in `research/bl_ppo/`. No full suite,
CUDA execution, training/collector trajectory or timed Mac benchmark was run.

`P1_E8_03_CODEX_V_REVIEW.json` with immutable sidecars records the independent
reviewer's scope, source hashes, exact commands, resolved findings and limits.
This review and the replay establish software preparation only; actual L4
canary, Y3 PASS, 1M qualification, funding and scientific approval remain absent.

## Boundaries and next action

No Colab/provider operation, allocation, dispatchable manifest, timed Mac run,
actor trajectory, CUDA change, main integration or release. Existing namespace,
ledger/resume and exact-pin canary requirements still apply before any future
operator dispatch. Preserve the old unexplained-assignment accounting warning;
a fresh inventory must establish all assignments/rates, not assume an empty map.
No tmux/process action, protected study access, serving pin/checkpoint edit,
prior hash-bound artifact edit, or `snake-dqn-l2mac` access occurred.

Next action is independent Astra review followed by Fable judgment on the
branch's concrete amendment, implementation, replay, forecast and Codex-V
receipt. Only after ratification may the operator prepare fresh identity-bound
controls/manifests/ledger and run the new exact-pin canary/source through the
qualified dispatcher. Existing source-05 controls keep the old recipe.
