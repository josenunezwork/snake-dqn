# T10 certificate/horizon diagnostics

Diagnostics only; no policy training, observation/reward modification, gate, promotion,
or serving changes. The binding scope is
`docs/research/specs/t10_certificate_horizon_spec_2026-10-17.md` and advice steps 1–2.

## Frozen measurement conventions

The certificate spec leaves “before it would be needed” without a numeric deadline.
The operational convention is **arrival at the actual current physical own tail by L
unit-speed cell steps**, with **128 steps** retained as a sensitivity check. The user
was asked about this convention asynchronously; no different deadline had been supplied
at launch. The target is the physical tail (ring-buffer `seg_count`), not any own-body
cell or just a TTL-one cell. Logical length can exceed the current segment count.

The flood is T9's existing `max(previous+1, destination_TTL)` four-neighbor relaxation.
All owners' current segments release at their current TTL; live heads seed zero.
Waiting is permitted. New trails, heading constraints, growth, respawns and feasible
movement during waiting are absent. Neither flag establishes an executable safety
certificate. Race-128 counts cells reached within 128 unit-speed steps strictly before
every live opponent; ties are excluded. Thresholds are **strictly above** L/2 and L.

Replay measurement is at every prepared, alive pre-action decision with L >= 800.
B8 follows the saved census action series. Dense follows saved trigger choices and
otherwise the same B8 rowwise greedy policy. All census predecision lengths, dense
trigger lengths/greedy actions, world identities, checkpoint hashes, intervention
counts, dense death decisions/lengths and both policies' endpoint survival, alive-frame
fraction and mass integral must match. No dense search is rerun. Opponents still use
the canonical replay dispatcher. Full scope is two arms × 32 episode identities ×
two policies. The two arms reuse the same 32 world seeds, so 64 episodes are not 64
independent worlds.

For fatal decision D and last eligible certified decision d, loss duration is D−d
recorded decision transitions (zero if still certified on the fatal decision). If the
certificate never holds in eligible states, the value is null with an explicit
status; if the episode never reaches L >= 800, it has a different null status. These
are not zeros. Missing length-eligible stretches break consecutive runs. Conditional
survival is alive at H5000 among episodes that ever hold the flag at L >= 800 for k
consecutive decisions, for k = 1,5,10,25,50,100,200,300. Each episode enters once;
this is a retrospective descriptive association with survivorship/selection effects,
not a causal effect or a future survival guarantee. Decision-landmark denominators
are also retained, but are not treated as independent samples.

## Horizon fits

Use the immutable 19,976-state T5 bank, all 103,264 legal branches, saved five
world-grouped folds, fixed T5 GBM and its 2,000 whole-world percentile bootstrap draws
(seed 20261017). Relabel known recorded outcomes as death on or before 30,60,100,200,300
transitions. A death exactly at the horizon is positive. A surviving D300 branch is
negative for every requested horizon. Illegal/unrecorded branches never enter a fit.
Every design appends six identical action one-hot columns. Input sets are 60 scalars,
60+24 T5 features, 60+24+18 T8 arrival features, and T5b fuse256+Q6 separately for each
of the five B8 finals. No best final is selected. No CNN/raster model refit is in scope.

Input order is aligned and checked against canonical state identities; saved folds
must equal the original allocation. OOF predictions and fixed model settings are
saved. The audit recomputes labels, fold identities, AUC and the full bootstrap from
saved predictions and requires exact D300 T5/T8 prediction parity. Source and input
hashes must remain unchanged during each job. Raw outputs remain in the external
artifact namespace; compact numeric evidence is committed with the final report.

## Reproduction and process ownership

Use fresh output directories. Each command uses exactly one numerical process with
one numerical thread. Run at most two such processes concurrently. No tmux operations,
external process control, GPU, network compute or shared environment changes are used.
Do not run the audit/tests while both computation slots are occupied.

```sh
export PYTHONPATH=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/deps:.
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
out=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t10-certificate-horizon-20261017
./venv/bin/python -m research.t10_certificate_horizon_20261017.certificate --out "$out/certificate"
./venv/bin/python -m research.t10_certificate_horizon_20261017.horizon --out "$out/horizon" --boot 2000
./venv/bin/python -m research.t10_certificate_horizon_20261017.audit --root "$out"
./venv/bin/python -m research.t10_certificate_horizon_20261017.report --root "$out"
./venv/bin/python -m pytest -q tests/test_t10_certificate.py tests/test_t10_horizon.py tests/test_t10_audit.py
./venv/bin/python -m flake8 research/t10_certificate_horizon_20261017 tests/test_t10_certificate.py tests/test_t10_horizon.py tests/test_t10_audit.py
```

Git publication after **every** task commit is:
`git fetch origin && git merge origin/main`, then `git push origin HEAD:main`, then
`git push origin t9-20261017`. Preserve concurrent upstream work and never force-push.
