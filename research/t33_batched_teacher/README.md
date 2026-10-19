# T33 many-worlds search teacher

Development-only distillation labels. The Phase 0 implementation and original
registration are imported read-only. No serving or promotion authority.
See `protocol.md` for the prospective calibration and inference checks.

## 1. Search-budget ablation

The default factorial is H32/H64 x N4/N8 x {masked-max Q, dueling V}, scale 1,
epsilon .02. The full reference is N8/H128. CRN shares epsilon RNG streams
across first actions at each rollout index; cloned food RNG and deterministic
seed/slot/frame opponent randomness are shared across first actions too.
The reference scorer preserves Phase 0's rowwise B8 evaluation. An additional
original independent-action N8/H128 reference records the CRN policy change.
There is no automatic choice of a winning recipe.

Full calibration command (operator only; three single-thread workers, fresh
output required; runtime is unmeasured, approximately an hour is the spec's
planning estimate):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-batched-teacher
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 10 ./venv/bin/python -m research.t33_batched_teacher.ablation --decisions 300 --workers 3 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/calibration-v1.json
```

Only the three audited calibration fixtures are used. Eligible bare-B8
decisions are sampled without replacement, then replayed without changing
visitation according to a variant's action. Each result includes both
reference conventions, scores, agreement, regret, timings, and identities.
The existing bank allocation digest and passing audit/file binding are checked
without opening absolute paths embedded in the Phase 0 audit.

`--margin-gate FLOAT` optionally searches only below that legal-Q margin;
skipped decisions still enter agreement and regret accounting. Synthetic
`--smoke --decisions 3 --workers 1` uses the disjoint L800 timing fixtures,
one per mix. Smoke results cannot establish quality of a cheap budget.
