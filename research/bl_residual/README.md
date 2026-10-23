# F4 frozen-B8 residual dose study

[F4_REGISTRATION.md](F4_REGISTRATION.md) contains the exact round-26 judged
registration (SHA256
`66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65`).
Factory-label fits and reserve looks require the operator's approval JSON binding
that registration, the judgment and the post-change `FitRecipe` digest. The
original F3 freezes, receipts, draft and handoff remain historical evidence.
The independent `CODEX_V_F4_REVIEW.json` records verification of this revision.
No F4 result changes serving: B8 s3 remains served; GO buys only a separately
registered fresh Mac strict-gate plan.

`model.py` freezes B8, adds an exactly zero-initialized `Linear(256,6)` residual
on `a100_a2.model.b8_features`, and folds the fitted residual into a standard
`Ego2sNet` checkpoint. Offline action metrics use the reloaded fused checkpoint
rowwise, preserving the delivered controller's float32 action selection.

`FitRecipe` fixes Adam fp32, lr .001, 2,000 updates, seed 3403 and batch 256.
Each batch draws 128 doom and 128 ordinary rows, drawing a life uniformly then
a row uniformly within that life. Survival ranking uses every exact survivor
against every exact dying legal, served-selectable action in both strata, with
unit pair weights and pair-mean normalization. Ordinary growth ranking uses
only exact survivors with positive mass gaps, weights `min(gap,4)/4`, and
normalizes by eligible pair count. No prevalence weights enter either training
loss. Retention is **KL(p_B8 || p_corrected)**, temperature 1, all six actions,
on ordinary rows, coefficient 1; residual mean-square regularization is 1e-4.
The advisory mask and normal-only fallback define served actions. UNKNOWN and
NOT_EVALUATED branches never supply dying labels. Only the final checkpoint
is used; no early stopping, alternate seeds, sweeps or checkpoint selection.

`data.py` validates pickle-free F0 NPZ projections and carries life/parent
identities and sampling probabilities into feature caches. Parent hashes fix
folds; fold zero is held out. The fixed reserve selects 16 S-Watch6 rescue lives
then 48 other fold-zero rescue lives in allocation order, with at most four
lives per parent across both selections. Total doses 256/1024/4096 include the
same 64-life reserve, leaving 192/960/4032 nested fit lives. Allocation reports
parent and stream counts and unused support, while enforcing append-only prior
metadata. Missing support is INSUFFICIENT; it never shrinks N.

`heldout.py` freezes a versioned census of every natural ordinary row whose
parent is fold zero and allocation index is at most the reserve's maximum,
including non-rescue lives and parents absent from the rescue reserve. Admission
uses an empirical zero-event harm count H and life-macro rescue retention.
H counts changed actions where the candidate exact-dies and B8 exact-survives,
or where either action outcome is not EXACT. D4096 admission requires H=0 and
retention >=.70. Unclipped `1/sampling_prob` rates, clipped-at-five rates, ESS,
coverage, regret, override frequency and margin bins are descriptive. These
are not population harm confidence bounds. Parent-cluster bootstrap resamples
parents, retaining their whole life clusters.

Look R-A jointly reads the frozen D256 and D1024 checkpoints once; Look R-B
reads frozen D4096 once. Both are create-only. When F2 strikes volume (including
an unpublished disposition), funding requires life-macro D1024-D256 retention
>=.05, parent-bootstrap LB90>0 (10,000 resamples, seed3403), and D1024 H=0.
D4096 additionally requires a pre-dispatch cost projection from measured D256
and D1024 collection receipts: units/selected life x3072 x1.2, compared with
qualified-placement base caps. Earlier unconditional fit faults stop the path.

`gameplay.py` supplies three arms (B8/D1024/D4096), exact-clone H5000 starts,
delivered action traces and boost shares. A separate fresh probe bank has 32
worlds (11 scripted/11 frozen/10 mixed). D4096 drift tripwires compare served
argmax on candidate states by pooled length decile (>5%), probe boost share
(>1.5x B8), and reserve retention (<.70). Lower-dose drift is descriptive;
unconditional nonfinite/illegal/action-collapse faults remain binding.
The development screen fixes 128 paired worlds per mix. Watch6 runs only after
every H5000 GO bar passes: 128 paired continuous-respawn sessions, 14,400
frames, hero against five FRP3-s12+v8 opponents, dense rule off.

`analyze.py` implements R8/R9 with stratified Welch inference, the two-test
one-sided Holm efficacy family (alpha .10), unadjusted conjunctive guards,
survival band v2, terminal dose futility and paired-session product bootstrap.
It preserves INVALID/INCOMPLETE precedence, path-dependent NOT_RUN_BY_RULE
units, the ten ordered labels and TOP_PASSED_DOSE_FAILED. Its outputs are
create-only, identity-bound and one-look. Synthetic fixtures cover every label
and the registered equality boundaries. No development screen grants release.

Colab preparation uses only the qualified CLI subtree
`d0d92bdb572b770284a1fea6152417ffa5f06bea`, verified against the ops registry.
`make_pack.py` prepares fresh immutable packs; historical packs are never
rewritten. Dispatch remains operator-owned through `tools/ops/dispatch.sh`;
a prepared blocked canary template is not admission or execution proof.
A current-code, same-pin live family canary remains required before main use.

Caps (base/effective Colab units, doubled once): D256 40/80; D1024 increment
120/240; D4096 increment480/960; fits whole-series10/20. First wall/unit cap
wins, all attempts/setup/export charged. Resume ledgers bind verified immutable
complete-fit units and skip exactly verified units; missing units may receive
at most two manually admitted outcome-blind completion attempts, never a
scientific retry. Probes, reserve looks and gameplay stay on the Mac in permitted
gaps. Strict gates and serving qualification remain separately registered and
Mac-only; release requires the owner's OK.

Use the shared interpreter for light verification, with each numerical library
thread capped at one:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
  tests/test_bl_residual_*.py
```

See the F4 handoff and independent Codex-V receipt for the exact commands,
approval JSON and synthetic verification limits for this revision. No real
labels, reserve outcomes or gameplay results are needed to run these tests.
