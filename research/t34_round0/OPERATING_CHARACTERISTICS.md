# Round 0 operating characteristics (pre-data, development only)

This check runs no gameplay. It neither fits a student nor evaluates the registered
Round 0 bank. The fixed look contains 240 independent paired worlds, 96 scripted /
72 frozen / 72 mixed, with equal weights per mix. One teacher arm and one look form
the efficacy family; no across-arm or across-look correction is needed. LB90 means
a one-sided 90% Student-t lower bound. Survival uses one-sided 95% lower bounds.
The pooled statistic uses independent-stratum Welch–Satterthwaite uncertainty,
not artificial matching of indices from disjoint mix namespaces.

## Design choice and expected cost at +40

The implementation request explicitly requires the complete 240 paired worlds.
Fixed N is therefore the admissible design for this package. Early stopping would
omit required observations and needs a prospective operator amendment before data.
Fixed N is **not claimed to be cheaper** than an unconstrained sequential design.
There is no observed Round 0 teacher variance or supplied A5 throughput in this
package. The comparisons below borrow independent historical development variance;
they are sensitivity calculations, not empirical Round 0 performance evidence.

A hypothetical mass-only two-look design at 120 and 240 pairs, applying the same
mass point bar >=40 and a Bonferroni one-sided alpha schedule 0.05/0.05, would use
about 180 pairs on average at true +40 under the historical pools, compared with
240 fixed pairs. That is an optimistic approximately 25% cost saving: it excludes
the scripted-survival, delivered-boost and survival-band restrictions at an interim
look. It is not an adopted stopping rule, nor an estimate of full-controller GO
probability. Allocation, replay and chunk overhead are also excluded. The retained
fixed-N cost is 240 pairs / 480 arm worlds, plus the separate 24-pair / 48-arm Mac
replay audit. Runtime and credit costs remain parameterized by the operator's A5
measurement in the manifest plan; no throughput has been invented.

The point bar `mass gain >=40` is itself restrictive at the exact true +40 target:
with approximately symmetric sampling noise only about half of studies meet that
point bar. This package does not claim 90% GO power at true +40. More worlds alone
do not remove that boundary effect.

## Independent inputs and simulation

`oc.py` verifies the historical pool SHA before reading it:

- `research/survival_band_v2_20261006/survival_pools_20261006.json`
- SHA256 `6295400a2bee43f425e83a6f23dfb275f1d2cdb343e18ea4e7541c69900dc06a`

The FRP-v2 Phase 1 and Phase 2 development pools closed before FRP-v3. They are
independent of T34 outcomes and belong to an older checkpoint/veto lineage. Each
mix is independently resampled; within a world its historical mass/survival pair
is jointly drawn, preserving its correlation. Delta distributions are centered
and shifted to prespecified hypothetical effects. Phase 1 contributes 480 values
per mix and Phase 2 contributes 160. These are development variance models, not
additional observations of the Round 0 teacher.

| Historical pool | Mass SD scripted / frozen / mixed | Alive-fraction delta SD scripted / frozen / mixed |
|---|---:|---:|
| FRP-v2 Phase 1 | 240.38 / 197.06 / 180.71 | .3738 / .3363 / .2954 |
| FRP-v2 Phase 2 | 247.15 / 213.92 / 176.14 | .3566 / .3493 / .2717 |

Three normal stress models retain the Phase 2 mass SDs, take equal per-mix
alive-fraction SDs .25/.35/.50, and use within-world mass/survival correlation .97.
These are explicitly synthetic assumptions. Neither shifted bootstrap nor normal
models constrain hypothetical survival differences to [-1,1]; this is the
amendment's additive-shift sensitivity and its limitation, not simulated gameplay.

Each of the five models uses 20,000 simulations per prespecified mass effect
20, 26.8, 40, 60, 80 and each survival scenario: no change; -.05 in all mixes;
-.075 in scripted only; -.075 in frozen only; -.075 in mixed only. Seed is
202610200. Every simulation uses the actual unequal 96/72/72 geometry. The error
check requires maximum joint mass-and-survival pass rate under each protected
regression <=.06 (1.2 x .05). The floor is mean **alive-frame fraction** >=.30 per
mix, exactly the survival-band-v2 definition, not binary no-death survival.

To conservatively bound full-GO error, the OC omits the absolute-floor, scripted
no-death and boost bars from its pass event. Adding those intersections can only
reduce GO probability; the rates below are not full-GO power. No guard margin,
alpha or floor was adjusted after looking at the OC output.

## Results

| Model | Mass-only pass, true +40 | Joint mass+survival pass, true +40, no survival change | Worst protected-regression joint error | Wilson 95% upper for worst cell | Counterfactual mass-only expected pairs at +40 |
|---|---:|---:|---:|---:|---:|
| FRP-v2 Phase 1 | .49855 | .23945 | .02165 | .02376 | 180.552 |
| FRP-v2 Phase 2 | .49845 | .26175 | .02325 | .02543 | 180.048 |
| Normal survival SD .25 | .50040 | .41455 | .03670 | .03940 | 180.078 |
| Normal survival SD .35 | .49920 | .20120 | .01855 | .02051 | 179.724 |
| Normal survival SD .50 | .50425 | .05155 | .00750 | .00879 | 179.652 |

The complete create-only result is [OC_ROUND0.json](OC_ROUND0.json), SHA256
`66c0594ec359f1dfca3c2162c22c22af8ae71bccd076a45882b090eb90acff80`.
Its simulator digest matches the final `oc.py` bytes.

All prespecified error checks pass the .06 cutoff. These are simulation estimates,
not a proof for an unknown teacher distribution. The worst cell is at mass +80
and frozen -.075 in the first, second, third and fifth models, and mixed -.075 in
the SD .35 model. The Wilson bounds are per-cell descriptive intervals, not
simultaneous confidence limits across the grid.

The historical joint pass probabilities of about .24/.26 at +40 disclose a
substantial chance that the survival guard blocks a candidate with no true
survival regression. This package keeps the ratified guards unchanged. Mass-only
power under the two historical models is .0653/.0730 at +20, .1584/.17055 at +26.8,
.93195/.92905 at +60 and .99865/.99795 at +80. Scripted and boost bars can reduce
actual GO probability further.

## Reproduction and interpretation

Run with one numerical thread; `oc.py` also sets all relevant thread caps before
importing NumPy/SciPy. The output is create-only and includes the source-pool and
simulator SHA256. Use a fresh output path for reproduction:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../snake-dqn/venv/bin/python -m research.t34_round0.oc \
  --reps 20000 --seed 202610200 --out /tmp/t34-round0-oc-reproduction.json
```

`analyze.summarize` is a pure numerical helper and labels its evidence authority
`UNVERIFIED_NUMERICAL_SUMMARY`. Production `analyze_paths` requires the independent
ratification, bank and collision audit; the complete 240-world immutable chunk
inventory; matching checkpoint/source/runtime identities; exact world/roster
bindings; trace hashes; and the deterministic 24-pair Mac replay with no unexplained
divergence. It rejects missing, duplicate, malformed, changed and preflight data.
A GO is Tier 1 development premise evidence and confers no release authority.

At the sole registered look, unconditional faults produce STOP. A complete-data
failure of a point bar (scripted .66 / +.10, pooled mass +40, absolute alive .30,
guard-B scripted +.03, boost [.005,.10]) also produces STOP. Survival is a clearly
failed guard if its upper 95% bound is <= its NI boundary (-.05 pooled / -.075
single mix). Otherwise, when point bars pass but a required lower bound fails
with its upper bound still compatible, the result is INCONCLUSIVE. Missing or
invalid provenance and replay divergences are INCONCLUSIVE and never authorize
replacement or cherry-picking. Every GO clause must pass. STOP wording is
"this teacher/controller failed its registered bar"; no bare-ceiling claim follows.
