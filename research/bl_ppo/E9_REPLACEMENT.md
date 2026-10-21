# E9 replacement — round-09 Z-4 prospective candidate registration

**Status: PROPOSED — awaiting a funding judge.** 2026-10-21.
Development funding design only; no ratification, dispatch, scientific result,
strict-gate admission or serving/release authority. E2 remains **BLOCKED**.
This document proposes a prospective replacement; it does not amend the bytes or
semantics of `E9_RULE.md`, `rule.py`, the E2 screen, banks or admission consumers.
No E2 endpoint data were collected or inspected for this task. The OC draws below
are synthetic planning data using historical T23 variance, not E2 observations.

Authority: external `big-lever/round-09/JUDGMENT.md` §(f), Z-4. Its three designs
are evaluated in order a, b, c; a funding judge may consider only the first one
meeting both probability bars. None currently qualifies (numbers below).
No new sample-size, threshold, arm, guard or route search is implicitly authorized.
The independent funding ruling, resource invoice, implementation review, and
fresh-bank audit must all precede any future E2 collection. This proposal alone
cannot be passed to `oc.validate_admission` or `manifests.materialize_e2`.

## Frozen scientific scope and common rules

Comparator: released **bare B8 s3**, checkpoint
`b8_s3_t7_20261016.pth` (`3e649783...`); exact full checkpoint hash, final R/F
checkpoint hashes and complete critic qualification receipts must be bound before
execution. R and F are the existing fixed E2 endpoints (one training seed each),
not a population of recipes. No endpoint/checkpoint/seed selection from outcomes.
Keep E2 training, critic preparation, tripwires, parity and completeness semantics
as frozen in `INTENT.md`, `E8_AMENDMENT.md`, `E8_PREFLIGHT.md` and the current
E2 code. Missing mandatory receipts or cap expiry means INCOMPLETE and cannot
fund anything. Scientific tripwires retain their registered KILL precedence.

Unit: fresh paired H5000 terminal-hero world. Run B8/R/F on each world and the
same CPU model, dead frames zero. Primary estimate is the equal-weight average
of the three mix mean paired mass-integral deltas (never row-weighted), with
variance `sum(s_m²/n_m)/9` and Welch-Satterthwaite df. R/F slots are fixed;
unavailable or stopped arms retain p=1. Use one-sided Student-t mass p-values
and Holm at family alpha .05 across R/F. GO also requires mass point >=10,
unadjusted LB90 >0, scripted paired no-death point delta >=0, and band v2:
equal-mix alive-fraction LB95 >-.05, every mix alive-fraction LB95 >-.075,
and every candidate mix absolute alive fraction >=.30. At the final efficacy
look, mass UB90 <=0 is KILL for an eligible arm. Other complete, valid arms
are INCONCLUSIVE. Integrity/completeness precedence
and operational gates remain binding; assumed qualification in an OC draw is
never a real qualification receipt.

E6 uses its existing 128 early source worlds (44/42/42) plus 128 distinct
ordinary roots, with two future RNG rows averaged within each source world,
as registered in `research/bl_prefix/REGISTRATION_E6.md`. Counts in candidate b
below describe the E2 whole-world screen, **not a changed E6 root allocation**.
E6 has separate R/F Holm .05, early LB90 >0 and ordinary point >=-2, with all
critic, matching, restoration, collection and cap gates intact. Its ordinary
true effect in this OC is zero. Full-world T23 variances are an explicitly
labelled proxy for E6 precision, not measured E6 variances.

## Three prospective designs, in judgment order

| ID | E2 screen, scripted/frozen/mixed | Looks | Eligible funding event |
|---|---|---|---|
| a | 192/64/64 fresh paired worlds | One final look | Any full E2 GO OR Holm-qualified E6 NEEDS_SCALE |
| b | 128/48/48 fresh paired worlds | One final E2 look | Any full E2 GO only; E6 reported separately |
| c | 96/32/32, then **additional independent** 96/32/32 | Futility only, then one final efficacy look | Any full final E2 GO OR Holm-qualified E6 NEEDS_SCALE |

For a and c the combined family reserves .05 for E2 and .05 for E6. It already
uses Holm **within each route**, rather than a four-way Holm; no change to that
multiplicity structure is credited as a power improvement. For b E6 is outside
the E2 funding family: E6 cannot fund, select or revive an E2 endpoint under this
proposal. Report its NEEDS_SCALE probability diagnostically only; a later E6
funding rule would require separate prospective authority. No multiplicity
benefit is invented from merely calling the routes uncoupled.

Candidate c look 1: for each arm stop for futility if its stage-1 pooled mass
UB90 is **strictly below +40**. Equality continues. No look-1 success, GO or
funding is possible from E2. Stop all further E2 screening if both arms stop;
otherwise measure B8 and both endpoints on every stage-2 world, but a stopped
arm remains ineligible (p=1), even if its recorded stage-2 rows look favorable.
An interim mass stop (including interim UB90 <=0) is FUTILITY_NO_FUNDING,
not a scientific KILL; the mass KILL rule is applied only at the final efficacy
look to continuing arms. Scientific tripwires remain KILL at either stage.
In the OC nine-cell E2 summary, futility maps to INCONCLUSIVE, with per-arm
futility probabilities reported separately; tripwire absence is assumed. E6 remains a separate route and
cannot make a stopped E2 arm GO. No other outcome-dependent stage-2 selection.

At look 2 pool all stage-1 and stage-2 worlds for each continuing arm. For
equal stage sizes n, combine per-mix sample variance as
`[(n-1)(s1²+s2²) + n*(mean1-mean2)²/2]/(2n-1)`.
Recompute the final Welch statistic and all common guards on the pooled data;
retain the original R/F Holm slots. There is no early efficacy spending: the
final rejections are a subset of the corresponding one-look full-N rejections.
Stopping can only reduce funding. Choose sequential versus fixed by expected
cost at true +40 as required by P7 §5, after meeting the probability bars;
expected cost alone cannot rescue an underpowered candidate.

## Prospective banks and budget — no allocation or dispatch in this task

Reserve namespaces `bl-ppo-e9-replacement-20261021/a/v1`, `/b/v1`, and
`/c/v1/stage-{1,2}` under that same prefix. Only the judge-selected design could
be materialized later; these are specifications, not existing banks or audit
receipts. Stage 2 is disjoint from stage 1. Every scientific bank is fresh,
separate from training/calibration/preflight and all previously used scientific
worlds; reject the entire allocation on collision. Require full inventory
coverage of sealed studies through existing receipts; never open a sealed tree.
Missing coverage blocks collection. No materialized bank or receipt is claimed.

Proposed replacement screening ceiling: **120 units total**, including failed
attempts, setup, evaluation, export, liability and reserves. Training is outside
this proposed screen invoice and retains its own existing cap; this proposal
spends zero units. All candidates reserve E6's separate 4-unit cap within the
120, including b's diagnostic-only E6. No increase
of the existing E2 20-unit total/2-unit screen reservation is effective until a
funding judge explicitly replaces that envelope. Candidate a must fit the
original screen reservation to run under it; otherwise it is INCOMPLETE under
that authority, even if it could fit the proposed larger ceiling.

At most 960 controller episodes for a/c and 672 for b (three controllers per
world). A measured **evaluation controller-episodes/s** rate, with matching
policies, mixes, horizon, workers and length distribution, is required; E4
collection transitions/s is not a substitute. No qualified rate is asserted in
this report. Conditional invoice uses 2 u/h L4 planning rate, one 900-second
setup/export allowance, timeout liability 2× estimate and 20% reserve:
`cap = [2*(episodes/rate+900)/3600*2]/.8 + E6_cap`.
These are explicit planning assumptions, not a qualified runtime manifest.
Measured overheads or current rate can only replace these prospectively before
admission; never reduce N to fit or silently borrow training/reserve units.
The report gives maximum and expected episodes and cost/rate sensitivities.
No candidate passes the OC bars even before a rate/cap feasibility decision.

## Measured-variance OC and disposition

Implementation: `oc_replacement.py` reuses existing `oc.py` functions
`measured_inputs`, `_metric`, `_holm`, `joint_outcomes` and `wilson_interval`.
It verifies the immutable e9-oc-04 sidecars, T23 source bytes and unchanged
`oc.py` hash before simulating. The final JSON binds the new and reused sources.
Normal paired-world marginal means plus chi-square variance estimates follow
e9-oc-04. T23 8M SDs: scripted 364.683, frozen 262.732, mixed 212.360; source
SHA256 `e97234956a3a7434153850d5e4fbb8a8018064728c454a68cee36313e615e37b`.
Equal-mix mass SE is 16.586 at 192/64/64 and 19.484 at 128/48/48.

Grid: true mass **and early dividend** 0/20/40/60/80, true alive and scripted
no-death deltas -.05/0/.03/.06, shared-reference correlation 0/.5/.9. R/F have
the same true effect in each scenario. Metrics are independent except shared
B8 correlations within a metric/routes; cross-metric covariance is unmeasured.
Candidate c uses independent stage draws and correct pooled sufficient stats;
its E6 shared component correlates with the pooled final E2 mean. Guards use
the same historical alive/episode variances as e9-oc-04.

“Joint funding” means any eligible **full** route (the OR above), not both arms
or E2 and E6 simultaneously. The JSON also partitions E2-only/E6-only/both/neither
and all nine R/F E2 outcomes. Evaluate minimum joint funding over true +40,
**all nonnegative survival scenarios including zero**, and all correlations;
require >=.60. Evaluate maximum family null-pass over true zero mass/early
dividend and all survival/correlation scenarios; require <=.10. Use pointwise
Wilson MC95 lower/upper bounds conservatively for qualification. These are
simulation uncertainty intervals, not simultaneous or gameplay confidence
bounds. The .05+.05 efficacy union bound assumes valid marginal p-values;
simulation of the global null alone does not prove strong error control.

Executed on the Mac with one math thread, seed 20261021, **100,000 replicates
per scenario**, 60 scenarios per design (180 total). The source and immutable
report hashes below identify the exact run.

| Candidate | Minimum joint funding at +40, survival >=0 | Pointwise MC95 lower | Worst family null-pass | Pointwise MC95 upper | OC bars |
|---|---:|---:|---:|---:|---|
| a: fixed 192/64/64 | .36216 | .35919 | .06593 | .06748 | FAIL power |
| b: 128/48/48, E6 diagnostic only | .08648 | .08475 | .03342 | .03455 | FAIL power |
| c: 96/32/32 + 96/32/32, futility only | .35954 | .35657 | .06393 | .06546 | FAIL power |

Best simulated candidate: **a**, funding **.36216**, null-pass **.06593**.
No first passing candidate exists. Candidate c uses about 926.97–954.96 expected
controller episodes at +40 over the nonnegative-survival/correlation grid,
versus a's fixed 960; b uses 672. The small sequential saving does not establish
the required power. All three null-pass upper bounds are below .10.

Immutable report:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/e9-replacement/report.json`
with `.sha256` and `.complete`. SHA256 `3597a19fbf73c27d50067bf85862d55f0237b849e6d23123f4712079795a6179`.
A byte-identical external `E9_REPLACEMENT.md` snapshot and its sidecars freeze
this PROPOSED registration separately; neither sidecar is a funding ruling.

Reproduce from repository root with the shared project venv (export is
create-only; choose a fresh output path):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_ppo.oc_replacement \
  --prior /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/e9-oc-04/report.json \
  --replicates 100000 --seed 20261021 --out /tmp/e9-replacement-reproduction.json
```

The guard structure explains why more mass precision alone is insufficient.
For b at zero scripted no-death gain and rho=.9, either-arm passage of that
point guard is bounded by `.5 + acos(.9)/(2*pi) = .571783`, independently of N
under this normal model; additional guards only reduce E2-only funding.
This is a model-specific ceiling, not a statement about learned policy gains.
For a/c E6 can add funding, but its ordinary guard and proxy precision remain
binding. No tested Z-4 design meets both bars within the proposed 120-unit
screen envelope (indeed none meets power even ignoring cost). “Best” refers to
the three frozen Z-4 candidates; no claim about every conceivable unregistered
120-unit design is made. No post-result threshold or guard relaxation follows.

S1 variance remains unavailable as in e9-oc-04; no search of sealed inputs or
invented SD substitution. E6 variance/covariance remains a proxy. Normal
marginals do not reproduce policy dynamics or the unknown cross-metric
covariance. Critic/identity/parity/operational gates are assumed successful in
the simulation; actual failure can only block funding. E16/E17/32M trajectory
OC remains INCOMPLETE_NOT_SIMULATED. This document grants no pipeline forecast.

**Disposition: E2 remains BLOCKED; all designs PROPOSED, awaiting a funding
judge.** Under Z-4's no-passing-design branch, Line R's week remains E4 + E3 +
E14 + Y3 + envelope ratification, subject to their own registrations and receipts.
No E2 code semantics changed and no job was dispatched.
