# E9 pre-data E2/E6 funding rule and operating characteristics

Frozen before E2 data; development funding only, no serving authority. Authority:
external big-lever round-02 §(f), round-03 §(f), round-05 §(f) and QUEUE.md.
`rule.py: RULE` is the machine-readable rule. E2 uses one-sided family alpha .05,
E6 reserves another .05, with Holm across the fixed R/F slots in EACH route.
An unavailable arm retains its slot with p=1. E2 GO additionally requires
mass point >=10, unadjusted LB90>0, scripted no-death point delta >=0,
survival band v2, complete critic qualification and no scientific tripwire.
E6 funding requires Holm-positive early dividend and ordinary point >=-2,
with its operational and critic gates intact. This package does not modify
E6's runner: the E9 family reservation must be enforced before any E6 scientific
funding claim is consumed. Any positive E2 OR E6 route is family funding.

The larger prospective screen is 128/48/48 scripted/frozen/mixed, under namespace
`bl-ppo-e2-screen-20261021/v2`. The original 96/32/32 bank bytes remain immutable;
`bank.materialize(allocation_version=1)` reproduces and validates the legacy
allocation. Larger banks need a new collision audit and fresh exports. Choosing
the larger option does not override the power gate.

## Mac simulation and result

Executed locally with 20,000 replicates/scenario, seed 20261021, one math thread,
no gameplay or Colab calls. Historical variance source is tracked
`t23_r0_screen_20261017/results/t23_verdict.json`, SHA256
`e97234956a3a7434153850d5e4fbb8a8018064728c454a68cee36313e615e37b`.
Its measured paired mass SDs are 364.683 scripted, 262.732 frozen and 212.360
mixed. Equal mix variance is `sum(s_m²/n_m)/9`, giving mass SE23.456 at96/32/32
and19.484 at128/48/48. Neither SD200 nor row-weighted pooling is used.

The simulation covers true mass and early-dividend effects0/20/40/60/80,
survival deltas-.05/0/.03/.06 and shared-reference correlation0/.5/.9.
It applies both routes' Holm rules, E2's mass and survival/no-death gates,
E6's ordinary guard, and reports GO/KILL/INCONCLUSIVE. Integrity, completed
critic qualification and tripwire absence are explicit assumptions; a missing
real receipt still blocks scientific execution.

| Allocation | Worst family null pass | Pointwise MC95 upper | Minimum full funding at+40, nonnegative survival | Pointwise MC95 lower |
|---|---:|---:|---:|---:|
|96/32/32|.05050|.05362|.27045|.26434|
|128/48/48|.06160|.06502|.30700|.30065|

The full-funding requirement>=.60 fails even after the sole permitted escalation.
The null-pass requirement<=.10 passes in these models. At128/48/48, true+40,
survival+.03, funding ranges .42125–.61475 across correlation sensitivity;
at survival+.06 it ranges .53110–.76345. These are model-conditional probabilities,
not a positive forecast for the unrun E2 endpoints. Monte Carlo intervals are
pointwise simulation uncertainty, not simultaneous gameplay confidence bounds.
E7's26/128 point fraction is.203125 and one-sided Wilson LB90=.161447.

## Admission is BLOCKED; broader OC remains incomplete

`selection.dispatch_admitted=false`. `oc.validate_admission` refuses this report
before scientific collection. Do not dispatch E2 or consume this as a positive
power/admission receipt. The frozen options stop at128/48/48; no further N,
threshold, arm or guard change is authorized by this result.

S1's final paired variance artifact was unavailable in the explicitly approved
local locations. It is recorded as unavailable, never replaced by assumed SD200.
T23 supplies measured marginal mass/alive/episode variances. Marginal Gaussian
means and chi-square variance estimates are an approximation: shared-B8 mean
correlation is varied, cross-metric/variance-estimate covariance is unmeasured.
E6 early/ordinary precision uses T23 full-world variances only as a labelled proxy;
it is not an observed E6 variance. Ordinary true effect is assumed0. Actual E2/E6
critic, guard and operational failures can only reduce admissible funding.

Round-05's requested E16+E17+32M joint trajectory OC is explicitly
`INCOMPLETE_NOT_SIMULATED`: no registered measured joint rescue/retention/
collateral model or signature-forecast error distribution is available here.
False funding, false final efficacy, missed winners and expected units for that
future pipeline are left null. No invented persistence gate substitutes for E17.
This limitation and missing S1 prevent describing E9 as fully qualified.

Immutable output: external `big-lever/exp/bl-e2/e9-oc-02/report.json`, with
`.sha256` and `.complete`; SHA256
`7d316c74e7156a51e9f5a20d7529cca09489f42d5e14af6e120bfd3c31bd52cc`.
The report binds its `oc.py`/`rule.py` source hashes and the semantic rule SHA256
`0275c6fdc52d67a117f732461844622516233bd2674e5a188a82a571e6d4fad6`.
The earlier e9-oc-01 export is retained; e9-oc-02 supersedes it for planning.
