# DENSITY TEST metric contract v1 — frozen before data

Development measurement only. Zero science, efficacy, promotion, release, or causal claims.
This file must be committed before any gameplay measurement. Its SHA-256 and source commit
are carried in every run. Do not revise definitions after data; add a prospective amendment.
No T34 world/artifact inventories are consumed. Namespace: bl-density-development-20261021.
These worlds are development-only and must never supply a registered efficacy verdict.

## Setups and allocation

One fixed bare B8 s3 controller, checkpoint SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
CPU, float32, rowwise inference, advisory mask with normal-only fallback, no hero veto.
Use the existing GridBatchSim evaluator, runtime prepare/observe hooks, standard deployment
profile and standard ordered scripted/frozen/mixed roster materializer. Frozen opponents
retain that evaluator's existing behavior; do not silently add a veto or substitute a pool.
Opponent policy carry and simulator RNG must be cloned, never reconstructed for branches.

- A: natural reset, standard map, standard five-opponent roster, H5000 terminal hero.
- B50/B70: natural reset; multiply both standard map dimensions by sqrt(0.5)/sqrt(0.7),
  round down to whole simulator cells. Seven opponents, obtained by cycling the standard
  ordered five-opponent roster. Keep absolute food settings and all other mechanics fixed.
  Report actual dimensions, area fraction, snake count and snakes per map cell. This changes
  food density as well as snake density; the combined setup is the measured intervention.
- C: standard map and five-opponent roster. Reuse bl_served_search.fixture_geometry.install:
  six disjoint serpentine body bands separated by two empty rows, all lengths equal to hero
  length, cycling 800/1200/1500 by episode index. These are nearby long opponent BODY bands;
  report nearest opponent head and body Manhattan distances at injection. No assertion that
  heads are close. The assigned frame 2500 is synthetic, not 2500 prior decisions. Run up to
  5000 actual decisions after injection. Report the installed geometry and synthetic status.

Production allocation: six episodes per arm per mix, 72 episodes total, six single-thread CPU
workers; C has two episodes at each injected length per mix. Seeds are deterministic hashes
of namespace/arm/mix/episode and are recorded, with distinct per-arm episode identities.
This is a small engineering comparison; low natural event counts may prevent realism estimates.
No seed/candidate selection, optional extension or efficacy hypothesis testing.

## Sampling, screen, and doom

Record length at EVERY live pre-action hero decision. Sample pre-action decisions t=0,50,...,4700
(the first action is t=0); each sampled state must be alive with logical length L>=800.
The last sampled root can complete all 300 decisions inside H5000. Moment counts are counts
on this grid, not counts of distinct traps: consecutive correlated roots may describe one death.

Cheap prefilter on the prepared selector snapshot: pass iff the minimum of the three existing
capped decision-time free-space counts is <160 OR the number of advisory-safe normal actions
is <=2. Use Vector61Runtime.decision_free_space; no tail-aware v8 flood is added to the screen.
This is a broad heuristic, not a certified enclosure detector. Record eligible samples, screen
passes/rejections, H300 calls and branch decisions executed. B8 itself already consumes its
v8 features; the prefilter saves continuation cost, not B8's normal forward cost.

Only passing roots run the existing bare continuation: clone complete world + opponent carry
inside the prepared selector; force the identical current B8 action, then continue bare B8
until terminal hero death or 300 transitions. Doom iff death occurs on one of these 300
transitions (including transition 300). Never expand alternative actions. Retain death offset,
hero length, features and root identity. This endpoint is a proxy for the requested late failures;
without death-cause evidence it does not establish self-enclosure or avoidability.

Arm A recall uses the uninterrupted original bare trajectory, not extra H300 on rejected roots.
For a sampled eligible root t and original fatal decision d, true doom iff 0<=d-t<300.
For a survivor to H5000, all sampled roots are non-doom. Verify every passing root's H300
label against this retrospective label; disagreement invalidates the measurement. Recall =
passing true dooms / all true dooms on the sampled eligible grid, per mix and overall. If
there are no true dooms, report null/INSUFFICIENT, never recall=1. Report rejected true-doom
counts explicitly. This is exact sampled-grid recall, not continuous-time recall.

## Efficiency and lengths

Per arm AND mix: screened-confirmed doom moments / total process CPU-seconds, and screened-
confirmed doom moments / ALL live hero decisions (including L<800). Report raw counts and
both denominators, also dooms per 1000 decisions. Sum per-episode process_time, including
world setup, features, cloning, continuation and collection; record wall-seconds separately.
Do not divide aggregate CPU work by parallel elapsed time. Record parent analysis CPU and
startup/import/compilation overhead separately; report end-to-end CPU-inclusive rate as well.
Any warm cache or tiny-run forecast must be labelled. CPU seconds are host-specific.

Length distribution: all pre-action live decisions, with count, min/max, p10/p25/p50/p75/p90/
p95/p99 and histogram <800,800–999,1000–1199,1200–1499,>=1500. Also retain sampled/doom
root lengths. Long injected roots are artificial, not natural history.

## Realism diagnostics

Compare each B50/B70/C doom-root distribution to A doom roots WITHIN the same mix. Use the
same screen on both sides. Therefore this is realism conditional on screen acceptance; missed
A dooms are disclosed through recall. Empty/low-count A is an explicit insufficiency, not
proof that an injected arm is realistic. Report C pooled and by injected starting length if
sample size permits, with per-feature sample counts. No pooled mix may conceal a mix shift.

Features: the 30 action-major v8 scalars already in B8's bv8 observation (six actions times
FEATURE_NAMES = area_g,head_risky,landing_ok,v5_eligible,spacious), followed by five statistics
over effective admissible Q values: q_gap (top1-top2, zero if only one admissible action),
q_max, q_min, q_mean, q_std (population). Effective mask is advisory if any action is safe,
else the three normal actions. Feature order is fixed. This uses Q value summaries, not a
claim that raw Q values are calibrated returns or the network's separate value-stream output.

Report per-feature two-sided empirical KS distance D=sup_x|F_A(x)-F_arm(x)|, sample counts,
no p-values. Report a classifier two-sample diagnostic: deterministic logistic classifier,
train-only standardization and episode-group holdout (all roots of an episode stay together),
balanced class weights; held-out ROC AUC with train/test episode IDs and row counts. At least
four episodes contributing doom roots in EACH class are required; otherwise INSUFFICIENT.
Constant features or nonfinite/malformed input are insufficient/invalid, never a fabricated AUC.
AUC near .5 means this classifier did not separate these samples, not equivalence; larger AUC
means easier separation. Correlated roots are not independent replicates. No inferential claim.

## Completion, bounds, and preflight

Tiny Mac preflight: nice -n 10, one math/Numba thread, one worker (never over two threads),
shortened episode horizon and tiny arm/mix allocation. It must exercise real B8, the actual
injection, clone and full H300 branch; tiny horizon/sampling results carry a preflight-only
label and never enter production summaries. No full H5000 Mac collection during K-1 timing.

One prepared L4 allocation, CPU inference only; <=90 min whole-session, <=3 units including
reserve, no dispatch in this task. Six CPU workers, one numerical thread each. Fixed complete
units are exported with .sha256 and .complete last; progress every ~120 seconds. Stop before
the internal work bound and mark INCOMPLETE if any scheduled unit is missing. Never treat a
truncated run as a completed comparison. Runtime downloads public checkpoint assets by hash;
only a tiny catalog and pinned source closure are uploaded. No retry or Colab calls here.
