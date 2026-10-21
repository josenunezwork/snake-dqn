# Final artifact family — round-07 X6 prospective registration

Status: REGISTERED_PRE_DATA_PREPARATION_ONLY, 2026-10-21. Read with
REGISTRATION_AMENDMENT_02.md. No nominees, final bank, efficacy look or independent
funding-judge approval is asserted. This is the protocol future registrations must
bind before data; it does not implement or authorize a final evaluator or release.

## Family freeze at the first strict-bank opening

The possible fixed artifacts are (1) bare R, (2) assisted B8, (3) planner over R.
Freeze the family immediately BEFORE opening the first strict bank: include exactly
the artifacts with a frozen nominee at that moment. A nominee must bind its full
checkpoint/controller/source identity and deterministic development nomination
rule. Planner over R is nominated from development, never constructed from a strict
winner. A slot without a nominee at freeze is excluded permanently and cannot be
added later. Included slots later withdrawn, unavailable or invalid receive p=1;
they remain in the denominator and cannot be replaced. No checkpoint swap,
new attempt, alpha reset after failure, or retroactive family change is allowed.

Create family-freeze.json with: schema bl-final-family/v1, UTC freeze time,
source commit, rule/registration SHA256s, complete included and excluded slot lists
with reasons, each included nominee identity and nomination-receipt SHA256,
comparator B8 full SHA256, skill/product design and bank SHA256s, sample sizes,
component tests, clustering, margins, look schedule, boost clause and approval,
burn-manifest head and look-ledger head. Publish payload+.sha256+.complete before
any strict bank opens. Missing values block opening; a template is not a freeze.
Timing/controller identity includes weights, planner, scheduler, W/K, deadlines,
fallback, numerical backend, load envelope, RNG and opponent carry.

## Composite p-values and one final look

For artifact i, the composite null is failure of at least one required inferential
clause: positive pooled mass, survival noninferiority at the registered pooled and
per-mix margins, Watch death-rate improvement, or Watch mass noninferiority.
Compute valid clause p-values using paired worlds for fixed-controller skill and
complete paired-session clusters for product; p_i = max(all required clause
p-values). Pre-freeze the exact tests, covariance handling and operating-characteristic
checks in each executable design. Missing/invalid clauses cannot be dropped or
imputed as passing. A family with missing executable statistical plans stays blocked.

Apply Holm stepdown at family alpha .05 to all m frozen nominees: sort ascending
p_i, compare p_(j) <= .05/(m-j+1), stop rejecting at the first failure. Equal p-values
use slot order above. With three nominees the thresholds are **.05/3, .05/2, .05**;
with two, .025/.05; with one, .05. No nominee at freeze does not consume a slot;
a frozen nominee later lost consumes its slot with p=1. Shared B8 controls are
allowed only under one prospectively frozen paired design; Holm does not require
independence across nominees. This controls false composite qualification, not
simultaneous confidence coverage or a population gain >=40 claim.

One final family look occurs after BOTH skill and product datasets are complete.
Operational integrity/stop checks confer no efficacy alpha and do not permit interim
outcome inspection or nomination changes. A sequential alternative requires a
separately approved pre-data within-artifact spending registration before opening
the family; it is not enabled by this one-look text. No new attempt after viewing
strict outcomes. Record every readout in LOOK_LEDGER.md's required ledger.

All stricter original product 95% requirements remain independent mandatory bars:
paired Watch hero deaths/1k interval excludes zero in the improvement direction;
full-horizon hero mass relative-difference lower95% >=-.05. The target of >=20%
fewer deaths remains a reported target. Point pooled mass >=+40, scripted no-death
gain >=+.03, survival band v2, boost, integrity and timing are additional mandatory
filters. Report every clause and nominal interval. Every comparator is bare B8 s3
SHA256 3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8.
No development/funding print, transfer receipt or calendar deadline substitutes for
fresh Mac strict/product evidence and separate serving qualification.

Bare wins an unresolved Claim A/Claim B tie: if both qualify and the paired A-B
pooled-mass two-sided90% interval contains zero, choose bare. Pre-freeze that paired
comparison; it is not a new superiority claim. If two assisted nominees qualify,
their selection rule must also be frozen in the executable design, never selected
postdata. Qualification recommends release; only the owner may authorize serving.

## Per-artifact delivered boost clauses, chosen prospectively

This registration selects the relative clause for EACH possible artifact, subject
to an independent funding-judge approval receipt bound before its efficacy data:

| Artifact | Delivered boost requirement |
| --- | --- |
| Bare R | 0.5 * s_bare <= s_R <= .10 |
| Assisted B8 | 0.5 * s_bare <= s_assisted_B8 <= .10 |
| Planner over R | 0.5 * s_bare <= s_planner_R <= .10 |

s is executed boosted hero actions /all executed living hero actions over the
registered terminal-hero H5000 skill rows, pooled across all mixes. Count delivered
fallback actions; proposed or stale discarded actions are not delivered decisions.
s_bare comes from the paired Mac B8 comparator rows with the same accounting; when
s_bare=0 the relative lower bound is zero. This is a delivered-artifact bar, not a
training tripwire; failing it fails that artifact's bar, not the mechanism globally.
It does not silently import the Path-B-only relative amendment or change any active
study. Until this artifact-specific choice receives its required judge approval,
its strict claim is blocked. Training tripwires and other product bars remain as
registered; no boost rule may be chosen after seeing data.
