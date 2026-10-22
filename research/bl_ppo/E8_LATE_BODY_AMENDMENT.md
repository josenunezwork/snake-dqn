# P1 E8 late-body support amendment — 2026-10-21

Status: OWNER_AUTHORIZED_PROSPECTIVE_PRE_ACTOR. Authority: the owner's explicit
2026-10-21 request to guarantee late-body critic evaluation support, prepare
replacement manifests, and publish the correction; **no Colab dispatch**.
This sibling amendment changes only P1's critic-refit state distribution and
qualification support construction. `REGISTRATION_P1.md` remains byte-identical
(SHA-256 `944aaf3798f9ad698698c5f7abe16e1f5325a034fb1b032ba3cc9387d9111a56`).
Source-02 remains a failed E8 / NOT_READY attempt; this amendment does not relabel
it or manufacture scientific admission, funding, release, or a live Y3 PASS.

## Source-02 diagnosis and evidence boundary

Evidence: `big-lever/exp/bl-p1/y3/source-02/colab-output/exports/origin/training-status.json`
under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`, SHA-256
`eaf1828f23c0aba469c3ae92500d294a14dbfebadf9d60c097f3d8ed22341c03`.
Its manifest pins `4bb1e32a7b6c960a2e23c3bab19715dee3d21bbc`.

The run collected 2,523,136 critic rows, including 2,010,624 fit-world rows;
actor transitions, actor updates and actor episode boundaries are all **zero**,
and actor_enabled is false. No P1 actor trajectory exists in these exports.
No new scientific actor or critic data are collected by this implementation;
Mac tiny tests are excluded software fixtures only.

The null EV/MSE/baseline and count-zero fields are **not proof of an empty
late-body set**. At the pinned source, `refit.qualification` calls
`learner.qualify_critic`, which raises before returning metrics on failure;
`run_train` then writes missing-metric defaults. The exported error is
`critic qualification failed: require late-body EV >= .1 and MSE < baseline`.
An empty L>=800 subset would instead raise
`critic qualification requires held-out L >= 800 rows`. Thus this evidence
records a rejected accuracy gate with lost metrics; its actual EV/MSE/support
cannot be recovered from these exports. No refit-bank snapshot was exported.

The operational length distribution confirms that long states were reached:

| Mix | L<400 | L400–799 | L800–1199 | L>=1200 |
|---|---:|---:|---:|---:|
| scripted | 266,521 | 223,429 | 230,048 | 127,618 |
| frozen | 520,157 | 255,773 | 71,429 | 257 |
| mixed | 434,143 | 282,201 | 111,199 | 361 |

There were 540,912 long-body decisions. The fixed ordinary bank's decile
boundaries were 53/121/194/275/366/468/582/740/938. These are collection/monitor
counts, not qualification support. The old bank splits whole worlds by
`world_id % 5 == 0`, retains eight hashed samples per life, and admits samples
to bounded reservoirs only after death/H5000. It neither guarantees a late
sample nor makes an unfinished life eligible. This construction can lack late
support even when the operational stream contains many long decisions.
The holdout stream had only 512,512 total transitions, so at least 28,400 long
decisions necessarily belonged to fit worlds. Neither raw fit exposure nor its
absence is inferred from the lost retained-bank support fields.

## Prospective support construction

1. Before fitting, collect exactly the first complete life in each of twelve
   dedicated fixed worlds, using the same frozen initial stochastic actor and
   established `collector._inject_long` / `inject_serpentines` construction used
   by long-body fixtures. Initial lengths are 900/600/400/300/150/150, including
   an L900 hero. Play until real hero death or H5000, retaining exact gamma-one
   suffix returns; rollout cuts never terminate a target. This is a constructed
   long-body distribution, not a claim of natural late-game history.
2. Fixed namespace is 2026102111. Allocate its first-life seeds against the
   existing study/evaluation exclusion sets, require uniqueness, then reserve
   those actual seeds against every training and calibration root. Use separate
   bank-world identities. The receipt binds root identities/digest, construction,
   horizon, collected and retained-target counts. Its actor RNG is isolated from
   the primary stream. All simulator work, including ignored later lives and
   whole-rollout padding, is charged.
3. Keep the first L>=800 row of each fixed life independently of hashed sampling
   and ordinary reservoir replacement; its target is admitted only when that
   life completes. Include the fixed rows in the same held-out late-body gate
   alongside whatever natural held-out completed rows the original sampler
   supplies. Retain provenance and separate support counts. No fixed held-out
   row enters optimization, PopArt target fitting, or baseline regression.
4. The primary critic refit starts naturally, as before. After at least 2M
   fit-world transitions, inspect **support only**, before EV/MSE. If its retained
   completed fit bank has any L>=800 rows, use it without changing training
   starts. If that count is zero, use one bounded critic-only supplementary
   stream: **25% of starts**, exactly one of every four roots per world, through
   the same serpentine construction. This fraction describes roots, not a
   promised transition fraction. Cap the supplement at H5000 steps per world
   (640,000 delivered transitions at 128 worlds); it does not increase any phase
   wall/unit cap. Keep supplementary fit identities separate from unfinished
   natural lives, and train only fit-world rows. Check support once at completion;
   if still absent, refuse. Do not examine accuracy and then add training.
5. Qualify once after the support construction and the existing complete-target
   fit pass. Preserve the original natural collector for actor continuation;
   injected worlds never become actor worlds. The ordinary 10k bank and natural
   late-body diagnostics continue to use actual natural refit rows. Fixture and
   supplemental work have separate provenance/counters and do not count toward
   100M actor transitions. Snapshots retain the full amended refit bank,
   qualification, support metadata and RNG/collector state for exact Y3 replay.

The amended code binds this file's digest and support recipe in the run identity;
preparations bind the complete executable source closure. Exact seed lists,
software preflight evidence and prepared manifest hashes are in the sibling
handoff/publication index. The absence of actor data makes this change
prospective for actor training; it does not erase source-02's accuracy failure.

## Unchanged requirements and operational scope

The gate remains **EV >= .1 AND MSE < the fit-world time/length baseline MSE**
on nonempty held-out complete-life L>=800 rows. Holdout rows never enter
critic optimization or baseline fitting. A finite failure retains numeric
metrics and support and refuses actor admission. Missing support remains a
refusal; nonfinite metrics retain the existing tripwire behavior.

The F model/temperature calibration, original training seed 20261021,
100,000,000 actor transitions, actor natural-reset distribution, opponents,
reward, gamma, horizon, learning rates, PPO epochs, monitors, tripwires,
evaluation bank, one-look decision, deadlines, budgets and reserve requirements
are unchanged. All added fixture collection/fit/export costs are charged inside
the existing phase caps; the 1M complete-job forecast must include them. No
extension or automatic retry is authorized. Replacement/qualification
preparations remain blocked on their actual predecessor evidence and catalogs;
empty `other_assignments={}` is a preparation assumption, never a live inventory.

No protected study packages, supervisor code, served checkpoints, serving pins
or tmux sessions are changed. The sealed study directory is never opened.
