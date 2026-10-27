# TRAIN-01-11: resident complete-life critic preparation

Prepared only; no dispatch. This owner-requested execution successor replaces
TRAIN01.md's E24 critic collection and CPU observation staging for packet 11.
Calibration remains E24. PPO's E1024/T256 recipe, telemetry, temperature solve,
critic loss, PopArt, Adam, fit/holdout gates, natural death/H5000 completion,
canonical task and opponents, monitoring and nomination rules remain unchanged.
Prior packets and the running packet-10 job remain untouched.

## Cohort and bounded storage

Critic preparation uses E360: 270 fit worlds and 90 holdout worlds, exactly
90 fit and 30 holdout worlds per scripted/frozen/mixed mix. An observation arena
retains the original seven 31x31 and four 37x37 uint8 planes and sixty float32
scalars. Rows retain scalar provenance and an integer arena slot. One indexed
copy per selected tick stages observations on the critic device; optimizer and
frozen holdout batches gather directly from it. No lossy encoding, artificial
life cut, or observation recomputation is introduced. Completed natural lives
still receive the exact gamma-one suffix return. The original shuffled,
without-replacement, cross-life 2048-row batches and eight-life contribution
bound remain intact. Slots are reclaimed after consumption; fit tails and
censored open lives release before the fresh holdout seam. The independent E6
fixture matches the frozen pre-change implementation's complete preparation
state, diagnostics, actor/critic weights and generator state exactly.

The maximum retained inventory is bounded by
`270*5000 + 7*5000 + 255 + 360 = 1,385,615` rows, including one whole tick before
draining. The arena permits 1,400,000 rows. Each row occupies exactly
`7*31^2 + 4*37^2 + 60*4 = 12,443` device bytes, hence 17,420,200,000 bytes
(**16.224 GiB**) total. CPU row objects contain metadata and slots, with no
per-row observation tensors. A conservative metadata allowance is 1 GiB of
host RAM, separate from device peak. Critic fitting, scalar baseline fitting
and quality analysis keep their previous definitions.

## Shared cap, fresh holdout, and seeds

Every simulated world row, including currently unused fit/holdout worlds, still
charges the single 8,388,608-row critic cap. Whole E360 ticks end at 8,388,360;
the last 248 rows are unused. This is the same whole-tick-floor rule as E24's
8,388,600 endpoint, with 240 fewer usable rows. The cap is not reset at holdout.

E360 was chosen over E512 to preserve room for fresh complete holdout lives.
After draining a fit tick, open lives retain at most `270*4999` rows and an
undrainable completed pool retains at most `7*5000+255=35,255`. Conservatively,
fit reaches at least 2M optimizer rows by tick 12,538. Clearing a straddling
holdout life and then completing its next eligible life takes at most another
10,000 ticks. Thus all 90 holdout worlds can contribute a complete eligible life
by tick 22,538, before the 23,301-tick cap. This count bound assumes valid task
execution. It cannot guarantee the mandatory 160 heldout length>=800 rows or
passing quality; either can still fail preparation before PPO begins. E512
would allow only 16,384 ticks and could fail count support solely from the seam.

The exhaustive 114,299,008-root audit and registry from packet 10 are reused
byte for byte. Allocation namespaces, intervals and seed derivation are unchanged;
all candidate roots remain covered. Critic roots stay in
`[300000000, 308389632)`. E360 uses 23,304 episodes per world, leaving 192 already
audited roots unreachable. Both cohort sizes are divisible by four, so canonical
root-index modulo-four fit/holdout ownership is preserved. Cohort changes do
change root-to-roster pairing and action/shuffle RNG interleaving; no identity
of E24 and E360 sampled trajectories is claimed. No new seed domain requires
a new exhaustive audit.

## Measurement and A100 projection

Create-only evidence is under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-11-engineering-20261027`.
All CPU numerical work uses nice 15 and thread caps of one. The real task fixture
uses production dimensions, B8 s3, both frozen checkpoints, and scripted rosters;
one warmup and three measured ticks per cohort produced:

| CPU cohort | Mean tick seconds | Charged rows/s |
| --- | ---: | ---: |
| E24 | 3.53797 | 6.78355 |
| E48 | 7.61243 | 6.30547 |
| E96 | 16.72614 | 5.73952 |

These short initial-world CPU measurements establish task and memory scaling;
they contain no completed lives and do not establish CUDA throughput. The actual
resident arena fixture, at E360 observation shapes and a scaled 16,384-slot
capacity, gathered 2048 rows / 25,483,264 bytes in 1.267 ms on CPU. The unchanged
sampler drained a larger E512-like 384-life, 1.92M-row wave in 11.078 CPU seconds
(937 full batches), an upper-cohort control for the selected E360 path.

The E24 task graph contains 43,122,192 unique tensor-storage bytes. Direct-ratio
scaling to both the already-resident E1024 training task and E360 prep task
budgets 2.316 GiB. Actor/critic/Adam tensors add 0.031 GiB. Including a 4 GiB
scratch/activation/allocator allowance gives **22.57 GiB projected device peak**
on the observed A100-SXM4-40GB. This is a sizing estimate, not a CUDA memory
measurement; long-body scratch/allocator behavior remains unmeasured.

The supplied live E24 measurement is 6024 charged rows / 60 seconds = 100.4 rows/s,
or 239.0 ms/tick. If its eager launch cost remains dominant, widening to E360
projects **1506 charged rows/s**, with a 1130 fit-slot rows/s ceiling before
natural completion. Staging writes only 3.36 MB per fit tick and each minibatch
gathers 25.48 MB, so a 10 GB/s device-copy allowance would cost about 0.34 ms/tick
and 2.55 ms/batch; these are assumed bandwidth budgets, not hardware benchmarks.
The fixed kernel/control dispatch count does not scale with world count, but
tensor work does. The CPU measurements alone cannot prove the launch-dominated
assumption. Retaining the original sampler also retains its per-life CUDA
randperm/host-synchronization costs.

At 239 ms/tick, the synchronized H5000 example completes fit and fresh holdout
by tick 15,000 (59.8 minutes); the conservative count bound of 22,538 ticks takes
89.8 minutes and the full charged cap takes 92.8 minutes. Add the observed 15.6
minutes for unchanged E24 calibration and a **10-20 minute allowance** for critic
fitting, sampler synchronizations and endpoint work: projected full preparation
is **about 90-125 minutes; plan on roughly two hours**. At twice the projected
tick cost, plan on approximately **3.5 hours**. At unchanged 100.4 rows/s, the
8.39M-row cap alone would still take 23.2 hours: the large-cohort GPU scaling is
a material unmeasured assumption. No A100 speed, capacity or quality pass is
claimed by these CPU fixtures, and no telemetry or PPO recipe change is used
to manufacture one.

## Packet and handoff

Fresh identity: `sandbox-a-train01-20261027-11`. The fresh detached checkouts are
`/Users/josenunez/Projects/ml/snake-dqn-train01-11-broker.noindex` and
`/Users/josenunez/Projects/ml/snake-dqn-train01-11-runtime.noindex`.
The empty ledger is source-bound to the new runtime pin. Qualified
`background-sparse-v1` delivery and its bounded launch-exec test are retained.
The frozen source hashes, exact operator command and packet identities are in
`packet-train01-11/SEAL.json` and `DISPATCH.txt`; verification commands and results
are in `VERIFICATION_TRAIN01_11.json`. State/output under `train01-attempt-11`
remain absent until operator dispatch. The technical envelope stays 8h base /
16h effective, with no new science cap or automatic retry. Dry-run admission
does not allocate or check current account inventory. No running job, transport,
serving code, checkpoint, tmux session, or Mac lease is modified.
