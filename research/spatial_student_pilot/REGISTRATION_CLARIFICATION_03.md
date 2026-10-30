# Pre-data review fixes and feasibility disclosure — 2026-10-30

Owner-requested response to independent ACCEPT-WITH-FIXES review. No pilot or
ruler dispatch/data exist. This sibling preserves the original registration and
clarifications01/02. N, worlds, seeds, label/scoring suffixes, H5000/H300, final8000
updates, actor architecture, decision bars and scientific six-hour cap are unchanged.

## B1 / M2 — workload pricing

The disjoint development ruler collects for at most420 seconds. Collection pricing
uses synchronized tick durations from the final quarter of observed ticks, with at
least20 ticks and a living L600+ hero in every tick of that window. Cold/setup time
is excluded from rates; missing mature exposure refuses main admission.

Search uses at least22 naturally mature train roots, balanced by round-robin mix,
and enough legal-action/suffix jobs to fill a128-job slab. H300 slab loop timing is
synchronized and excludes restore/setup. Ruler slabs retain E128 after hero death
so early random-policy deaths cannot shrink the measured batch. A separate128-job
H300 closed-loop probe runs both B8 and student forwards, just as assessment's
mixed scoring slabs do. Assessment scoring slabs are priced at that student rate.

Each gameplay arm starts with the full production E128 chunk and retains its
size. Each has a300-tick cap and45-second loop cap (reduced if needed to reserve
30 seconds for export); fewer than20 ticks refuses rate qualification. Loop timing
excludes simulator/opponent initialization and state fingerprinting. Main gameplay
is charged as two full H5000 chunks per arm (128+64 charged at E128), independent of
early death and world-step counters. No short probe is presented as an H5000 result.
Search charges three128-job H300 slabs per32-root unit, conservatively using all
six legal actions; assessment adds two mixed scoring slabs per32 roots. The fixed
3600-second setup/export allowance remains separate. Scientific execution is
unchanged; fixed-slab retention and tick limits apply only to ruler diagnostics.

## H1 — restored-root action verification

The legal mask remains exact. A captured/restored B8 action mismatch is tolerated
only with finite Q values, a legal captured action, top-two legal Q margin<=1e-5
AND restored-best minus captured-action Q<=1e-5 (fixed absolute float32 epsilon).
Every tolerated mismatch is logged with expected/actual action, margin, gap,
epsilon and slab position. Larger gaps, changed masks and nonfinite Q still refuse.
Tick0 remains forced from job.action. No search label/tie rule or trajectory action
is changed by this verification-only tolerance.

## H2 — measured architecture feasibility; proposal only

A Mac development probe used random weights (seed1901),16 distinct natural C/T
H5000 long-body parents, hero lengths965–997,12 warmups and100 repeats per fixture,
one CPU thread, full live extraction/observation/upload/actor/readback. Historical
stage SHA256112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0.
MPS p50=3.3968755ms, p95=6.53252915ms; CPU p50=13.0279375ms,
p95=14.38788395ms. The observed registered-actor path fails4ms regardless of learning
quality; reproducing that latency at the completed endpoint forces KILL. These
random-weight historical fixtures do not constitute final-candidate qualification.
The Mac had background Spotlight/FileProvider activity; no unrelated process was
stopped. Do not dispatch the present architecture on an assumed latency pass.

Small single-layer/stem-width changes did not meet4ms. The smallest tested passing
proposal uses the existing encoder, directly gathers the full-resolution stem at
the hero head, concatenates deep2x4 context and36 scalars, removes the decoder and
full-resolution residual block, and widens the fused head128->160. It has2,013,550
parameters, retaining the2–4M range. Its random-weight MPS p50=2.1751875ms,
p95=3.93052465ms on the same16 fixtures (n1600). This is a narrow development pass,
not a guaranteed128-root final qualification or proof of globally minimal design.
It changes the registered architecture and is ONLY A PROPOSAL: it has not been
installed in model.py, trained, exported as a candidate or silently admitted.
A prospective independent architecture amendment, implementation and fresh proofs
would be required before adopting it. Retained scripts/results are under
`snake-dqn-artifacts/t34-sol/spatial-student-fix/`.

## H3 — mature-parent support

The historical C/T v1 terminal-hero H5000 collection reports265 eligible deaths
among768 deaths, but death eligibility is not the ever-L600 parent denominator.
Decoding its50000 anchor rows shows554/820 train parents and141/204 assessment
parents observed alive at L600+. Projecting these fractions onto the new split
estimates518.8683/768 train and176.9412/256 assessment parents, above256 and64.
Historical selection, split and namespaces differ; this is a feasibility estimate,
never imported science data or a guarantee of two sampled roots per parent.

The disjoint ruler independently counts unique observed parents reaching L600,
including living non-death parents, by its frozen train/assessment partition.
Main preparation requires projected train support>=256/768 AND assessment>=64/256
using these partial-run reach fractions (no speculative future-growth credit),
in addition to mature support in all three mixes. Main still requires all frozen
root counts/parent support; no replacement worlds or reduced endpoint are allowed.

## M1 / M3 — readout and export admission

A valid completed endpoint with fewer than16 baseline late deaths is KILL, per the
original registration's mandatory GO clause and "every other valid completed
result" rule. Integrity/count/completion failures still have precedence.

Ruler exports per-stage decoded and gzip unit projections and a total export
projection. It serializes mature root/compact records, extrapolates body span to
at least1000, applies25% headroom to all record bytes and charges record-wise gzip
without cross-record savings. Collection charges4096 demos+512 roots per unit;
search32 roots; fit uses actual full optimizer/RNG snapshots with5% headroom;
assessment/gameplay each reserve256KiB/unit. Charge all86 units, final actor and
96MiB miscellaneous exports. Main refuses any projected unit>512MiB decoded or
128MiB gzip, or total>2GiB. These are estimates; immutable publication and transport
continue enforcing their actual caps. Missing mature or fit samples refuse.
