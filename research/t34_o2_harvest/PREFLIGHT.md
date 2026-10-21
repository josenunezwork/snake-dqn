# O2 preparation verification — 2026-10-21

Outcome: **PREPARED_NO_DISPATCH**. No Colab allocation, authentication, job
execution, or O2 selection trajectory was performed. No serving change.

The published executable pin is `f1ca784e979a5c162e869f8cf670b5dc68ad8b32`
(implementation commit `de63dda7`, then concurrent-main merge). The pre-data
registration commit is `0e44585d`; registration SHA-256:
`b8e085be268e05cf84fd31adb72de2262d7f23e95d8ce4e2bd81349990d82bac`.
The registration binds 251 files. Later main merges preserved all bound bytes.
This file reports verification; it changes no frozen rule or registration.

## Implementation and independent review

New package only: `research/t34_o2_harvest/`. The D rank delegates to the existing
rescue ranker; D+O uses the same all-action tree, rescue priority, surviving bare
and alternative H500 continuations, and the O1 integer attribution threshold.
The exact O2 online schedule and <=24,000 search-row allowance per assisted arm
are disclosed in the preregistration. This is an unpaced ranking comparison,
not equivalence to Path B's asynchronous delivered controller.

Independent read-only review checked the controller, paired identities,
provenance, statistics, source binding and transport. Found issues were fixed
before freezing. These included root preparation, runtime portability, artifact
volume, receipt order, fault serialization/classification, archive discovery,
and bounded in-flight work. Typed faults admit no additional worker wave.
No reviewer or implementer opened the prohibited sealed Round 0 artifact root.
The existing O1/T34/T3 machinery and qualified transport were imported read-only.

## Freshness and statistical disclosure

The 96 selection worlds (32/mix) and nine excluded fixtures are collision-clear:
105 identities, seven historical FRP exclusion inventories, 34 published bank
inventories including O1/L2/E-series, and reconstructible hash-only namespaces.
Immutable package supplements cover incoming main through `22643c90`.
External `freshness-publication.json` checks the next 40 incoming changed files
and refreshed inventories through publication `9d5ad1d3`; no collision.
Opaque binaries, unrecorded external runs and other scan limits remain explicit.
The sealed root was pruned before history traversal; its published bank supplies
the identities. Unmaterialized E9 replacement namespaces remain prospective.

The 20,000-replicate OC is a historical-model sensitivity, not O2 outcome data:

| Scenario / property | Range across registered historical models |
|---|---:|
| True B8 gain +40, incremental +10: unadjusted B8 LB90 positive | 70.50–72.39% |
| Same scenario: both Holm contrasts | 13.16–25.06% |
| Same scenario: both contrasts and observed mean >=40 | 10.22–18.55% |
| Same scenario: conditional full pass | 0.810–1.495% |
| Global null: any Holm rejection (nominal 10%) | 9.84–10.02% |
| Global null: conditional full pass | 0.105–0.195% |
| Incremental null: mass/point-rule pass | 9.655–9.89% |
| Guard A acceptance at no change | 4.32–4.71% |

Conditional full-pass rates assume passing boost/absolute floor, no faults and
the stated scripted no-death effect. **N96 is underpowered against an 80% target.**
At true +40, the observed mean >=40 rule limits full-pass probability to about
50% under symmetric continuous error for any N. Historical normal planning
suggests N168–180 for conservative +40 superiority alone and N735–807 for
guard A; these are prospective calculations, not new banks or extensions.
The judgment's 96-world / <=24-unit tranche is retained and no adequate-power
claim is made. More compute does not remove the point-bar boundary.

## Actual Mac preflight and process identity

The frozen registration preceded both excluded preflight runs. Both used the
real launch entrypoint, `nice -n 10`, CPU inference, one numerical thread per
worker, 40 live frames per arm, fixed roots 10/20, H500 continuation and no event
roots. AC power, open lid and no reported thermal warning were checked.

- Serial: three paired triples, one worker, **75.419317167 seconds**.
- Parallel: the same three paired triples, two workers, **43.490128583 seconds**.
- All nine arm records match byte-for-byte after removing only
  `elapsed_seconds` and `search_elapsed_seconds` recursively. Actions, state
  digests, config, roots, scores, counters and provenance remain compared.
- Proof: `serial-parallel-proof.json`, SHA-256
  `8023597f34b2faaf1e4878ce5f133347e465a782465c5a1a4c1b52b4cfd4431c`.

This proves the tested same-Mac serial/pool records. It is not cross-platform
identity, paced timing, representative late-body throughput, or efficacy.

## Forecast and manifests

Slowest warm measured rates: base **256.5737 rows/s**, complete shared-search
**263.7900 rows/s**. The forecast charges three study worker waves plus one
conservative fixture wave, 3x CPU slowdown, and 600 seconds setup/export per
chunk. It covers the registered **6,048,000 study-row upper bound**.

- Forecast: **3,485.107623 seconds / 58.0851 minutes per chunk**, rounded to a
  **59-minute estimate**; **2.9043 aggregate runtime-hours / 5.8085 units** at 2 u/h.
- Each job: 32 paired worlds, 11 CPU workers, L4 host with >=12 vCPUs, no CUDA
  inference, 240-minute wall cap, 8-unit cap, 1.6-unit reserve.
- Aggregate: <=24 units /12 runtime-hours, including 4.8 units reserve.
  Each job is below 24 hours; account concurrency is at most three assignments.
- This is a Mac extrapolation, not a measured L4 result. Each runtime must finish
  its excluded H5000 timing triples and admit the independently checked forecast,
  then pass three observer/plain parity fixtures before study work.

All manifests passed the qualified supervisor's local `validate` and local
`source.prepare`; no allocation API or live Colab command was invoked.

| Manifest | World slice / mix | SHA-256 |
|---|---|---|
| `chunk-00.json` | 0:32 / scripted | `207a18f2b48075034c13916f50cf28401988db299c0d64f8a5c71dce8d6aa3c1` |
| `chunk-01.json` | 32:64 / frozen | `f92ff25dae038c96439dd4e5e659ffd48260d98608eabe91142525f1c32a3643` |
| `chunk-02.json` | 64:96 / mixed | `58aa221f4eef93be0ff36046fa36b64e0db05c2d1844da14b9b59d5c56c4be9f` |

## Verification and delivery

Current package: **56 targeted tests passed in 7.43 seconds**. The separate
targeted package/source/supervisor run passed **130 tests**, with three
deselections, in 593.01 seconds. Black at 100 columns, isort's Black profile,
flake8, Bash syntax and registration/source checks passed. No broad project
suite, training, paced probe, strict gate or serving qualification was run.

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o2-20261021/`.
It contains immutable preflight bundles and sidecars, manifest/input copies,
proof, forecast, freshness supplement, local validation and `DISPATCH.txt`.

- `delivery.json`: `a093b6314410bf4419adc5fad2613b2150a7ca39525f0a89d99937e130a2d321`.
- `manifest-validation.json`: `1d79f23060c380f336c63feb255604aafb9003b92d396b5737e7a1064cfe49b5`.
- `DISPATCH.txt`: `d1b0a55c4b645dfe49371859f24fa1c4bebc7a761abc9beb5d896ee81960aade`.
- Serial bundle: `ef40e59f68648dcab6cda25004274a3484c78d2da29153ba2b5e3af2417f1cf8`.
- Parallel bundle: `3716c9156325002cab194259b1920590fc2161e4e5a7b43e120d259c29452c89`.

Future one-look analysis accepts original stopped supervisor publications,
not arbitrary extracted directories. It consumes the canonical look marker.
A GO remains development evidence only: Mac paced test, fresh Mac strict gate,
serving qualification and owner release approval remain required.
