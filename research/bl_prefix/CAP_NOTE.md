# E6/E7 minimal cap proposal — full design retained

**Prospective proposal only; NOT DISPATCHED.** Original registrations remain
unchanged. No scientific source bank or R/F endpoints exist. Request independent
Fable funding ratification under the campaign's standing approval rule before
using this proposed profile. Resource-ready preparation does not grant dispatch.

The full-H5000 Mac fixture preflight completed 3 source worlds (14,478 steps),
3 restored continuations (3,867 steps), 36 E6 rollouts and 117 E7 rollouts.
The current-E8 compatibility check completed another 36 full E6 rollouts
(47,946 steps in 76.971149s with two workers). E7: 199,034 steps in 312.660509s
with two workers. Root lengths at frame3500 were1141/956/782. These are burned
operational fixtures, not scientific effects or measured L4 throughput.

Use per-worker seconds/step: parallel pool wall * workers / total steps,
compared with individual-worker/source rates. E6 conservative rate .0032107433;
E7 .0031417799. Preserve the immutable raw timing files; RATE_NORMALIZATION.json
corrects their derived parallel-rate fields without changing raw measurements.

Forecast =15min setup/export/classification allowance + full step bound * rate
*3 (unmeasured L4 CPU slowdown) /12 workers /60. E6 bound37.76M includes4096
screened sources,256 selected replays,128 additional terminal-sealer replays,
and every primary/secondary control and continuation. E7 bound24.96M includes
all ancestors,8 training prefixes, training B8 and both paired confirmations.
No N, secondary outcome, menu or design is cut.

| Job | Forecast | Forecast units at2u/h | Original cap | Minimum total cap* | Proposed cap | Reserve20% | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| E6 |520.156944min|17.338565u|4u|43.346412u|43.4u|8.68u|1041.6min (17.36h)|
| E7 |341.745108min|11.391504u|20u|28.478759u|28.5u|5.7u|684min (11.4h)|

*Minimum =forecast_minutes/12, retaining the registered2x timeout margin and20%
reserve at2u/h. Round up only to the next0.1u. Each timeout is below24h. First
wall/unit cap still wins; setup/export/failures charge the same allocation.
No allocation, retry or unit purchase occurred. All12 workers remain single
threaded, deterministic and CPU-only on the proposed L4 assignment.

Artifacts: big-lever/exp/bl-e6e7/production-preflight-v2/ and
production-preflight-v2-e8-check/. The second check retired two exploration IDs
used as operational environment RNG; OPERATIONAL_BURN_ADDENDUM.md excludes them
from science. The regenerated E7 plan uses3072 unburned audited identities.
Sealer attribution resolves only terminal causal closures with T1's method;
earlier onset remains explicitly unresolved and its coverage is reported.
