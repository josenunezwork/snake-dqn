# R30-1 post-F4 G5 mechanical preparation — HOLD

Prepared from clean build `94635bd682a4b911b84a7fce002bb199ffa640a4` for runtime
`a7893b97c73fcf3e7d7cef4bb3990636aa147f6b`. No runtime source changed.

Canary operational receipt PASS, CUDA observation, all three exported payload
sizes/SHA-256/sidecars/completion markers, and recorded verified stop were checked.
No live Colab recheck occurred. Export payloads were hashed without deserialization.

Canary supervisor wall: 1,100.253499 s; worker aggregate: 715.993883 s.
Transfer 102.130675 s <=150 s; staging 173.496517 s <=300 s.
R30-1 allowance: 144 - 18 historical - 20 canary liability =106 min.
A measured main estimate must be <=53 min to pass the 2x guard.

Permitted receipts lack the phase timings needed to project 3 to 2,000 Adam
updates and 2 to 500 solver evaluations. No measured full-main estimate <=53 min
is identifiable. The requested positive forecast field records only an explicitly
labelled unseparated-cost screening surrogate:

`(1100.2534990829881 - 715.9938825959999 + 715.9938825959999 * 2000 / 3) / 60`
`= 7961.891911341448 min`, rounded upward to `7962`; `2 * 7962 >106`.

This is not a calibrated full-main prediction or certified upper bound. The
HOLD forecast fails the executable guard. Solver/certificate limits are caps,
not measurements. Independent read-only mechanical review also recommends HOLD.

Per the owner stop rule, main packaging and its dispatch dry-run were not run.
No main job or dry-run output exists; no G5 ACCEPT is claimed. Frozen template,
series, inputs and native bundle remain unchanged. No Colab calls, dispatch,
scientific/diagnostic deserialization, D4096 access, or t34-round0-study-01 access.
Local work inherited nice 15 with numerical thread limits 1.

New immutable artifacts (each has SHA sidecar and completion marker):

- packet: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/G5_REVIEW_PACKET_01.json`
  SHA-256 `f4d2f3073ec2d64356ab002b0aec302ed11ed5088ca6d39012a5916f05d7fd81`.
- forecast: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/runtime-forecast-reviewed.json`
  SHA-256 `f72a70c856196d4b5861b94037a155844f06a65cdafd31fe997334aa2ca88f77`.
- arithmetic: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/runtime-forecast-reviewed-arithmetic.md`
  SHA-256 `7d01dc5d5bb7355a293d3b8bd9249a84600ac6bc659f893027986a29d2f6c042`.
- review advisory: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/G5_MECHANICAL_REVIEW_ADVISORY_01.json`
  SHA-256 `8aa7b6e27f31983f8df97f252786aa518ec6528b541b28addf9b20a544ee3127`.
- canary receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/native-canary-01/runtime-output/receipt.json`
  SHA-256 `1fac4779c322a0970f31c32b3a1ed732946e3f704247797d4eaf3deb86da5215`.
- native bundle: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/native-canary-01/private-inputs.tar.gz`
  SHA-256 `af82651cec44d3878b91b7410c3b844bf890d779e0bceb305ac3d93a70ef9455`.
- series: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/series-r301.json`
  SHA-256 `0f08db01bf02164a1bea5a5b72a58632d9f6d607d1c5e6bd95574d49042f07df`.
