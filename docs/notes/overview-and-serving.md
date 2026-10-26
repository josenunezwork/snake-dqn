# Project overview and serving state

**Summary:** The project combines DQN-family snake agents with a browser product; the recorded release was bare `b8-s3` (SHA prefix `3e649783`) with mixed Watch opponents. (src: README.md:3-7; docs/STATUS.md:29-36)
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/overview-and-serving.md`; primary records cited inline.
**Related:** [Index](README.md), [GLOSSARY](GLOSSARY.md), [Models and training history](models-and-training-history.md), [Redesign and B8](redesign-and-b8.md), [Governance and owner rules](governance-and-owner-rules.md), [T34 campaign](t34-campaign.md), [Big-lever program](big-lever-program.md)

## What the project is

Snake DQN is a reinforcement-learning project for a multi-snake, slither.io-style arena using DQN-family training methods. (src: README.md:3-4)

The documented product used FastAPI and React to stream server-side games for watching trained agents or playing against them. (src: README.md:4-6)

The platform used PyTorch for models and training, and the FastAPI + React/TypeScript web UI replaced the retired PyQt5 desktop GUI. (src: AGENTS.md:124-127)

The repository documented three separate model stacks. (src: AGENTS.md:129-135)

- Apex used distributed DQN with prioritized replay, a target network and n-step returns, a feedforward `ApexNetwork`, `vector61` observations and Python `GameState`. (src: AGENTS.md:133)
- B8 used a distilled DQN / dueling CNN, `Ego2sNet` with approximately 1.64 million parameters, the cell-exact numba `GridBatchSim`, and ego2s-b raster observations plus 60 scalars. (src: AGENTS.md:134)
- Raster/PQN used synchronous PQN Q(λ), `RasterDuelingNetwork`, `raster31v2` observations and NumPy `BatchSim`, without replay, a target network or prioritized replay. (src: AGENTS.md:135)

B8 was the ego2s CNN rather than the Apex MLP; the Raster/PQN stack remained unpromoted in the cited record. (src: AGENTS.md:129; AGENTS.md:137-140)

The README named T34 ExIt-B8 (exact-simulator search teaching plus DAgger into B8) as the active campaign. (src: README.md:33-36)

`src/scripts/tournament_eval.py` provided diagnostic evaluation and a separate live-only Mac strict-promotion mode whose passing receipt did not change the served default. (src: README.md:85-89)

## Repository layout

The README mapped documentation and research entry points as follows. (src: README.md:97-112)

| Area | Documented entry point | Source |
|---|---|---|
| Documentation | `docs/README.md`, `docs/GLOSSARY.md`, `docs/TIMELINE.md` | (src: README.md:99-101) |
| Research packages | `research/INDEX.md` | (src: README.md:102) |
| Specifications and advice | `docs/research/specs/INDEX.md`, `docs/research/advice/INDEX.md` | (src: README.md:103-104) |
| Tests | `docs/architecture/TESTS_MAP.md` | (src: README.md:107) |
| Web app | `docs/architecture/WEB_MAP.md` | (src: README.md:108) |
| Working rules | `AGENTS.md` | (src: README.md:110) |

B8 training lineage was documented in `research/redesign_m4*` and `research/redesign_t7_phase_r_20261016`, with T34 work in `research/t34_*`. (src: README.md:82-84)

The 2026-10-20 PT test catalog at commit `3d57df7f` recorded 556 entries, including 545 `test_*.py` files, support modules, helpers and fixtures in a flat layout. (src: docs/architecture/TESTS_MAP.md:3-11)

`src/model/inference_agent.py` supplied a forward-only `InferenceAgent` that dispatched on `obs_spec` for Apex and raster evaluation/serving. (src: AGENTS.md:467)

`docs/ml_algorithm.md` scoped its algorithm description to the Apex 58/61-dimensional vector stack. (src: docs/ml_algorithm.md:9-15)

The project virtual environment is required for Python commands, and the README documented `make web` as building the frontend and serving on `http://localhost:8000`. (src: README.md:45-55)

The README documented `make test`, `make test-fast`, `make test-web`, `make lint` and `make format`, and prohibited running the suite while a governed study occupied the Mac CPU slots. (src: README.md:64-73)

## Web product (modes, protocol, leaderboard)

The 2026-10-20 PT web map described a server-authoritative Python engine with a React view over a WebSocket. (src: docs/architecture/WEB_MAP.md:3-5)

`web/serve.py` bound `127.0.0.1`, pinned one Torch thread and ran `web.backend.app:app` under uvicorn, with port 8000 by default and a `PORT` override. (src: docs/architecture/WEB_MAP.md:10-20; docs/architecture/WEB_MAP.md:77)

The frontend development server used port 5173 and proxied `/api` and `/ws` to port 8000. (src: docs/architecture/WEB_MAP.md:13-14)

`GameSession` owned one game state and selected policy, supported Watch / Play / Train, and resolved checkpoint, roster, veto, Path B (T34 staged assistance) and speed-ceiling configuration. (src: docs/architecture/WEB_MAP.md:27; docs/architecture/WEB_MAP.md:34)

For B8, Train requests were refused and returned to Watch with `policy_training=false`, epsilon zero and a forward-only message. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:54)

The wire contract used frame DTOs from `serialize.py`, `PROTOCOL_VERSION = 2`, and `/ws/stream`. (src: docs/architecture/WEB_MAP.md:28; docs/architecture/WEB_MAP.md:48-52)

Client controls included play/pause, reset/new game, speed/mode/epsilon/hero/food selection, checkpoint loading, human movement/boost, Play opponents, weight saving and raster-stream selection. (src: docs/architecture/WEB_MAP.md:56-61)

Malformed controls set `session.last_error` without raising, while another connection owning a scored Play run prevented mode/reset/new-game/checkpoint changes and foreign steering. (src: docs/architecture/WEB_MAP.md:61-63)

REST endpoints covered health, checkpoints, metrics, state, leaderboard, recent scores and player lookup; score submission used server-authoritative `POST /api/scores`. (src: docs/architecture/WEB_MAP.md:43-46)

The human leaderboard used SQLite `ScoreStore` in `src/data/score_store.py` and `scores.db`, separate from replay storage, with a `SNAKE_SCORES_DB` override. (src: AGENTS.md:477; docs/architecture/WEB_MAP.md:45-46)

The B8 release made Play opponents heavier, so leaderboard rows before and after 2026-10-17 PT were not comparable; scored runs recorded the served checkpoint name. (src: docs/research/b8_s3_release_2026-10-17.md:71-72)

The frontend kept the arena visible alongside Controls, Play, Inspector, Raster, Network and Dashboard tabs. (src: docs/architecture/WEB_MAP.md:84-85)

The web map recorded Path B as default OFF, eligible only for the released mixed Watch6 B8 hero on macOS through `SNAKE_SERVE_PATHB`. (src: docs/architecture/WEB_MAP.md:34; docs/architecture/WEB_MAP.md:74)

The same map recorded `SNAKE_MECHANICS_V2` as ON by default, with `0/false/no` selecting legacy v1 mechanics. (src: docs/architecture/WEB_MAP.md:75)

## Released agent (b8-s3, roster, rollback env vars, fps ceilings)

The 2026-10-17 PT release recorded `RELEASE` for bare `b8-s3` (B8 seed 3), replacing `frp3-s12` (SHA prefix `eec144bf`) plus v8. (src: docs/research/b8_s3_release_2026-10-17.md:3-6; docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:28-29)

The release claim was more mass at equal survival against the previously served agent with no veto, not greater safety; late-game deaths remained at parity. (src: docs/research/b8_s3_release_2026-10-17.md:3-6)

`CHECKPOINT_RELEASED_DEFAULT = NAME_B8_S3` selected checkpoint SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, with the gitignored copy at `saved_snakes/b8_s3_t7_20261016.pth`. (src: docs/research/b8_s3_release_2026-10-17.md:30-34)

B8 used `Ego2sServingPolicy` and `ego2s-bv8`, with `veto_variant: none` in its strict pin. (src: docs/research/b8_s3_release_2026-10-17.md:14; docs/research/b8_s3_release_2026-10-17.md:38-40)

B8 consumed v8-derived training labels and live input features, but had no serving veto wrapper. (src: docs/research/b8_s3_release_2026-10-17.md:56-57)

Pin or byte-check refusal fell back to verified `frp3-s12` plus v8, then `champion` (SHA prefix `43d4e2c5`), with a recorded reason. (src: docs/research/b8_s3_release_2026-10-17.md:35-36; docs/research/serving_time_search_feasibility_2026-10-03.md:64)

The later owner-approved roster release placed the bare B8 hero in slot 0 against `frp3-s12` plus v8 opponents in slots 1 through N−1. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:3-7)

Unset or blank `SNAKE_SERVE_WATCH_ROSTER` selected mixed, invalid values selected mixed with a visible reason, and opponent pin/wrapper refusal restored all-B8 without a partial mixed roster. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:18-21)

The B8 hero never appeared in `wrapped_ids` and had no serving veto counter. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:23-25)

The roster switch changed neither Play behavior nor checkpoint bytes and introduced no additional leaderboard-comparability change. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:37-40)

### Rollback environment variables

Environment selection was read per session build and required a server restart to apply; explicit `GameSession(checkpoint=...)` and client `load_checkpoint` bypassed the registry. (src: docs/architecture/WEB_MAP.md:65; docs/architecture/WEB_MAP.md:79-80)

| Selection | Recorded effect | Source |
|---|---|---|
| `SNAKE_SERVE_WATCH_ROSTER=all_b8` | Restored all-B8 Watch; unsetting restored mixed. | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:31-37) |
| `SNAKE_SERVE_CHECKPOINT=frp3-s12` | Restored `frp3-s12` + v8 with remaining serving variables at defaults. | (src: docs/research/b8_s3_release_2026-10-17.md:42-50) |
| `SNAKE_SERVE_CHECKPOINT=champion` | Restored `champion` + v8 with remaining serving variables at defaults. | (src: docs/research/b8_s3_release_2026-10-17.md:42-50) |
| `SNAKE_SERVE_VETO_VARIANT=v7`, `v5` or `v2` | With no checkpoint named, selected `champion` + that variant; explicit `frp3-s12` with an older variant remained unwrapped. | (src: docs/research/b8_s3_release_2026-10-17.md:50-52) |
| `SNAKE_SERVE_VETO_WATCH_HERO=0/false/no/off` | Disabled the Watch-hero veto switch, documented as ON by default. | (src: docs/architecture/WEB_MAP.md:71) |
| `SNAKE_SERVE_VETO_PLAY_AI=1/true/yes/on` | Enabled v2 wrapping of Play AI, documented as OFF by default. | (src: docs/architecture/WEB_MAP.md:72) |

### Frame-rate ceilings

The ego2s ceiling was 24 fps through six driven AI, 12 fps for seven through twelve, and `max(1, floor(144/n))` above twelve, excluding the Play human and including respawnable AI. (src: docs/research/b8_s3_release_2026-10-17.md:65-67)

The server clamped `set_speed`, recomputed ceilings after roster/mode/checkpoint rebuilds and exposed `speed_ceiling` to the UI controls. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:233-238)

Other checkpoints retained a 120 fps server ceiling and the existing 60 fps UI range. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:238-239)

These ceilings were processing budgets rather than achieved-cadence promises because the server slept a full frame interval after processing. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:91-92)

## Monitoring thresholds

The release window requires at least seven days from the first retained valid Watch tick and 100,000 actual hero decisions for each roster mode and size, whichever ends later. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:77-82)

All-B8 history must not be pooled into mixed windows or used to supply another mode/size baseline. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:57-59; docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:77-80)

Any served fallback reason, including refused mixed opponent pins/wrappers, or any ego2s-path exception remains a rollback trigger. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:83-84)

| Watch roster | Driven AI | Baseline hero deaths / 1,000 decisions | Rollback rate | Source |
|---|---:|---:|---:|---|
| mixed | 6 | 0.272 | > 0.544 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:51) |
| all_b8 | 6 | 0.424 | > 0.848 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:52) |
| all_b8 | 8 | 0.496 | > 0.992 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:53) |
| all_b8 | 12 | 0.768 | > 1.536 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:54) |

Equality does not trigger rollback, and mixed Watch8/12 report `NOT_HEALTHY` for missing registered death baselines even though their latency was qualified. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:56-59)

Latency rollback uses default-speed `GameState.update` p95 above 1.5× the matched qualification value, excluding the first 200 updates after construction/rebuild/reset and using eligible 12 fps samples only. (src: docs/research/b8_s3_release_2026-10-17.md:87; docs/research/b8_s3_release_2026-10-17.md:115-117)

| Case | Driven AI | Update p95 rollback above (ms, as recorded) | Source |
|---|---:|---:|---|
| Mixed Watch6 | 6 | 11.688329 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:66) |
| Mixed Watch8 | 8 | 14.278734 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:67) |
| Mixed Watch12 | 12 | 19.712000 | (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:68) |
| All-B8 Watch6 | 6 | 12.072490 | (src: docs/research/b8_s3_release_2026-10-17.md:123) |
| All-B8 Watch12 | 12 | 23.125516 | (src: docs/research/b8_s3_release_2026-10-17.md:124) |
| Play6 | 5 | 8.684793 | (src: docs/research/b8_s3_release_2026-10-17.md:125) |
| Play12 | 11 | 17.589807 | (src: docs/research/b8_s3_release_2026-10-17.md:126) |

All-B8 Watch8 had no qualified latency baseline; its death baseline did not qualify latency. (src: docs/research/b8_s3_monitoring_erratum_2026-10-17.md:38-43)

The server emitted `b8-release-monitor:` JSON telemetry to stderr, consumed by `scripts/monitor_b8_release.py`; exit 0 meant available measured metrics were within bounds, while exit 2 meant a trigger or incomplete/invalid telemetry. (src: docs/research/b8_s3_release_2026-10-17.md:92-111)

The monitor reported window completion separately and never treated missing data as healthy. (src: docs/research/b8_s3_release_2026-10-17.md:107-111)

Late-death complaints require review without automatic rollback. (src: docs/research/b8_s3_release_2026-10-17.md:89-90)

The records named an owner check-in on 2026-10-24 PT and required incomplete windows to continue afterward. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:80-82)

Neither the original release nor the mixed-roster release task installed a persistent monitoring job. (src: docs/research/b8_s3_release_2026-10-17.md:130-132; docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:96-98)

## Serving qualification chain (T12/T12B/T12C, T13/T13b, T15/T15B)

T7 (fresh-bank B8 skill evaluation) supplied `STRICT_PASS` and Tier 2 `PASS`, while T12/T12B (Mac serving qualification and its check-4 repair) supplied serving-correctness evidence. (src: docs/research/b8_s3_release_2026-10-17.md:12-17; docs/research/b8_s3_release_2026-10-17.md:25-26)

The Tier 2 scripted episode-survival difference versus champion + v8 was −0.015625 with SE 0.022947 against a threshold of ≥ −0.05, leaving a disclosed narrow margin of 0.034375. (src: docs/research/b8_s3_release_2026-10-17.md:13; docs/research/b8_s3_release_2026-10-17.md:58-59)

All following qualification and release steps were recorded on 2026-10-17 PT. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:1; docs/research/t13_product_roster_2026-10-17.md:1; docs/research/t13b_six_snake_roster_2026-10-17_results.md:1; docs/research/t15_mixed_roster_qualification_2026-10-17.md:1; docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:1)

- T12 (extended Mac serving qualification) returned `SERVING_FAIL` / `HOLD`: checks 1–3 and 5–8 passed, while check 4 failed latency/UI-speed validation. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:7-11; docs/research/t12_b8_serving_qualification_2026-10-17.md:41-54)
- Its eight checks covered live-observation identity, absent serving veto, registry/pin/rollback, latency/ceilings, Watch stability, Play, UI/confidence and Train refusal. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:45-54)
- Watch12 failed with a 55.9117 ms tick against the 41.6667 ms budget at 24 fps and a 60.94329 ms tick against 58.82353 ms on the sole allowed 17 fps retry. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:78-81)
- T12B (check-4-only repair) returned `SERVING_PASS` with zero overruns across eight 120-tick probes at default 12 fps and enforced Watch6/Play6 ceilings of 24 fps and Watch12/Play12 ceilings of 12 fps. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:265-276)
- T12B measured default/ceiling cadence of 9.499/16.025 fps for Watch6, 8.788/8.855 for Watch12, 9.606/15.827 for Play6 and 8.942/8.968 for Play12, and did not certify real-time 12/24 fps delivery. (src: docs/research/t12_b8_serving_qualification_2026-10-17.md:285-289)
- T12C (checkpoint release) made B8 seed 3 the default and superseded the earlier HOLD/not-yet-released wording after live default and rollback verification. (src: docs/STATUS.md:161-165)
- T13 (twelve-snake product-roster measurement) was descriptive rather than a gate, using 25 paired episodes of 5,000 frames per roster. (src: docs/research/t13_product_roster_2026-10-17.md:3; docs/research/t13_product_roster_2026-10-17.md:17)
- T13 measured hero deaths of 0.768 per 1,000 decisions for all-B8 versus 0.368 for all-FRP3 with a v8 hero, and hero H5000 mass of 121.51 versus 239.73. (src: docs/research/t13_product_roster_2026-10-17.md:21-23)
- T13 mixed B8 hero / FRP3 + v8 opponents measured 0.624 hero deaths per 1,000 decisions and H5000 mass 190.67. (src: docs/research/t13_product_roster_2026-10-17.md:27)
- T13b (six-snake roster decision) required both an all-B8 versus mixed death-rate gap within 0.10 per 1,000 decisions and an all-B8 versus FRP3 hero p95-length gap within 100 cells. (src: docs/research/t13b_six_snake_roster_2026-10-17_results.md:7-9)
- The death gap was 0.096 per 1,000 decisions, but the p95-length gap was 127 cells, producing `PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`. (src: docs/research/t13b_six_snake_roster_2026-10-17_results.md:11-14)
- T15 (mixed Watch qualification) returned `SERVING_PASS` for all six checks: identity, opponent-only veto scope, selection/fallback/Play regression, latency/ceilings, continuous Watch stability and live UI. (src: docs/research/t15_mixed_roster_qualification_2026-10-17.md:3; docs/research/t15_mixed_roster_qualification_2026-10-17.md:66-73)
- T15 recorded zero overruns at 12 fps and enforced 24/12/12 fps ceilings for mixed Watch6/8/12. (src: docs/research/t15_mixed_roster_qualification_2026-10-17.md:103-116)
- Its continuous mixed Watch6 baseline was 34 deaths over 125,000 hero decisions, or 0.272 deaths per 1,000 decisions. (src: docs/research/t15_mixed_roster_qualification_2026-10-17.md:132-134)
- T15 retained an `INVALID` original UI envelope and applied its final verdict only to the corrected, independently audited assembly. (src: docs/research/t15_mixed_roster_qualification_2026-10-17.md:162-168)
- T15B (owner-approved roster release) switched the Watch default to mixed after T15 qualification and the T13b rule. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:3-7)

T15 supplied no new promotion evidence and did not isolate the opponent checkpoint effect from the opponent-wrapper effect. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:42-45)

## Earlier served agents (dated)

`champion` was the `champion_a5_freespace_20260621.pth` Apex checkpoint with a 61 → 512 → 256 dueling network and six direction/speed actions. (src: docs/research/serving_time_search_feasibility_2026-10-03.md:64)

| Release date | Recorded Watch-hero default | Source |
|---|---|---|
| 2026-10-01 PT | `champion` + v2 free-space veto, release commit `95e0db9`. | (src: docs/experiments/apex_veto_serving_and_v3_2026-10-01/README.md:66-74) |
| 2026-10-02 PT | `champion` + v5, commit `03abe8e`, recorded on local main and not pushed. | (src: docs/experiments/apex_veto_v5_2026-10-01/README.md:82-90) |
| 2026-10-03 PT | `champion` + v7 space preference, λ = 4. | (src: docs/experiments/apex_veto_v7_2026-10-02/README.md:135-138) |
| 2026-10-03 PT | `champion` + v8 space-and-head veto, λ = 8, after `STRICT_PASS` and `SERVING_PASS`. | (src: docs/experiments/apex_veto_v8_2026-10-03/README.md:165-172) |
| 2026-10-06 PT | `frp3-s12` + v8 replaced champion + v8 after `STRICT_PASS` and `SERVING_PASS`. | (src: docs/research/frp3_serving_result_and_release_2026-10-06.md:3-13) |

The v2 wrapper used a one-step spaciousness test; v5 added a boost-landing check; v7 added space preference at λ = 4; v8 used λ = 8 plus opponent-head reranking. (src: docs/research/serving_time_search_feasibility_2026-10-03.md:67-70)

The FRP3 serving qualification recorded 125,000 Watch decisions, 305 vetoes and 18 hero deaths, with mean wrapper cost 0.61 ms per decision. (src: docs/research/frp3_serving_result_and_release_2026-10-06.md:22-26)

## Serving rules (serve on this Mac, W<=4/K<=8 clause)

Serving stays on the owner's Mac under the 2026-10-21 00:30 PT decision; GPU-host serving is excluded, and Colab GPUs are for training, labels and evaluation. (src: big-lever/LIVE_FACTS.md:13)

That record identified the Mac as an Apple M5 Pro with 18 CPU cores and an Apple GPU through MPS/Metal, without CUDA. (src: big-lever/LIVE_FACTS.md:13)

The 2026-10-21 03:45 PT owner clause permits served search with W ≤ 4 processes and K ≤ 8 outstanding on this Mac, with the §4 timing bars unchanged. (src: big-lever/LIVE_FACTS.md:21)

Strict gates and serving qualification remain Mac-only under the campaign authority. (src: docs/STATUS.md:20; docs/STATUS.md:90-96)

Release and serving changes still require the owner's approval; a judge may only recommend release. (src: docs/STATUS.md:24)

The 2026-10-21 14:30 PT L1′ diagnostic (Mac search-latency probe) was `NOT_READY`, with approximately 19.7–27.2 s CPU and 32–41 s MPS per-root search times, about 15–25× over 0.83–1.458 s ceilings. (src: big-lever/LIVE_FACTS.md:24)

The 2026-10-22 23:40 PT M1 (serving probe) record reported `FAIL` → `KEEP_B8` and described the bare served game as “overran 43x,” with update p95 13.2 ms. (src: big-lever/LIVE_FACTS.md:59)

The 2026-10-23 05:05 PT GC A1 probe (garbage-collection freeze) recorded `FAIL`, “4x more gen-2 collections” and update p95 13.8–23.3 ms, with no serving change and `KEEP_B8`. (src: big-lever/LIVE_FACTS.md:65)

## Superseded and conflicting records

- The B8 release superseded T12/T12B statements that `frp3-s12` remained the default under HOLD; those qualification records retained their historical evidence. (src: docs/STATUS.md:161-165; docs/STATUS.md:145-159)
- The owner-approved mixed default superseded all-B8 wording in the original B8 release, T15 opt-in qualification and monitoring erratum. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:3-14; docs/research/b8_s3_release_2026-10-17.md:60; docs/research/t15_mixed_roster_qualification_2026-10-17.md:4-6; docs/research/b8_s3_monitoring_erratum_2026-10-17.md:47-50)
- The per-roster monitoring errata restricted the original 0.848-deaths-per-1,000 trigger to all-B8 Watch6 and supplied separate size/mode thresholds. (src: docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:49-59; docs/research/b8_s3_monitoring_erratum_2026-10-17.md:10-22; docs/research/b8_s3_release_2026-10-17.md:86)
- The retained all-B8 Watch6 baseline was T12's 0.424 deaths per 1,000 decisions, not T13b's fresh descriptive rate of 0.392. (src: docs/research/b8_s3_monitoring_erratum_2026-10-17.md:16-22; docs/research/t13b_six_snake_roster_2026-10-17_results.md:11)
- The released T12B-based all-B8 Watch6 latency trigger was 12.072490 ms, superseding the T12 draft value of 12.596509 ms. (src: docs/research/b8_s3_release_2026-10-17.md:123; docs/research/t12_b8_serving_qualification_2026-10-17.md:184-185)
- `docs/STATUS.md` recorded B8 as released, superseding the older AGENTS summary naming FRP3 and conflicting with its retained wording that Apex was the incumbent champion. (src: docs/STATUS.md:29; AGENTS.md:23-30; AGENTS.md:137)
- FRP3's 2026-10-06 PT checkpoint release superseded champion + v8 as the served default. (src: docs/research/frp3_serving_result_and_release_2026-10-06.md:3-5; docs/experiments/apex_veto_v8_2026-10-03/README.md:165-168)

## Unverified

- Completion or outcome of the 2026-10-24 PT owner check-in and seven-day monitoring windows was not established by the cited records; they specified a check-in and continuation rule rather than a completed result. (src: docs/STATUS.md:47-55; docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md:77-82)
- A complete sourced description of every top-level directory remained unavailable: the README linked `docs/architecture/SRC_MAP.md` and `docs/architecture/REPO_TOP_LEVEL.md`, but those source files were absent during verification. (src: README.md:106-109)
- Exact WebSocket client limits and multi-viewer behavior beyond the shared session and scored-Play control restrictions were not established by the cited web map. (src: docs/architecture/WEB_MAP.md:26; docs/architecture/WEB_MAP.md:61-63)
- A later live production configuration, including whether Path B was enabled, was not verified; the sources described the released selection and Path B's documented OFF default. (src: docs/STATUS.md:29-36; docs/architecture/WEB_MAP.md:34; docs/architecture/WEB_MAP.md:74)
