# snake-dqn notes

**Summary:** A fact-only, cited, consistently written set of notes on the snake-dqn project, rebuilt from the project
documents, research records and judge rulings; every claim carries a source citation.
**As of:** 2026-10-25 PT · **Sources:** `docs/`, `AGENTS.md`, `research/*/README.md` and handoffs, and the big-lever
rulings under `snake-dqn-artifacts/big-lever/`; intermediate fact sheets are in `_facts/`.
**Related:** [STYLE](STYLE.md), [GLOSSARY](GLOSSARY.md)

## How to read these notes

- Each note opens with a one-sentence summary and an **As of** date; nothing after that date is asserted.
- Status words (`STRICT_PASS`, `KILL`, `HOLD`, `FUND_SYSTEMS`, `STOP_FUNDING`, ...) are the labels the sources use.
- Where sources disagree, the newer ruling, receipt or erratum wins in the body; both appear under
  "Superseded and conflicting records". Claims that could not be confirmed sit under "Unverified".
- Citations are `(src: path:line)` relative to the repository, `big-lever/...` for the artifacts tree, or `git <sha>`.
- The writing rules are in [STYLE.md](STYLE.md). The fact sheets in `_facts/` are the raw, cited inputs each note was
  written from; they are kept for audit and are not part of the reading order.

## Notes

| Note | One-line summary | Lines |
|---|---|---|
| [Project overview and serving state](overview-and-serving.md) | The project combines DQN-family snake agents with a browser product; the recorded release was bare `b8-s3` (SHA prefix `3e649783`) with mixed Watch opponents. | 244 |
| [Models and training history (June to early October 2026)](models-and-training-history.md) | This history covers DQN/Apex, PQN/raster, vetoes and FRP through the 2026-10-08 PT vector61 freeze, following the frp3-s12 (`eec144bf`) release with v8. | 188 |
| [Redesign and the B8 release (October 2026)](redesign-and-b8.md) | The October redesign produced B8 seed 3, released without a veto after T7 and serving qualification, while the tested late-death training arms closed without adoption. | 139 |
| [Governance and owner rules](governance-and-owner-rules.md) | T34 governance permits separately scored bare and assisted heroes under owner-ratified contracts, while release authority remains with the owner. | 209 |
| [T34 ExIt-B8 campaign (2026-10-19 to 10-21)](t34-campaign.md) | T34 tested search teaching and assisted play; Path B stopped, Claim A closed after D0 KILL, and Round 0 closed on integrity failure without an efficacy result. | 156 |
| [Big-lever advisor program](big-lever-program.md) | The advisor program revised its release plan into bounded Mac and A100 research lanes while retaining Mac qualification and owner release authority. | 197 |
| [Big-lever experiments and results](big-lever-experiments.md) | These experiments established an offline search ceiling, but L2′'s FUND_SYSTEMS result did not establish delivered serving value. | 202 |
| [F4 frozen-B8 residual dose study](f4-study.md) | F4's retention gain passed its funding bars, but D1024 failed the harm screen, producing terminal STOP_FUNDING and sealed D4096 discard; the post-mortem exporter was accepted conditionally. | 155 |
| [F0 hindsight label factory](label-factory.md) | F0 produced exact hindsight development labels; D256 and D1024 completed, while D4096 was sealed NOT_RUN_BY_RULE after funding stopped. | 157 |
| [Colab, RunPod and Mac compute](colab-and-compute-infra.md) | This note records compute operations and qualification history; new Colab work uses the Mac supervisor, while strict gates and serving qualification remain Mac-only. | 185 |
| [GPU engine and speed program](gpu-engine-and-speed.md) | A100 R2 (padding repair) passed its speed bar, but GPU label routes remained closed; T4A (batch-invariance test) certified only the pinned padded development configuration. | 131 |
| [Round 30, Round 32 and F5 successor work](r30-r32-f5-successors.md) | Round 30’s calibration route closed operationally with its main on HOLD, Round 32 completed synthetic rehearsal, and F5 adoption remained on HOLD. | 188 |
| [Incidents, errata and recorded lessons](lessons-and-incidents.md) | Round 0 (offline teacher study) closed on an integrity failure before outcomes were read; this record collects incidents, errata and the resulting recorded process rules. | 157 |
| [Glossary](GLOSSARY.md) | This glossary defines the models, studies, labels, metrics, and roles across the ten fact sheets; T34 uses paired worlds as its inference unit for fixed controllers. | 177 |

## Reading order for a new reader

1. [Project overview and serving state](overview-and-serving.md)
2. [Governance and owner rules](governance-and-owner-rules.md)
3. [Models and training history](models-and-training-history.md), then [Redesign and the B8 release](redesign-and-b8.md)
4. [T34 campaign](t34-campaign.md), then the [big-lever program](big-lever-program.md) and its
   [experiments](big-lever-experiments.md)
5. [F4 study](f4-study.md), [label factory](label-factory.md), [Round 30 / 32 / F5 successors](r30-r32-f5-successors.md)
6. [Colab, RunPod and Mac compute](colab-and-compute-infra.md), [GPU engine and speed](gpu-engine-and-speed.md)
7. [Incidents, errata and lessons](lessons-and-incidents.md), [Glossary](GLOSSARY.md)

## Provenance

Produced 2026-10-25 PT on branch `notes-cleanup-20261025`: 10 Opus fact-sheet agents (one per topic, read-only on all
sources), reviewed by a Fable coordinator (citation path/line checks, random sample verification), then 14 Astra writers
(one note each, re-verifying every citation against the sources before use), then a coordinator consistency pass.
