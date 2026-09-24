# Apex growth imitation: qualification stopped before training

The frozen study ended at `STOP_TRAIN_QUALIFICATION`. Its TRAIN teacher-input exclusion gate found 11 exact state-and-action-mask overlaps among 3,072 teacher rows against the pinned held/diagnostic inventory. The study therefore did not admit optimization. This result does not support or reject full-trajectory imitation learning: no student network was trained or evaluated.

All three planned student seeds—2026100201, 2026100202, and 2026100203—are `NOT_ADMITTED`, with zero scientific updates. The held bank was not materialized. No fixture forward, loss/fit curve, held evaluation, or learned-survival result exists. The original plan is closed: do not resample, filter, relabel, or train this bank, and do not backfill omitted stages.

## Qualification evidence

The native teacher and random qualification runs each covered 48 cases over 64 frames (3,072 frames per arm). The teacher collected food in all 48 games (150 total), reached first food in all 48, and met the joint and survival criteria in all 48. The random policy collected 13 food across 48 games, reached first food in 13, met the joint criterion in zero, and survived all 48. These checks establish that the qualification environments ran and that the teacher cleared its native behavior gate. The random survival result is not evidence of learned survival.

The joint criterion means collecting at least two foods and subsequently reaching a state with a legal safe boost action. It does not require taking a boost action; this teacher used only normal-speed actions.

The teacher rows contained 3,072 examples across the three phases: 96 before first food, 1,055 between first and second food, and 1,921 after second food. Their action counts were 109 for action 0, 2,766 for action 1, and 197 for action 2. The saved rows had no conflicting action labels for identical inputs. Neither this label consistency nor teacher performance establishes that a network can fit or reproduce the teacher. The decisive exclusion failure prevented any optimization from being admitted.

## Inventory boundary and limits

The frozen inventory covers exact saved vector61 inputs from the enumerated Apex qualification, held/diagnostic gameplay, and scientific-optimization artifacts in the 2026-09-13 study lineage. It records 180 source entries, 79,095 unique held/diagnostic inputs, and 146,575 unique scientific-optimization inputs. It is not an inventory of every historical vector61 run. Within this declared scope, 11 TRAIN teacher input-and-mask rows overlapped the pinned held/diagnostic inventory.

Some prior inputs are unavailable or inferred, and those limits bound the exclusion claim:

- The original two-frame records from the earlier qualification-v2 attempt were not saved raw. Their input coverage is inferred from the equivalent first two frames of saved teacher case 0, together with pinned source and control-flow evidence. This is an inferred prefix, not an original raw record.
- Discarded qualification optimizer probes did not uniformly preserve full replay-input sets. The optimization inventory represents completed scientific updates, not every discarded probe.
- Scientific replay digests enumerate accepted pre-action inputs; bootstrap next-state inputs are not separately enumerated as supervised inputs.
- Runs outside the enumerated lineage are not covered.

The audit passed by independently confirming the frozen stop decision and its saved counters. That validates the recorded qualification conclusion within the stated inventory boundary; it does not remove the boundary limits or create evidence for model learning.

## Resources and immutable record

Three natural supervisor receipts completed: `qualification-train`, `analysis`, and `audit`. Together they record 9.739288790966384 guarded seconds, a peak child RSS of 523,485,184 bytes, and a minimum available-memory observation of 36,098,670,592 bytes. Resource guards remained unchanged. No model forward or optimizer update occurred; native work was limited to the 96 qualification games and saved-artifact analysis/audit.

The receipt SHA-256 values are `qualification-train`: `58472ec4ad0c48d80583cf0fa420ec7262255c522a378be8fd9b3a2a42e0d892`, `analysis`: `44251c909585ae86c08271d3b83ee06e4af7ad095aea21b6d183e044f58d3a95`, and `audit`: `2151db24e52d5812671abef3d0c2499a9f600ab8c389a318d703d10cebf6e9f`.

The immutable record is under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-growth-imitation/`:

- Original design draft `apex-growth-imitation-design/design.json`: SHA-256 `72d119b964a25406913014f760373b8c713ab54565ffaebf87a878b2dbf2029e`.
- Prospective implementation clarification `design-v2.json`: SHA-256 `42d717b7935d2501cb84e468c1091e91d9911f0f2135a508cef604c2bc0957dc`.
- Frozen intent `intent.json`: SHA-256 `b39f8235d445be064abe3faf3493afd37ad4839c63d6647fa8741ad3cb1c0403`.
- Qualification report: SHA-256 `701d9281131c4706106abb03044e273dcb1bf4ab2b9d8c3c82d880bb7efc8cf4`.
- Independent audit report: SHA-256 `2e6342a470e8569b374ec99ea9a0c962df0a34900afe563834b7d4d1060b0ea`.
- Final closeout: SHA-256 `3911b896d1475c8bdc6281195cee16ecc97c69850d1b212075fedd2783752b67`.

The frozen runtime used analyzer SHA-256 `5800a9a7ef1280c1361fe957b340c793f439f8c186b6ba3a582dc10aca36d00c`. A later wording-only edit in the local analyzer source (beginning with hash prefix `1276`) was not copied into the frozen runtime and was not admitted as study evidence.

The concise saved summary is [summary.md](summary.md), and the existing representative qualification paths are copied in [representatives.json](representatives.json). Both files are byte-identical to the sealed analysis artifacts. There are no figures because no optimization or student-policy records were produced.
