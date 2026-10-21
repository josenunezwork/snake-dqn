# O2 upload qualification supplement — 2026-10-21

Outcome: **PASS_NOT_DISPATCHED** for all three existing manifests. This is a
standalone Mac-side admission check introduced by concurrent main work and
merged at `bdf27aa09e37e2cafa0c5494b5af703965b5e4aa`.

The executable pin remains `f1ca784e979a5c162e869f8cf670b5dc68ad8b32`.
The registration, bank, controller, 251-file frozen closure, three manifest
files, prior `PREFLIGHT.md`, and all earlier artifact bytes are unchanged.
The source preparer adds its four standard transport/dependency files to make
the actual 255-file sparse source delivery.

Using `research.t34_o2_harvest.upload_readiness --full`, each manifest passed
the shared `research/bl_upload` qualifier. It measures the real pinned sparse
source fallback locally, checks exact input bytes/hashes, and forecasts upload
against saved goodput. It neither allocates nor invokes Colab.

| Property | All three chunks |
|---|---:|
| Normal initial upload | 142,102 bytes |
| Normal upload forecast | 72.1944 seconds |
| Qualified normal budget | 150 seconds |
| Compressed source fallback | 1,039,804 bytes |
| Fallback including fetch bound | 217.5688 seconds |
| Supervisor hard initial-upload cap | 300 seconds |

The forecast uses the saved slowest completed large-upload rate of
44,484.787 bytes/s, a 65-second setup allowance and per-input overhead. It is
a prospective bound, not a measurement of a new connection. The runtime still
downloads and hash-verifies the six public checkpoint inputs from its catalog.

Additional targeted tests: **9 passed in 21.62 seconds**, covering the two O2
upload-readiness tests and seven shared qualification tests. The original
56 core O2 tests and 130-test targeted transport run remain the prior evidence.

All receipts are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o2-20261021/` with SHA-256
sidecars and completion markers:

| Receipt | SHA-256 |
|---|---|
| `upload-qualification-00.json` | `7aba40c27a9a81ef96f63c3cf2c7217e5ae977e4f555dd8bd1d57cb81806f995` |
| `upload-qualification-01.json` | `d8c7f4362c09b20f4575b43c9c21bd86dae6bb8ff81f2df70d51b45b99c59b25` |
| `upload-qualification-02.json` | `921ee9609b8c4eed2f15b1c21d22a4ea7d262ad085436658f70e16c9e6f6e6a7` |
| `delivery-upload-supplement.json` | `9cdb1ed5f2b9c740cf5cde5147333f3086ebd1f1156572595f9797913dfad3f0` |

`freshness-final-merge.json` checks the next 58 incoming changed files and
refreshes published bank inventories through `bdf27aa0`: CLEAR. The sealed
Round 0 artifact root was not traversed. No selection trajectory was run.

Read these qualifications alongside the unchanged `DISPATCH.txt`. Any future
manifest edit, including an assignment-inventory refresh, needs a new local
validation and fresh upload qualification receipt. No dispatch, study funding,
release or serving authority is conferred by this supplement.
