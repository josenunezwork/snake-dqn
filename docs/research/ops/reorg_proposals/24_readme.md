# Reorg proposal 24: root README.md

Scope 24 refreshed the root `README.md` on 2026-10-20 and shortened it from 264 lines to about 100.

## What changed
- The served agent section is corrected. It now says bare B8 s3 without a veto and a **mixed** Watch roster, and it adds the `SNAKE_SERVE_WATCH_ROSTER=all_b8` rollback (source: docs/STATUS.md "Released agent").
- A current-campaign pointer was added: T34 ExIt-B8, then HANDOFF_2026-10-20.md START HERE, then CAMPAIGN_LOG.md, then STATUS.md.
- Commands were checked against the Makefile targets (install, install-dev, web, test, test-fast, test-web, lint, format) and against web/README.md.
- A "Where things are" table was added. Some of its targets are being created by parallel agents and may not exist yet: docs/GLOSSARY.md, docs/TIMELINE.md, docs/architecture/{SRC,TESTS,WEB}_MAP.md, docs/architecture/REPO_TOP_LEVEL.md and docs/research/catalog/.
- Removed from the README:
  - Long strict-promotion and PQN-diagnostic command walkthroughs. These belong in the plan docs and the runbook.
  - The September repair-status narrative.
  - Old badges.
  - The "Apex is the incumbent" framing, which is stale because B8 is served and vector61 is frozen.

## Proposals (not applied)
1. **Move the removed walkthroughs.** Put the strict-promotion request/receipt command block and the PQN smoke/shakedown/screen block (from README at 6af08b0f) in `docs/evaluation.md` (new), and link it from docs/README.md.
2. **Check the parallel links after they land.** Once the other organizing agents push, confirm that every "Where things are" link resolves. If a path differs, for example if the maps live at the repo root, fix only the README links.
3. **Remove or regenerate `docs/architecture.svg`.** It predates B8/GridBatchSim and probably still shows the Apex-only architecture. This is unverified.
4. **Correct web/README.md.** Its install step `pip install fastapi "uvicorn[standard]" websockets` is redundant with `make install`. It also says the Raster panel shows `raster31v2`, but B8 uses ego2s. Its owner should verify both.
5. **Merge the two glossaries.** `docs/research/GLOSSARY.md` and the new `docs/GLOSSARY.md` could become one file with a redirect stub.
