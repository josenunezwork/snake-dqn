# Erratum: LH-1 long-horizon screen amendment header (2026-10-08)

This erratum covers
`lh1-screen:docs/research/governance_amendment_long_horizon_screen_2026-10-04.md`. The document
lives on branch `lh1-screen` and is not on main.

**Erratum.** The document's opening says "Status: **DRAFT, proposed amendment** … **not in
force** until the research loop owner ratifies it". That opening is stale. The ratification
section at the end of the same document governs. **The amendment was ratified on 2026-10-05**
in commit `ecddfab` ("LH-1 amendment: ratified 2026-10-05 after calibration"). The adopted
rule set is C2 descriptive and C3 binding at mass >= 300. The calibration verdict was LH-1
ADOPTABLE.

**Why the header was not edited in place.** The ratified file is **hash-bound**. The LH-1
screen intent `lh1-screen:research/longh_screen/screens/frp3_s12.intent.json` binds its exact
bytes: `amendment_sha256` `85c3b851df162586610da4ae043f719aa00d3512bcc57f27b782924a4bd5e2fe`,
written by `research/longh_screen/lh1.py`. The calibration intent binds the earlier draft
(`0313acfe…`). Editing the file would break that binding, so the bytes stay as they are.

**No outcome changes.** The FRP-v3 LH-1 screen
(`/Users/josenunez/Projects/ml/snake-dqn-artifacts/lh1-screen-frp3-s12/run-v1/receipt.json`)
ran under the ratified amendment and recorded `CLEAR`. Nothing is relabelled. See
[research INDEX](INDEX.md) and [STATUS.md](../STATUS.md).
