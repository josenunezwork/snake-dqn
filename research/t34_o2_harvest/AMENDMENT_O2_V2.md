# O2 prospective resize amendment — 2026-10-21, pre-data

Owner instruction: prospectively resize the underpowered unused N96 O2 bank,
use measured T23 SDs, require >=.80 joint mass-superiority power at +40/+10,
family-wise null-pass <=.05, and if required N>1000 cap at the largest balanced
N fitting <=150 units. Prepare <=8-hour single-runtime Z-9 jobs; validate only;
no Colab. The explicit instruction authorizes preparation and publication.

This sibling amendment and `REGISTRATION_O2_V2.md` supersede only v1's sample
size/bank extension, variance-planning source, multiplicity alpha, compute caps,
source pin, manifest set and corresponding execution/analysis plumbing. All
original v1 bytes remain unchanged. No O2 selection trajectory exists, and no
outcome informed this amendment. Excluded old preflight fixtures provide only
operational planning context. T23 historical SDs provide variance planning.

The uncapped requirement is 5097 worlds (1699/mix), exceeding 1000; `OC_V2.json` reports its minimum
and preceding-N failure. The requested budget fallback registers N1260,
420/mix, <=149.733333 units, ten jobs <=462 minutes, with 20% reserve.
`plan_v2.py` provides an exact integer-minute capacity proof. Capped joint power
is reported without an adequate-power claim. All original 96 unused selection
rows are retained; 1164 added rows and nine new excluded fixtures are audited.
Entire added allocations for rounds 0–4 were rejected for collisions;
all banks/reasons are preserved. Each consecutive round regenerates every
added identity and fixture. Round 5 is collision-clear (1173 added/excluded
identities, plus refreshed published-bank exclusion for all 96 inherited
worlds). Rounds 3–6 shared one batch collision scan; the first clear whole round
was selected, without per-world substitution or any outcome data.

The contract's observed mean >=40 point bar is retained. It prevents .80
full-GO power at a true +40 boundary for any N under symmetric continuous
error. The .80 sizing calculation therefore concerns both mass-superiority
hypotheses, with survival/boost/point/fault clauses retained as additional GO
requirements. No threshold was silently relaxed to claim adequate power.
Holm alpha is tightened prospectively from .10 to .05.

No protected package, released checkpoint, serving pin, active tmux session or
sealed Round 0 artifact directory is modified or opened. Colab supervisor code
is imported read-only. Old V1 manifests remain evidence but are superseded for
future O2 use; the artifact-root `DISPATCH.txt` points only to the V2 manifest set
and explicitly records NOT DISPATCHED.
