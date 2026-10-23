# P1 Y3 origin release PLAN — option (b), 2026-10-23

Status: PLAN ONLY; explicit owner publication OK required. No release, tag, upload,
allocation, training, or dispatch was performed. Repository `josenunezwork/snake-dqn`;
proposed tag/release `p1-y3-origin-20261023`; target commit `03e0af75c64538f53a6971e6fec5d9a173bc921b`.
Transport authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_P1_REPAIR_TRANSPORT.md`,
SHA-256 `200f510ed1c4a040469af7d16eb2f3055b200166a7858dd6b975c16ac5d0c17c`.

Total **595,248,268 bytes** in **9 assets**.

| Asset | Bytes | SHA-256 | Scan result |
| --- | ---: | --- | --- |
| `p1-y3-origin-pause-snapshot-00000.pt` | 427,911,156 | `dbdc10c61914cccd156e4a73de67b8004cacce133ae334686d393153fe4109ce` | PASS_NO_SECRETS_WITH_PATH_OR_PII_DISCLOSURES |
| `immutable-ordinary_bank-e63802f2e97cbb5d.pt` | 144,091,565 | `a581990666ea5c17f3ee51cf354eb9213043f2f1899cf6b4f81b4a5a688f0136` | PASS_NO_SECRETS_OR_PII_FOUND |
| `immutable-refit_bank-f7fd77c1276d8df6.pt` | 15,110,573 | `7b9325445ad7fab12de58d119cc5c9300dceaa9419b0bf9526d779fa83655ed0` | PASS_NO_SECRETS_OR_PII_FOUND |
| `p1-inputs.tar.gz` | 2,517,008 | `7f2daf66a56d491ccc8b5676379142fee1dfdadf1c9abea1ae273d0a50ede795` | PASS_NO_SECRETS_WITH_PATH_OR_PII_DISCLOSURES |
| `late-body-rows-0000.npz` | 2,433,946 | `e5794210e7b236fba8ab191a47379f309af522241bf9bc4aed8f7b71fd7a12be` | PASS_NO_SECRETS_OR_PII_FOUND |
| `late-body-rows-0001.npz` | 1,442,768 | `a92bbb5b49e6f966e1cfa77e239be02d8e0f94268c2f95dc77c2eab185c72731` | PASS_NO_SECRETS_OR_PII_FOUND |
| `late-body-rows-0002.npz` | 1,093,629 | `c40429054110c5fc61c4c6655efff53f7f4413279d056738b2c4d24e94e8d124` | PASS_NO_SECRETS_OR_PII_FOUND |
| `failed-supervisor-receipt.json` | 384,986 | `aaa9f106c029b0235a174adbb38d617dad7c631af9dda49ce6fb01033f1db6eb` | PASS_NO_SECRETS_WITH_PATH_OR_PII_DISCLOSURES |
| `immutable-fixture_late_body_bank-64031cc19f35a73a.pt` | 262,637 | `64031cc19f35a73a0c72ddcf71d7ae3fd2a71aec96c50bec5ebccac4ea3b6522` | PASS_NO_SECRETS_OR_PII_FOUND |

Origin payloads are exact byte copies from the new pack's `origin-inputs/`,
verified equal to the prior `-02` payloads and SHA sidecars. The sole exception
is `p1-inputs.tar.gz`: this is the normal new-attempt control/empty-ledger bundle,
generated once under the existing attempt-ID rule and then staged byte-for-byte.
It is selected in descending input size because even moving every origin input
leaves about 154 seconds of qualified fallback upload when the bundle stays local.
No origin snapshot, bank, receipt, row, or evidence artifact was rebuilt or recompressed.
The attempt's bundle changes source pin, diff witness and namespace as required.
All unchanged runtime destinations are recorded in the manifests.

Selection proof: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-plan.json` (SHA-256
`e30f3909e6ba008780cbd10e594fea6f0c0077d07700b01c2fa103a7c2c325f5`). It moves local inputs in descending size
until the existing qualified forecaster, including sparse source fallback,
passes 150 seconds. Residual inputs and sidecars stay on ContentsClient.
The last step forecasts **146.921533 seconds**;
the immediately preceding step is **154.825506 seconds**.
Full source-preparation verification is `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/upload-preflight.json`.

## Secrets and PII attestation

Every asset was scanned by the pinned scanner, with archive traversal, parser-only
pickle metadata inspection, JSON config/env traversal and NPY header/string scans.
Numeric tensor/array storage was classified separately. **No secret, token, key,
OAuth credential or keychain material was found; no metadata parse failures.**
This is a recorded scanner result, not a substitute for owner publication consent.

The scan found absolute local paths containing the local username, `/content`
runtime paths and an assignment/endpoint identifier in the failed receipt.
These remain unchanged in the original bytes. The owner's approval must cover
these disclosures as well as all asset names and bytes. Per-asset exact absolute
paths and any email identifiers are listed in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/public-path-disclosures.json`;
all locations/method counts/results are in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-scan.json`
(SHA-256 `b62418309a7f62bc24539905ec5093015f8333eb1d857782af9d0415354aeb7d`). No matched secret value is emitted.
Publication is blocked on any secret finding or unsupported metadata.

## Claude's exact publication commands — only after explicit owner OK

Do not run these during preparation. Never add `--clobber`, delete/replace an
asset, or reuse the tag for changed bytes. If interrupted, inspect remote state;
do not automatically retry publication.

```bash
gh release create p1-y3-origin-20261023 --repo josenunezwork/snake-dqn --target 03e0af75c64538f53a6971e6fec5d9a173bc921b --draft --latest=false --title 'P1 Y3 immutable origin inputs (2026-10-23)' --notes-file /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/RELEASE_NOTES.txt

gh release upload p1-y3-origin-20261023 --repo josenunezwork/snake-dqn \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/p1-y3-origin-pause-snapshot-00000.pt \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/immutable-ordinary_bank-e63802f2e97cbb5d.pt \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/immutable-refit_bank-f7fd77c1276d8df6.pt \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/p1-inputs.tar.gz \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/late-body-rows-0000.npz \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/late-body-rows-0001.npz \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/late-body-rows-0002.npz \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/failed-supervisor-receipt.json \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/release-assets/immutable-fixture_late_body_bank-64031cc19f35a73a.pt

gh release edit p1-y3-origin-20261023 --repo josenunezwork/snake-dqn --draft=false --latest=false
```

After publication, verify every public asset through the qualified Mac preflight:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_repair_transport preflight --manifest /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/canary-manifest.json --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-03/public-fetch-preflight-after-owner-ok.json
```

Require PASS for all rows with exact size/SHA and total <=300 seconds. This creates
a new immutable measured receipt with timing/rate; it fills the measured-public
line beside `measured-forecast-03.json`, whose prepublication values remain null.
Failure is HOLD and back to a judge; no cap is raised. The real canary then fetches
these same release files on the VM and must pass its separate <=300-second fetch,
<=150-second residual transfer and <=300-second staging bars, plus exports/release.

The release stays immutable until P1 closes. Replacement and 1M need fresh
receipt-bound materialization and, if public exports are needed, their own new
release tags and explicit owner publication scope; this plan grants none.
