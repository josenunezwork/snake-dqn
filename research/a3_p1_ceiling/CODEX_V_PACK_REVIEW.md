# Independent Codex-V actual-pack receipt

Verdict: **PASS — actual pinned A3 dispatch packs; no new findings.** Independent
read-only reviewer `/root/codex_v`; root transcribed the final review response.

Pack: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a3/dispatch-20261022-02`.
All four manifests pin `19a9d1e76f49b0f8dbd09ac2aa2eefd37c311909`, use the real
A3 shell, CUDA and at least 12 vCPUs. A3/Colab executables match the source review.
The pin is published on origin/main and the A3 branch.

Reviewer independently verified input hashes and matching canary/main inventories,
both empty allocation-bound ledgers, correct resume declarations and retry maximum,
bounded canary exemptions/family, all sidecars, and summed caps 240 min/40 u/
8 reserve. Qualified Colab tree: `553af3f692515b2bf66173356fe4d099acc96942`.
Both source/upload qualification receipts are `PASS_NOT_DISPATCHED`: normal
forecasts ~69.47 s <150 s, fallback bounds ~253.51 s <300 s.

| Manifest | SHA-256 |
|---|---|
| `a100/canary.json` | `8d9e33c412cbda7eb89906d901d4bae5ea8c3e0063cc3cba5b8f6669ea4037d4` |
| `a100/main-template.json` | `eea85589943b7f1386b560b25a8ef5678b65e5c020ab41af7118823460525206` |
| `l4/canary.json` | `93025d1077a3e382db1a99ccf2f6537ec5be04dd1515acb0e1d3f7a705833eba` |
| `l4/main-template.json` | `901a2517c13b672c331a706f83142932ae01dc1c3279cf616ff2f2dc79c8dfb1` |

Root reports both real supervisor canary validations exited 0; the unbound A100
main template refused on the expected missing SHA-bound canary receipt. No live
canary/main run or GPU performance result is asserted. Each target still needs
its genuine same-pin canary PASS, verified exports and backend release before
binding and dispatching its main assay. Reviewer edited no files and started no jobs.
