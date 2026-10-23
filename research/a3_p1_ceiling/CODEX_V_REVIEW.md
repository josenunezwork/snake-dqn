# Independent Codex-V review receipt

Verdict: **PASS for source and offline preparation**. Reviewer: independent
`/root/codex_v` agent, custom `reviewer` role, read-only. Root transcribed this
receipt from the reviewer's final response and two explicit PASS addenda. Review
base: `936791756d5dc77d29c82391321387de886ca917`, plus the exact new-file hashes below.

Reviewer read AGENTS.md, PLAN_A100.md and round-20 JUDGMENT.md in full, then
inspected benchmark, runtime, preparation, aggregation, shell and relevant P1/Colab
paths. Corrected findings re-reviewed: nested incomplete actor updates cannot pass;
baseline-only kill precedence; actual gradients captured only in the numerical
comparison; GPU-busy sampling during collection and learning; immutable export
timing retained; both target ledgers verified before outcomes; qualified shell
device parser compatibility; ledger CLI positional parsing.

Root execution evidence: 12 narrow tests passed in 0.47 s, 87 existing Colab
canary/schema tests passed in 0.83 s; Black/isort/flake8/shell syntax passed.
Reviewer independently inspected and SHA-verified all seven tiny wrapper units,
plus final tiny optimized comparison and export timing. Two Adam epochs per
repeat; maximum absolute gradient difference `3.0640512704849243e-7`, within
`atol=2e-5`, `rtol=2e-4`. Final tiny payload SHA:
`a82e7c696475f33fb81e89d2b0f584ffa5d75e9416c49f9402642cad1714d4df`.
Root additionally ran the real ledger CLI and verified-all-seven skip path.
The reviewer ran no Python or simulator.

| Reviewed file in this directory | SHA-256 |
|---|---|
| `__init__.py` | `7f27dc18c29499768e867efc0c5792a5aee21ae8e0b7dad9f4b1e130720749b8` |
| `analyze.py` | `eb9a8a51c3b96269d8a6fef3fdfb44d40037d0234755c0c12b700439bc3f8aa3` |
| `benchmark.py` | `1271962195a48d158bb2fc8e2d615236ea99482d9dff3d2ac364dc7a1e5c8309` |
| `prepare.py` | `4a21ff541c225b88e14bcfd80f99176410120e9c6cc2ed1ec0bbaa0535fbe600` |
| `runtime.py` | `076ea0132d4190f8621784f0d18c2083af44c9f727525ce53e9a3a30ada468cb` |
| `colab.sh` | `07a9864ad0dd8f2727bb2967afed1c1e0b5e9722ce2b32be1ec19bd0a11a35c5` |
| `test_benchmark.py` | `16104980d20039a7afe3058b98e5d8512d453e36bd0b2c20ff5b12fbad908ec5` |
| `test_runtime.py` | `8648e0cb59612c63634728449fe8f8e98436f5d70b82c3847b6ca08d8296e3b4` |
| `PROTOCOL.md` | `6f84561cc841ea9e031fa1d96b0954b94cc23613410f6c2b884894efb80f61c2` |

Final cap review: A100 canary 16 min/3 u/0.6 reserve, assay 104 min/29 u/5.8 reserve;
L4 canary 20 min/1 u/0.2 reserve, assay 100 min/7 u/1.4 reserve. Sum 240 min/40 u/
8 reserve; base envelopes exactly half, estimates half the effective timeout.
Qualified Colab subtree unchanged: `553af3f692515b2bf66173356fe4d099acc96942`.

Residual limits: offline pinned-pack validation follows this source commit. Live
A100/L4 same-pin canaries, production memory use, CUDA gradients and timing are
unverified. Chunking does not remove the full retained autograd graph; OOM remains
INCOMPLETE. No GPU result, P1 qualification or serving authority is asserted.
