# Z-9 E4 recovery preparation — 2026-10-21

Status: **VALIDATED, NOT DISPATCHED**. No Colab provider calls or gameplay run.

Published executable pin: `62d6bf0c6d9cf274fcf9aa8a5084ca216f808b34`.
This pin includes the shared qualification helper and experiment integrations.
The implementation's preceding helper commits are
`465ba7740883bf3845a3c2623fdc57e5a38af94b` and
`a35ee0d01cf4ea8b63649fe65c01bea0433c7b22`.

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/z9-e4-recovery-01/`.

| Artifact | SHA-256 |
| --- | --- |
| `e4-manifest.json` | `e8138135c676e32c5bedcd07fcc1a79a0625b9f82084377f8bbbad9826d3a710` |
| `upload-preflight.json` | `eb9028fbb21942bde401178bf9f1cfb216d0d27a4ce76427d6035288ab2ebb46` |
| `DISPATCH.txt` | `a92dfad24c49072024ece09b357cb584d5e6d41f4bdcbabbd61a333e20576c9f` |

The output path is `e4-output` beneath the artifact root.
It does not exist at preparation. The job is the 128-world L4 E4 ruler,
with 6/12 collector processes, a 15-minute estimate, 30-minute timeout,
2-unit cap, 2 u/hour rate, 0.4-unit reserve and 600-second lease. The manifest
contains an empty account-inventory placeholder and remains non-dispatched.
Refresh inventory and requalify the resulting manifest before any future run.

Only the 2,409-byte public-input catalog is a local job input. Its SHA-256 is
`736f7b4e7730f760b8b81db00772d14abcf84c74b261a8dc7630732fdd383a10`.
The runtime downloads the existing public B8 and league release assets and
verifies each of the six checkpoint identities before the ruler starts.

| Upload qualification | Measured bytes / forecast |
| --- | --- |
| Shared initial-upload cap | 300 seconds |
| Qualified upload forecast budget | 150 seconds |
| Conservative observed large-input rate | 44,484.78697659882 bytes/s |
| Bootstrap + source helper + catalog | 18,168 bytes / 65.4084 seconds |
| Including actual pinned sparse-source fallback | 2,451,754 bytes / 122.1144 seconds |
| Including the 120-second source-fetch bound | 242.1144 seconds |

The runtime download entrypoint was exercised locally with real public bytes
using `python -m research.bl_upload.inputs`. Evidence is in sibling
`z9-runtime-check-01/verification.json`: six files, 30,720,004 bytes, all
catalog identities verified. This is input reconstruction evidence, not a
measurement of a Colab upload, CUDA execution, or E4 throughput.

Verification:

- 61 targeted package tests passed across upload qualification, E4/E2, E3,
  served search, O2 readiness, GPU spike, E16 and prefix transport.
- 36 targeted supervisor source, transfer and public-input tests passed.
- Changed Python files passed Black, isort and flake8; changed launchers
  passed shell syntax checks; three missing-receipt CLI refusals passed.
- The regenerated manifest passed the real supervisor's allocation-free
  `validate` command. The package readiness CLI checks its saved receipt
  with `--verify`; neither operation starts Colab.

The protected supervisor and T34 round/path/exit packages are unchanged.
No serving changes, checkpoint mutations, tmux changes or automatic retries.
Historical manifests and failed receipts are retained. Other experiment
families must use their package-local Z-9 readiness entrypoint before dispatch;
E4 is the first regenerated recovery manifest in this task.
