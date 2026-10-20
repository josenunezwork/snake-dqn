# Addendum (Claude, 2026-10-19 evening): input delivery is SOLVED — no new publication

All six T34 inputs are ALREADY public, byte-identical, in the existing release used by S1:
https://github.com/josenunezwork/snake-dqn/releases/download/s1-inputs-20261019/
- `B8_s3_final.pth` (sha256 3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8) — direct asset.
- `league.tar.gz` contains, byte-identical to T33 INPUTS_REQUIRED.json: champion_a5_freespace_20260621.pth,
  best_apex_fs.pth, best_apex_stage1_fs.pth, best_apex_pre_fs.pth, frp3_m3_s12_u60000_20261005.pth.

Use the existing public-input path (`research/colab_cli/inputs.py`, as S1 does): manifest rows with
url + sha256 + size_bytes; the runtime downloads, verifies, extracts league.tar.gz and verifies each
member's SHA-256 against the table above. Do NOT ship checkpoints via ContentsClient uploads: measured
S1 Contents goodput was ~13 KB/s and a ~62 MB staging hit the 300 s cap (staging-failure-review.json).
Contents uploads stay for tiny control files only. Do not create or modify any GitHub release.
