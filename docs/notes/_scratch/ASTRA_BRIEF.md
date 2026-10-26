# Brief for note writers (Astra), notes cleanup 2026-10-25

You are rewriting ONE topic of the snake-dqn project notes as a clean, fact-only, consistently written note.

## Inputs (read these first, in order)
1. /Users/josenunez/Projects/ml/snake-dqn-notes/docs/notes/STYLE.md — follow it EXACTLY (file shape, citation form, dates, labels).
2. Your fact sheet: /Users/josenunez/Projects/ml/snake-dqn-notes/docs/notes/_facts/<topic>.md — the ONLY source of claims.
   Its header defines path abbreviations used in its `src:` fields; expand them to real paths in your note.
3. The sources the fact sheet cites, to verify. Repo-relative paths are under /Users/josenunez/Projects/ml/snake-dqn/;
   `big-lever/...` and absolute paths are under /Users/josenunez/Projects/ml/snake-dqn-artifacts/.

## Hard rules
- Write ONLY your target file(s) under /Users/josenunez/Projects/ml/snake-dqn-notes/docs/notes/. Never edit any other file.
- Everything else is READ-ONLY: never edit/move/create files in /Users/josenunez/Projects/ml/snake-dqn or snake-dqn-artifacts;
  never touch files with a .sha256 sidecar; never open label/outcome/model payloads, `outputs/` dirs, the sealed D4096 tranche
  (bl-labels-f0-speed/d4096-old-equiv-02-pack-01/) or t34-round0-study-01/; never touch tmux, running jobs, exp/mac-lease.json*,
  or the worktrees snake-dqn-r30post / snake-dqn-f4pm / snake-dqn-f5annex.
- No git commands that change state (no add/commit/checkout/stash). `git log`/`git show --stat` read-only is fine.
- Keep CPU light: prefix shell commands with `nice -n 15`; use grep/sed/head, never run project code or tests.
- Facts only from the fact sheet. Before using a fact, VERIFY its citation (open the file at the line; `sed -n 'N,Mp' path`).
  If the citation does not support the claim, DROP the claim or move it to the note's "Unverified" section with the reason.
  Do not add facts from your own knowledge. Do not invent labels, numbers, or dates.
- Dedupe: the fact sheet may repeat a fact; the note says it once. Where the sheet records a conflict, the NEWER
  ruling/receipt/erratum wins in the body; both go in "Superseded and conflicting records".
- Drop: plans never executed (unless their cancellation is the fact), live operational detail (tmux/PIDs/worktree paths)
  unless a ruling binds to it, speculation, and "next steps".
- Cross-links: link other notes by filename only from this list (they exist or are being written in parallel):
  overview-and-serving.md, models-and-training-history.md, redesign-and-b8.md, governance-and-owner-rules.md,
  t34-campaign.md, big-lever-program.md, big-lever-experiments.md, f4-study.md, label-factory.md,
  colab-and-compute-infra.md, gpu-engine-and-speed.md, r30-r32-f5-successors.md, lessons-and-incidents.md,
  GLOSSARY.md, README.md.
- Length 80–300 lines. Finish by re-reading STYLE.md and checking: summary line present, every factual sentence cited,
  all status labels in capitals as sourced, dates ISO + PT, no present-tense liveness, Superseded + Unverified sections present.
- Do NOT commit. Your final message: ≤120 words — file written, line count, number of facts dropped/moved to Unverified and why.
