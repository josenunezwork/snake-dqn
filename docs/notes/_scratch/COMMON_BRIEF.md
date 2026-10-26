# Common brief for fact-sheet agents (notes cleanup, 2026-10-25)

You are producing a FACT SHEET for one topic of the snake-dqn project. Output goes ONLY to
/Users/josenunez/Projects/ml/snake-dqn-notes/docs/notes/_facts/<topic>.md (overwrite it; it is yours).

## Hard rules
- READ-ONLY on every source. Never edit, move, rename or create files anywhere except your one output file.
- Never touch: files with a `.sha256` sidecar (read them, never write), existing docs, tmux sessions, running jobs,
  `exp/mac-lease.json*`, the worktrees snake-dqn-r30post / snake-dqn-f4pm / snake-dqn-f5annex.
- Never open label/outcome/model payloads (.npz/.npy/.pt/.pth/.pkl/.jsonl data tranches, `outputs/` dirs), the sealed D4096 tranche
  (anything under `bl-labels-f0-speed/d4096-old-equiv-02-pack-01/` outputs) or `t34-round0-study-01/`. Read .md/.json metadata/.sha256/README only.
- Keep CPU light: prefix shell commands with `nice -n 15`; use grep/sed/head, not whole-tree dumps. Do not run any project code.
- Do not use memory files or your own recollection as a source. Only the files listed under SOURCES in your prompt
  (plus files they link to, under /Users/josenunez/Projects/ml/snake-dqn/ or /Users/josenunez/Projects/ml/snake-dqn-artifacts/).
- Do not write any other report file. Your final chat reply is a ≤150-word summary (what you covered, counts, what you flagged).

## Fact sheet format (exactly this)
```
# FACTS: <topic title>
Sources read: <comma-separated paths>   (sheet written 2026-10-25 PT)

## <Section heading>
- [YYYY-MM-DD] <one fact, one sentence, precise numbers/units/names> — src: <path>:<line> | <path>:<line>
- ...

## Superseded / conflicts
- [YYYY-MM-DD] <older claim> SUPERSEDED BY [YYYY-MM-DD] <newer claim> — src: <old path:line>; <new path:line>

## UNVERIFIED
- <claim> — why unverifiable (source missing / contradictory / line could not be found)

## Incidents and lessons (facts only)
- [YYYY-MM-DD] <what happened> → <what was changed/learned, as recorded> — src: ...

## Terms (for the glossary)
- <TERM>: <one-line definition as the sources use it> — src: ...
```
- Every bullet carries `src:` with a file path (absolute or repo-relative from /Users/josenunez/Projects/ml/snake-dqn/) AND a line
  number (`path:line` or `path:line-line`), or a commit sha (`git <sha>`), or a hash-bound ruling filename.
  Verify each line number with grep/sed before writing it. If you cannot locate the line, put the claim under UNVERIFIED.
- Dates: the date the source records for the event (PT), not the file date. If only a file date exists, say `[file 2026-10-24]`.
- Facts only: no plans ("next we should"), no speculation, no adjectives, no narrative. State outcomes as the source labels them
  (e.g. STRICT_PASS, KILL, HOLD, FUND_SYSTEMS, STOP_FUNDING) and keep the numbers (mass delta, LB90, seeds, units, hours).
- Where sources conflict, the NEWER ruling/receipt wins; record both in "Superseded / conflicts".
- Flag status that may be stale ("running", "pending", "ETA") as of the source date — write `[as of YYYY-MM-DD HH:MM PT]` and never
  assert current liveness.
- Target 60–200 bullets. Prefer the primary record (registration, judgment, result file, receipt) over summaries (STATUS/TIMELINE/handoffs).
- Do not duplicate facts that clearly belong to another topic; one line pointing to the other topic name is enough. Topics:
  overview-and-serving, models-and-training-history, governance-and-owner-rules, t34-campaign, big-lever-program,
  f4-study, label-factory, colab-and-compute-infra, gpu-engine-and-speed, r30-r32-f5-successors.
