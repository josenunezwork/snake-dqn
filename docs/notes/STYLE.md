# Notes style guide

Every file under `docs/notes/` follows this guide. The notes are a fact-only, cited, consistently
written record of the snake-dqn project as of the date in each note's header. They are not plans,
not status dashboards, and not narrative.

## 1. File shape

```
# <Title: 2–6 words, sentence case>

**Summary:** <one sentence, ≤ 35 words, stating what this note covers and the single most important fact.>
**As of:** 2026-10-25 PT · **Sources:** fact sheet `_facts/<topic>.md`; primary records cited inline.
**Related:** [other-note](other-note.md), [GLOSSARY](GLOSSARY.md)

## <Section>
<short paragraphs or bullets>

## Superseded and conflicting records
<what older record was replaced by what newer one, with both citations>

## Unverified
<claims that could not be confirmed against a source; say why>
```

- Headings: `#` title once; `##` sections; `###` only inside a long section. Sentence case. No emoji.
- Length: 80–300 lines. Split a topic rather than exceed 300 lines.
- Every note links to `README.md` (index) and `GLOSSARY.md` in **Related**.

## 2. Facts, citations, and status

- One fact per sentence or bullet. Keep every number with its unit and comparator
  (e.g. "+83.0 mass pooled, LB90 +64.1, vs bare B8 s3").
- Every factual sentence or bullet ends with a citation in this exact form:
  `(src: path:line)` or `(src: path:line-line)` or `(src: git <7-char sha>)`.
  Several sources: `(src: a:12; b:40)`. Paths are repo-relative to
  `/Users/josenunez/Projects/ml/snake-dqn/`; artifact paths start with
  `big-lever/` (meaning `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/`).
  Hash-bound rulings are cited by filename.
- Do not cite memory, chat, or your own inference. A claim without a locatable source goes to
  **Unverified**, worded as a claim, with the reason.
- Status words are the source's labels, in capitals as written there:
  `STRICT_PASS`, `SERVING_PASS`, `KILL`, `HOLD`, `ACCEPT`, `FIX-FIRST`, `FUND_SYSTEMS`,
  `STOP_FUNDING`, `NO_CONTINUE`, `INCOMPLETE`, `PARTIAL`, `GO`, `NO-GO`. Do not invent labels.
- Liveness is never asserted. Anything that was running, pending, or scheduled is written as
  "as of 2026-10-24 11:45 PT it was running" with the source; never "is running".
- Conflicts: the newer ruling, receipt, or erratum wins. State both, newest first, in
  **Superseded and conflicting records**. In the body use only the winning value and add
  "(supersedes …)" when the older value is well known.

## 3. Dates, times, units, names

- Dates: ISO `YYYY-MM-DD`. Times: 24-hour Pacific, `HH:MM PT`; convert UTC stamps
  (`T00:29:51Z` → `17:29 PT` on the previous calendar day when applicable) and say "(from UTC)".
- Numbers: thousands separators (`20,000`), ≤ 3 significant decimals unless the source is exact
  (`0.0662`), `×` for ratios (`20.13×`), `%` attached (`3.8%`), `≥ ≤` with a space before the number.
- Units as the sources use them: `u` = Colab compute units; `mass` = H5000 mass-integral delta
  unless stated; `LB90`/`UB90` = 90% lower/upper bound; `pp` = percentage points; `s`, `h`, `fps`.
- Commits: 7–8 hex characters in backticks (`6cc4580c`). Checkpoints by their registry name
  (`b8-s3`, `frp3-s12`, `champion`) with SHA prefix on first mention.
- Study and lane names exactly as the sources spell them (`T34`, `Round 0`, `L2′`, `K-1`, `D1024`,
  `F4`, `R-A`, `I0S`). First use of any such term in a note gets a GLOSSARY entry or a one-clause
  gloss in parentheses.
- People/roles: "the owner"; "Sol" (Codex implementer), "Astra" (Codex advisor/judge),
  "Fable" (Claude judge), "Opus stand-in"; never personal names or emails.

## 4. Tense and voice

- Past tense for events and results ("the judge ruled", "the fit completed").
- Present tense only for standing rules still in force at the **As of** date ("strict gates run on
  the Mac") and for definitions.
- Active voice, plain words, no adjectives of judgment ("great", "unfortunately"), no hedges
  unless the source hedges (then quote the source's own qualifier).
- No first person, no instructions to the reader, no "next steps".

## 5. What to leave out

- Plans that were never executed, unless a note records their cancellation as a fact.
- Duplicated narrative that another note already carries: link to it instead.
- Live operational detail (tmux names, PIDs, worktree paths) except where a ruling binds to it.
- Any content of label/outcome payloads, sealed tranches, or model weights.

Files that exist only on another branch are cited as `(src: git branch <branch> path:line; git <sha>)`.
