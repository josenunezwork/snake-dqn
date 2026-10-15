# Spec: T2b Huber-delta sweep, probes and analysis (declared before data, 2026-10-15)

Read AGENTS.md, then docs/research/specs/t2_probe_spec_2026-10-15.md (your T2 spec) and
docs/research/t2_screen_status_2026-10-15.md (your T2 result). Development screen, not gate evidence.

## Why
T2: HUBER (delta 100, anchor 0.1) raised scripted episode survival +0.125 vs CTRL (SE ~0.03) but
cost 80-140 mass vs CTRL (GO needs >= -30); HUBER_s12 halted on the action-collapse tripwire;
ANCHOR0 null; HUBER_ANCHOR0 killed (mass collapse). The anchor stays at 0.1; the question is
whether a milder delta keeps most of the survival gain at an acceptable mass cost.

## Runs (already launched by Claude, do not touch)
tmux `t2b-lane0`/`t2b-lane1` run `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t2b-mac-20261015/lane{0,1}.sh`:
`HUBER10_s{10..14}` (`--td-huber-delta 10`) and `HUBER30_s{10..14}` (`--td-huber-delta 30`), each an
8M fine-tune from B8_s{k} final with the M6 recipe and `--anchor-floor 0.1` (identical to T2 HUBER
except delta). `status.txt` lines `NAME exit=RC` when each ends.

## Task
1. Stop your idle T2 watcher (tmux session `t2p-run`, which you started) cleanly.
2. Extend `research/t2_screen_20261015/` (watcher + analysis) to arms HUBER10 and HUBER30 with run
   dir `redesign-t2b-mac-20261015`, same probe settings, same round_index 60 bank (96 scripted / 32
   frozen / 32 mixed), probe outputs into the existing probe dir. REUSE the existing CTRL and
   incumbent probe rows (do not re-probe them). Start the watcher in tmux `t2b-probe`, at most 2
   probe processes while t2b lanes are alive, up to 4 after.
3. Decision per arm, same rule as T2: GO iff paired scripted episode-survival delta vs CTRL >= +0.06
   AND mass delta vs CTRL >= -30 on every mix AND no other mix's episode survival down > 0.05; KILL
   iff primary <= 0; else EXTEND once (96 more scripted worlds, round_index 61, arm + CTRL; reuse
   the ANCHOR0 extension's CTRL rows if they exist on that bank). Holdout agreement is not logged by
   the multi-actor trainer: report it as unavailable and do not use it as a blocker. A halted run
   (exit != 0) is dropped and blocks GO for that arm (as T2). If both arms GO, the preferred arm is
   the one with the higher mean mass delta vs CTRL.
4. Also tabulate all T2 + T2b Huber arms together (delta 10 / 30 / 100) as a dose-response table:
   survival (both metrics) and mass vs CTRL and vs the incumbent per mix.
5. Write docs/research/t2b_delta_sweep_2026-10-15.md (header "development screen, not gate
   evidence"; the rule above verbatim; tables; decisions). Tests for any new analysis code; flake8.
   Commit on branch t2p-20261015 (already fast-forwarded to main); do not push; do not edit INDEX.
   Training takes ~4-5 h: keep polling until the doc exists.
Final message: per-arm decision with deltas/SEs, the dose-response table, commit sha, paths.
