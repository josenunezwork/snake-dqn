# Spec: T9 divergence study, fatal vs surviving continuations after the LAD (2026-10-17, diagnostic)

Read AGENTS.md, `docs/research/advice/2026-10-17_open_next.md` (step 0, binding), the T1 study
(`docs/research/t1_commitment_study_2026-10-15.md`, `research/t1_commitment_20261015/`, rows at
`snake-dqn-artifacts/t1-commitment-20261015/rows.jsonl`) and its replay/clone machinery.

For each late death (L >= 800) with a LAD (46): re-create the LAD state, play (a) the recorded fatal
continuation and (b) the best surviving action then B8 greedy, and trace both for the 100 decisions after
the LAD (or to death): size of the head's connected free region (tail-aware, uncapped) and of the largest
free component; whether the head is in the largest component; distance (BFS and Manhattan) from the head to
the own tail; hero cells and opponent cells within radius 15 and 30 of the head; fraction of the head
region's boundary that is own body vs opponent body vs wall; whether the head is moving toward the nearest
food and whether that food lies in a smaller component; the cells reachable before any opponent within
16/32/64/128/256 steps (arrival race, opponents 1 cell/frame, tail vacating). Summarise per decision offset
(0, 10, 25, 50, 100) the paired difference fatal minus surviving with medians/IQR, and state which quantities
separate the branches earliest and most consistently. Also report how often the decisive divergence point
lies outside the 31x31 local window (pocket boundary beyond 15 cells from the head).
Doc `docs/research/t9_divergence_2026-10-17.md`, tests for new helpers, flake8. CPU only, at most 2 processes;
NEVER touch tmux t7-phase-r or processes you did not start. Git: branch t9-20261017 in
/Users/josenunez/Projects/ml/snake-dqn-t9; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t9-20261017`.
Final message: the separating quantities with numbers, the outside-window share, sha.
