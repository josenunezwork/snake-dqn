export const meta = {
  name: 'snake-dqn-review',
  description: 'Exhaustive read-only code + config review of the snake-dqn RL codebase: fan out per-subsystem reviewers, adversarially verify each finding against real code, synthesize a deduped prioritized report',
  phases: [
    { title: 'Review', detail: 'one reviewer per subsystem, structured findings' },
    { title: 'Verify', detail: 'adversarially refute each finding against the actual code' },
    { title: 'Synthesize', detail: 'dedupe, find systemic patterns, prioritize' },
  ],
}

// ---------------------------------------------------------------------------
// Shared context fed to every reviewer/verifier so they check the right
// invariants and don't re-flag intentional design.
// ---------------------------------------------------------------------------
const SHARED = `
PROJECT: Multi-agent RL platform that trains AI "snakes" with Apex DQN (PyTorch + PyQt5).
Repo root: /Users/josenunez/Projects/ml/snake-dqn  (paths below are relative to it).
This is a READ-ONLY review. Do NOT edit, format, or run training. Reading + Grep + light Bash (git, wc) only.

KNOWN INVARIANTS / INTENTIONAL DESIGN (verify these hold; do NOT re-flag them as new bugs):
- State vector: 58-D base input (direction one-hot 0-3, length 4, food 5-7, food density 8-23,
  danger map 24-39, boundaries 40-43, nearest enemy 44-49, enemy trend 49, 2nd enemy 50-52,
  kill-opp 53, per-action danger 54-56, boost-available 57). An OPT-IN 61-D "free-space"
  variant adds 3 trap-avoidance features. CNN/conv variants are newer and may be rougher.
- Action space: 6 relative actions = 3 directions (turn-left/straight/turn-right) x 2 speed modes
  (normal/boost). Boost costs body segments and requires a minimum length.
- Config split: the LIVE Apex learner reads apex.* keys; some training.* knobs are reconciled into
  apex.* at load time (a prior bug had them silently dead). Verify the reconciliation is correct/complete.
- Checkpoint/resume contract: resume must ABORT unless config gamma / n_step / hidden size match the .pth.
- Device: auto CUDA > MPS > CPU; on Apple Silicon CPU is the deliberate default (MPS slower for this tiny net).
- Eval: tournament_eval.py (hero vs FROZEN opponents, paired seeds, mass + CI) is the trusted evaluator;
  self-play eval inflates skill — that's known, don't re-flag it as a bug.
- DRQN/GRU: forward returns (q_values, hidden_state), supports single-step and sequence inputs;
  sequence training uses burn-in. A known GPU gotcha: "best" checkpoint saves can freeze — *_latest_* is used.

WHAT TO HUNT (in priority order):
1. Correctness bugs: wrong tensor shapes/indexing, off-by-one, wrong RL targets (double-DQN, n-step return,
   target-net sync, gamma application, terminal masking), priority/IS-weight math, SumTree math, action
   masking, state-vector index/normalization errors, GRU hidden-state handling across steps/episodes.
2. Robustness: unhandled edge cases, None/empty/zero-division, file/DB/socket/process resource leaks,
   exceptions that silently swallow or crash a worker, checkpoint save/load integrity, resume safety,
   numerical stability (log/exp/sqrt/div, NaN/inf, clamping).
3. Concurrency (Apex): IPC queue deadlock/backpressure, shared-memory races, stale-weight broadcast,
   worker shutdown/cleanup, orphaned processes, pickling hazards.
4. Config integrity: dead/typo'd keys, schema mismatches vs loader, unsafe defaults, inconsistent values.
5. Quality: dead code, duplication that should be reused, over-complex code that can be simplified,
   missing/incorrect type hints, misleading docstrings, test gaps for the risky paths above.

Be precise: cite exact file + line range, quote the offending code in 'evidence', give a concrete minimal fix.
Prefer real defects over style. If unsure, mark confidence 'low' and state what would confirm it. Do not pad.
`

const CATEGORIES = [
  'correctness-bug', 'rl-algorithm', 'robustness', 'resource-leak',
  'race-condition', 'concurrency', 'config-issue', 'numerical-stability',
  'dead-code', 'reuse-duplication', 'simplification', 'type-safety',
  'consistency', 'test-gap', 'documentation', 'performance', 'security',
]

const FINDING_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    summary: { type: 'string', description: 'One sentence on the overall health of this subsystem.' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string' },
          file: { type: 'string', description: 'repo-relative path' },
          lines: { type: 'string', description: 'e.g. "123" or "120-145"' },
          severity: { enum: ['critical', 'high', 'medium', 'low', 'nit'] },
          category: { enum: CATEGORIES },
          description: { type: 'string', description: 'What is wrong and why.' },
          evidence: { type: 'string', description: 'Quoted offending code or exact reasoning chain.' },
          impact: { type: 'string', description: 'Concrete consequence (crash, wrong gradient, leak, etc.).' },
          suggested_fix: { type: 'string', description: 'Concrete minimal fix.' },
          confidence: { enum: ['high', 'medium', 'low'] },
        },
        required: ['title', 'file', 'lines', 'severity', 'category', 'description', 'evidence', 'impact', 'suggested_fix', 'confidence'],
      },
    },
  },
  required: ['summary', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { enum: ['confirmed', 'refuted', 'uncertain'] },
    reasoning: { type: 'string', description: 'Code-grounded justification, quoting the actual lines you read.' },
    adjusted_severity: { enum: ['critical', 'high', 'medium', 'low', 'nit'] },
    fix_correct: { type: 'boolean', description: 'Is the suggested fix correct and side-effect-free?' },
    fix_note: { type: 'string', description: 'If the fix is wrong/incomplete, the corrected fix.' },
  },
  required: ['verdict', 'reasoning', 'adjusted_severity', 'fix_correct', 'fix_note'],
}

// ---------------------------------------------------------------------------
// Subsystems. File lists are explicit so reviewers stay focused.
// ---------------------------------------------------------------------------
const SUBSYSTEMS = [
  {
    key: 'model',
    label: 'Neural network models',
    files: ['src/model/apex_network.py', 'src/model/base_network.py', 'src/model/gru_network.py',
      'src/model/conv_network.py', 'src/model/conv_gru_network.py', 'src/model/model_factory.py',
      'src/model/checkpoint_manager.py', 'src/model/__init__.py'],
    focus: 'Forward-pass tensor shapes; dueling V+A combination (mean-subtraction); GRU/DRQN hidden-state '
      + 'handling for single-step vs sequence; CNN input reshaping and channel wiring; weight init; '
      + 'mixin composition (Base/Dueling/Noisy/Visualization); checkpoint_manager save/load and best/latest logic.',
  },
  {
    key: 'rl-core',
    label: 'Apex DQN policy & priorities (the learning algorithm)',
    files: ['src/training/apex_policy.py', 'src/training/base_dqn_policy.py',
      'src/training/apex_priorities.py', 'src/training/action_mask.py'],
    focus: 'TD/loss computation, double-DQN target selection, n-step return + gamma^n, terminal masking, '
      + 'target-network sync cadence, epsilon schedule, IS-weight application, TD-error -> priority mapping '
      + '(alpha/beta/epsilon), action masking correctness vs the 6-action space, gradient clipping.',
  },
  {
    key: 'buffers',
    label: 'Replay buffers & SumTree',
    files: ['src/training/sum_tree.py', 'src/training/base_buffer.py', 'src/training/replay_buffer.py',
      'src/training/multistep_buffer.py', 'src/training/sequence_buffer.py'],
    focus: 'SumTree propagate/retrieve/total math and capacity wraparound; prioritized sampling (segmented, '
      + 'no-replacement, min-prob), n-step accumulation + episode-boundary flush; DRQN sequence storage, '
      + 'burn-in, padding/masking; off-by-one on insert/overwrite; is_ready thresholds; dtype/device handling.',
  },
  {
    key: 'apex-distributed',
    label: 'Apex actor + distributed buffer process (IPC)',
    files: ['src/training/apex_actor.py', 'src/training/apex_buffer.py'],
    focus: 'Multiprocessing lifecycle, IPC queue backpressure/deadlock, shared-memory races, weight-broadcast '
      + 'staleness, pickling of tensors/np arrays, worker exception handling (silent death), graceful shutdown '
      + 'and orphan/zombie cleanup, queue.get/put timeouts, CPU/GPU tensor transfer.',
  },
  {
    key: 'learner-coordinator',
    label: 'Apex learner + distributed train coordinator',
    files: ['src/training/apex_learner.py', 'src/scripts/apex_train.py'],
    focus: 'Learner step loop, optimizer/scheduler, target sync, checkpoint save (best vs latest freeze issue), '
      + 'torch.compile gating (CUDA-only), resume contract enforcement (gamma/n_step/hidden match), process '
      + 'spawn/join/cleanup, signal handling, config reconciliation into apex.*, logging cadence.',
  },
  {
    key: 'game-snake',
    label: 'Snake entity + AI snake (state vector construction)',
    files: ['src/game/snake.py', 'src/game/ai_snake.py'],
    focus: 'The full 58-D / 61-D free-space state vector: every index, normalization range, sign convention; '
      + 'relative-action mapping to 6 outputs; boost cost / min-length gating; growth & body update; '
      + 'sector binning for density/danger maps; enemy feature computation; off-board / empty-enemy edge cases.',
  },
  {
    key: 'game-world',
    label: 'Game world: state, logic, food, factory, human',
    files: ['src/game/game_state.py', 'src/game/game_logic.py', 'src/game/food_manager.py',
      'src/game/snake_factory.py', 'src/game/human_snake.py', 'src/game/__init__.py'],
    focus: 'Collision detection (self/other/wall, circular vs rectangular arena), kill attribution via '
      + 'collision-pair tracking, food spawn/respawn and overlap avoidance, frame stepping order, '
      + 'DI wiring in SnakeFactory (get_frame/set_frame closures), human input mapping, respawn/reset.',
  },
  {
    key: 'core-config',
    label: 'Core config system & device manager',
    files: ['src/core/game_config.py', 'src/core/config_loader.py', 'src/core/device_manager.py',
      'src/core/reward_contract.py', 'src/core/__init__.py'],
    focus: 'Frozen-dataclass immutability, YAML->config field mapping completeness, the config-section '
      + 'reconciliation (training.* -> apex.*), unknown/typo key detection, unsafe defaults, device '
      + 'auto-selection + test override singleton, validation of ranges, reward_contract correctness.',
  },
  {
    key: 'curriculum-metrics',
    label: 'Curriculum, online trainer, metrics, logging, checkpoint contract',
    files: ['src/training/curriculum.py', 'src/training/online_trainer.py', 'src/training/metrics_tracker.py',
      'src/training/tensorboard_logger.py', 'src/training/checkpoint_contract.py', 'src/training/__init__.py'],
    focus: 'Curriculum phase-transition thresholds & monotonicity, online trainer step/update loop, metrics '
      + 'aggregation correctness (running means, windows, divide-by-zero), TB logging keys/cadence, '
      + 'checkpoint_contract validation completeness (matches the resume contract).',
  },
  {
    key: 'data-db',
    label: 'Experience/memory SQLite handler (3.3k LOC)',
    files: ['src/data/memory_db_handler.py', 'src/data/__init__.py'],
    focus: 'SQLite connection lifecycle & thread-safety (check_same_thread, per-thread conns), transaction '
      + 'commit/rollback integrity, cursor/connection leaks, SQL string building (injection / param binding), '
      + 'schema creation & migration, blob (tensor/np) serialization round-trip, bulk insert performance, '
      + 'WAL/pragma settings, error handling on corrupt rows.',
  },
  {
    key: 'scripts-eval',
    label: 'Evaluation & checkpoint-surgery scripts',
    files: ['src/scripts/tournament_eval.py', 'src/scripts/evaluate_checkpoints.py', 'src/scripts/ensemble_eval.py',
      'src/scripts/audit_replay.py', 'src/scripts/rebase_checkpoint.py', 'src/scripts/widen_input.py',
      'src/scripts/build_drqn_warmstart.py'],
    focus: 'Eval statistical correctness (paired seeds, hero vs frozen, CI computation, sample size), '
      + 'deterministic seeding, checkpoint surgery correctness (widen_input 58->61, drqn warmstart, rebase) '
      + 'preserving weights/shapes, ensemble aggregation, audit_replay integrity checks.',
  },
  {
    key: 'gen-experiences',
    label: 'Experience generation (2.3k LOC)',
    files: ['src/scripts/generate_experiences.py'],
    focus: 'Experience tuple construction (s,a,r,s\',done) correctness, reward computation matching the live '
      + 'reward contract, n-step alignment, episode boundaries, parallelism/seeding, DB write batching, '
      + 'memory growth, resumability, off-by-one on transitions.',
  },
  {
    key: 'offline-overnight',
    label: 'Offline training + overnight campaign automation',
    files: ['src/scripts/offline_train.py', 'src/scripts/overnight_campaign.py'],
    focus: 'Offline RL loop correctness (sampling, target computation, overfitting guards), checkpoint cadence, '
      + 'overnight campaign orchestration robustness (subprocess management, failure isolation, disk/time '
      + 'budget, artifact bundling, log integrity, no silent stage skips).',
  },
  {
    key: 'ui-game',
    label: 'UI: slither renderer, game widget, network visualizer',
    files: ['src/ui/slitherio.py', 'src/ui/game_widget.py', 'src/ui/network_visualizer.py'],
    focus: 'PyQt5 paint/update loop correctness, timer cadence, coordinate transforms, signal/slot wiring, '
      + 'thread-affinity (GUI vs training thread), repaint thrash/perf, None-guards on model/state, '
      + 'resource cleanup on close, off-by-one in rendering of the 58/61-D features.',
  },
  {
    key: 'ui-panels',
    label: 'UI: inspector panel + training dashboard',
    files: ['src/ui/inspector_panel.py', 'src/ui/training_dashboard.py'],
    focus: 'Live metric display correctness, update cadence, division/format edge cases on empty data, '
      + 'signal/slot threading, memory growth from unbounded history, widget teardown.',
  },
  {
    key: 'main-cli',
    label: 'Main entry point & CLI',
    files: ['src/main.py'],
    focus: 'argparse wiring (every flag actually used), mode dispatch (gui/human/headless/load/config), '
      + 'resume contract enforcement at the CLI boundary, config initialization order, seed setup, '
      + 'graceful KeyboardInterrupt/shutdown, exit codes, mutually-exclusive flag validation.',
  },
  {
    key: 'configs',
    label: 'All YAML configs',
    files: ['configs/ (review every *.yaml)'],
    focus: 'Cross-check EVERY key against what config_loader/game_config actually consume (flag dead/typo keys); '
      + 'value sanity (lr, gamma, n_step, epsilon, hidden size, batch, arena size); internal consistency within '
      + 'a config (e.g. min_boost_length vs boost cost); consistency across config families; resume-compat '
      + 'fields (gamma/n_step/hidden) matching their intended checkpoints; eval configs pointing at valid paths.',
    extra: 'First read src/core/config_loader.py and src/core/game_config.py to learn the real schema, then '
      + 'list every configs/*.yaml and audit each. Use Grep to confirm whether a YAML key is read anywhere in src.',
  },
  {
    key: 'tests-utils',
    label: 'Test suite quality + utils + colab',
    files: ['tests/ (review the suite)', 'src/utils/nn_utils.py', 'src/utils/tensor_utils.py',
      'src/utils/__init__.py', 'colab/h100_apex_v1.py', 'tests/conftest.py'],
    focus: 'Test gaps on the highest-risk paths (RL targets, buffers, IPC, checkpoint/resume, state vector, '
      + 'config reconciliation); weak assertions / tests that cannot fail / tautological tests; flaky or '
      + 'order-dependent tests; fixtures leaking global state (device override, config singleton); utils '
      + 'correctness; whether colab script drifted from src.',
    extra: 'List tests/ first. You do not need to read every test line-by-line — map coverage, then deep-read '
      + 'the tests guarding risky code and spot what is NOT tested.',
  },
]

// ---------------------------------------------------------------------------
function buildReviewPrompt(s) {
  const fileList = s.files.map((f) => '  - ' + f).join('\n')
  return `You are an expert PyTorch / RL / systems reviewer auditing the snake-dqn codebase.
${SHARED}

YOUR SUBSYSTEM: ${s.label}
FILES:
${fileList}
${s.extra ? '\nAPPROACH: ' + s.extra + '\n' : ''}
FOCUS for this subsystem:
${s.focus}

METHOD:
- Read every listed file IN FULL (chunk large files; do not skim). Use Grep to trace contracts/callers when a
  finding depends on how a symbol is used elsewhere.
- Confirm the named invariants actually hold in the code; a violated invariant is a high-severity finding.
- Report up to 12 of your highest-value findings, strongest first. Fewer is better than padding. Every finding
  must cite exact file + line range and quote the offending code in 'evidence'. If a file is clean, say so.
- Separate real defects from intentional design listed in KNOWN INVARIANTS — do not re-flag intentional behavior.

Return via the structured schema (a 'summary' sentence + the 'findings' array).`
}

function buildVerifyPrompt(f, sub) {
  return `You are an ADVERSARIAL verifier on the snake-dqn review. A reviewer of the "${sub}" subsystem claims the
issue below. Your job is to REFUTE it by reading the ACTUAL code. Default to 'refuted' if the finding misreads
the code, rests on a false premise, or describes intentional/correct behavior.
${SHARED}

CLAIMED FINDING
  title: ${f.title}
  location: ${f.file}:${f.lines}
  severity: ${f.severity}   category: ${f.category}   reviewer-confidence: ${f.confidence}
  description: ${f.description}
  evidence: ${f.evidence}
  impact: ${f.impact}
  suggested_fix: ${f.suggested_fix}

STEPS:
1. Read ${f.file} around lines ${f.lines} (and trace callers/contracts with Grep as needed).
2. Decide if the defect is REAL: the code genuinely behaves as claimed AND that behavior is wrong/risky/suboptimal.
3. Check whether 'suggested_fix' is correct and would not break something else; if not, give the corrected fix in fix_note.
4. Re-rate severity from actual impact (downgrade hype, upgrade understated danger).

Be concrete and quote the lines you actually read. Return the structured verdict.`
}

// survive rule: keep a finding unless it was clearly refuted.
function survives(verdicts) {
  const vs = verdicts.filter(Boolean)
  if (vs.length === 0) return true // verifier died -> don't silently drop; let synthesis judge
  const confirmed = vs.filter((v) => v.verdict === 'confirmed').length
  const refuted = vs.filter((v) => v.verdict === 'refuted').length
  if (confirmed >= 1) return true
  if (refuted >= 1 && confirmed === 0) return false // all non-confirming and at least one refute
  return true // only uncertain -> keep for synthesis
}

// ---------------------------------------------------------------------------
phase('Review')
log(`Reviewing ${SUBSYSTEMS.length} subsystems across ~30k LOC, then adversarially verifying every finding.`)

const perSubsystem = await pipeline(
  SUBSYSTEMS,
  // Stage 1: review the subsystem.
  (s) => agent(buildReviewPrompt(s), {
    label: `review:${s.key}`,
    phase: 'Review',
    schema: FINDING_SCHEMA,
  }).then((r) => ({ sub: s.key, label: s.label, result: r })),

  // Stage 2: verify each finding from THIS subsystem (starts as soon as the review lands).
  (reviewed) => {
    if (!reviewed || !reviewed.result || !reviewed.result.findings) return reviewed
    const findings = reviewed.result.findings
    return parallel(findings.map((f) => () => {
      const nVerifiers = (f.severity === 'critical' || f.severity === 'high') ? 2 : 1
      return parallel(
        Array.from({ length: nVerifiers }, (_, i) => () =>
          agent(buildVerifyPrompt(f, reviewed.sub), {
            label: `verify:${reviewed.sub}:${(f.title || 'finding').slice(0, 28)}#${i + 1}`,
            phase: 'Verify',
            schema: VERDICT_SCHEMA,
          })),
      ).then((verdicts) => ({ ...f, sub: reviewed.sub, verdicts: verdicts.filter(Boolean) }))
    })).then((verifiedFindings) => ({ ...reviewed, verifiedFindings }))
  },
)

// Collect surviving findings.
const all = []
const rejected = []
for (const entry of perSubsystem.filter(Boolean)) {
  for (const f of (entry.verifiedFindings || [])) {
    const keep = survives(f.verdicts)
    // pull the worst adjusted severity / fix correctness signal from verifiers
    const adj = f.verdicts.map((v) => v.adjusted_severity).filter(Boolean)
    const fixIssues = f.verdicts.filter((v) => v.fix_correct === false).map((v) => v.fix_note).filter(Boolean)
    const record = {
      title: f.title, file: f.file, lines: f.lines, severity: f.severity,
      adjusted_severity: adj[0] || f.severity, category: f.category, subsystem: f.sub,
      description: f.description, impact: f.impact, suggested_fix: f.suggested_fix,
      confidence: f.confidence,
      verifier_verdicts: f.verdicts.map((v) => v.verdict),
      verifier_reasoning: f.verdicts.map((v) => v.reasoning),
      fix_corrections: fixIssues,
    }
    if (keep) all.push(record)
    else rejected.push(record)
  }
}
log(`Findings after verification: ${all.length} kept, ${rejected.length} refuted.`)

// ---------------------------------------------------------------------------
phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    executive_summary: { type: 'string' },
    health_by_subsystem: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { subsystem: { type: 'string' }, verdict: { type: 'string' } },
        required: ['subsystem', 'verdict'],
      },
    },
    systemic_patterns: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { pattern: { type: 'string' }, where: { type: 'string' }, recommendation: { type: 'string' } },
        required: ['pattern', 'where', 'recommendation'],
      },
    },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          rank: { type: 'number' },
          title: { type: 'string' },
          severity: { enum: ['critical', 'high', 'medium', 'low', 'nit'] },
          category: { enum: CATEGORIES },
          locations: { type: 'string', description: 'file:lines (may merge duplicates across files)' },
          problem: { type: 'string' },
          fix: { type: 'string' },
          merged_from: { type: 'number', description: 'how many raw findings merged into this one' },
        },
        required: ['rank', 'title', 'severity', 'category', 'locations', 'problem', 'fix', 'merged_from'],
      },
    },
    stats: {
      type: 'object', additionalProperties: false,
      properties: {
        critical: { type: 'number' }, high: { type: 'number' }, medium: { type: 'number' },
        low: { type: 'number' }, nit: { type: 'number' },
      },
      required: ['critical', 'high', 'medium', 'low', 'nit'],
    },
  },
  required: ['executive_summary', 'health_by_subsystem', 'systemic_patterns', 'findings', 'stats'],
}

const synthPrompt = `You are the lead reviewer consolidating a full-codebase review of snake-dqn (PyTorch + PyQt5 Apex DQN).
Below are findings that were EACH already adversarially verified against the real code (refuted ones removed).
${SHARED}

TASKS:
- DEDUPLICATE: merge findings with the same root cause (even across files/subsystems); set merged_from accordingly.
- Find SYSTEMIC PATTERNS: issues that recur across files (e.g. the same missing guard, the same resource leak shape,
  config keys that are read nowhere, repeated normalization mistakes).
- PRIORITIZE: produce a single ranked 'findings' list, rank 1 = most important. Use the adjusted (verifier) severity.
  Keep concrete file:lines in 'locations'. Write 'problem' and 'fix' tightly and actionably.
- Give a crisp 'executive_summary' and a one-line health verdict per subsystem.
- Compute 'stats' = count by final severity across the deduped findings.
Do not invent new findings; only consolidate what is given. Prefer correctness/robustness items at the top.

VERIFIED FINDINGS (JSON):
${JSON.stringify(all, null, 1)}
`

const report = await agent(synthPrompt, { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, effort: 'high' })

return { report, kept_count: all.length, refuted_count: rejected.length, raw_kept: all, refuted: rejected }
