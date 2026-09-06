// Mirrors the backend frame DTO (web/backend/serialize.py). The frontend never
// computes any of this — it renders whatever the server streams.

export interface Arena {
  width: number;
  height: number;
  segment: number;
  wall: number;
  arena_type: string;
}

export type Point = [number, number];

export interface SnakeDTO {
  id: number;
  color: [number, number, number];
  name: string;
  alive: boolean;
  boosting: boolean;
  length: number;
  segments: Point[];
  head: Point;
  is_hero: boolean;
}

export interface Stats {
  alive: number;
  frame: number;
  food_eaten: number;
  deaths: number;
  kills: number;
  best_length: number;
  loss: number | null;
  epsilon: number;
}

export interface StateGroup {
  name: string;
  start: number;
  end: number;
}

export interface InspectorDTO {
  input_size: number;
  state: number[];
  groups: StateGroup[];
  q_values: number[];
  action_labels: string[];
  // Raw greedy argmax over the Q-values (kept for compatibility). Display
  // executed_action as "action taken" when present; this is the "greedy pick".
  chosen: number;
  // The action the hero ACTUALLY took last step (post-masking, post-exploration).
  // Null/absent on older backends.
  executed_action?: number | null;
  free_space: number[] | null;
}

// Compact ego-raster payload for the raster31v2 serving path (mirrors
// web/backend/serialize.py::_hero_raster). Absent (null) on the vector61 path.
// The tactical planes are heading-rotated so the snake faces "up" in the grid.
export interface HeroRasterDTO {
  tactical_size: number; // 31
  tactical_code: number[][]; // (S, S) type code per cell (see tactical_codes)
  tactical_value: number[][]; // (S, S) uint8 value byte per cell
  tactical_channels: string[]; // (9,) channel names, index = expanded plane
  tactical_codes: Record<string, number>; // name -> type code, e.g. own_head: 8
  strategic_size: number; // 25
  strategic: number[][][]; // (3, S, S) density planes
  strategic_channels: string[]; // (3,) names
  scalars: number[]; // (26,)
  mask: boolean[]; // (6,) legal-action mask
  // Named index ranges into `scalars` (mirrors the inspector state groups);
  // lets the viewer label scalar clusters. Absent on older backends.
  scalar_groups?: StateGroup[];
}

export interface NetvizSummary {
  status: string;
  top_action: string;
  top_index: number;
  top_q: number;
  margin: number;
  input_count: number;
  hidden_count: number;
  output_count: number;
  input_activity: number;
  hidden_activity: number;
  architecture: string;
}

export interface NetvizDTO {
  input: number[];
  hidden_sample: number[];
  hidden_count: number;
  output: number[];
  // Dueling decomposition (present when the head is a Dueling DQN): scalar state
  // value V(s) and per-action advantages A(s,a). Null on older frames / non-dueling.
  value?: number | null;
  advantages?: number[] | null;
  summary: NetvizSummary;
}

export interface SessionState {
  playing: boolean;
  speed: number;
  mode: string;
  training: boolean;
  hero_id: number;
  checkpoint: string | null;
  config: string;
  input_size: number;
  num_snakes: number;
  epsilon: number;
  food_target: number;
  food_count: number;
  error: string | null;
  // True when re-issuing the failed action with the reward-contract override
  // would succeed — the UI offers "Fine-tune anyway" instead of a dead end.
  error_overridable?: boolean;
  // True while the live training run leans on the reward-contract override
  // (fine-tune under current rewards, not a resume).
  reward_override_active?: boolean;
  // Observation contract of the served policy ("vector61" | "raster31v2").
  obs_spec?: string;
}

export interface PendingRun {
  score: number;
  length: number;
  food_eaten: number;
  kills: number;
  frames: number;
  duration_seconds: number;
  checkpoint: string | null;
}

export interface PlayState {
  active: boolean;
  human_id: number | null;
  opponents: number;
  human_alive: boolean;
  length: number;
  food_eaten: number;
  kills: number;
  frames: number;
  score: number;
  run_started: boolean;
  run_over: boolean;
  submitted: boolean;
  pending: PendingRun | null;
}

export interface Frame {
  type: string;
  frame: number;
  // Wire-format version (currently 2). The client shows a one-time warning
  // toast for versions it doesn't know; absent on pre-versioned backends.
  protocol_version?: number;
  // Number of connected WebSocket clients.
  viewer_count?: number;
  // Honest architecture label derived from the served policy's obs spec, e.g.
  // "Apex DQN (vector61)" or "Raster Dueling (raster31v2)".
  architecture?: string;
  // Basename of the loaded checkpoint (stable key for trace resets).
  checkpoint_name?: string;
  // True on the single post-pause frame and the ~1 Hz heartbeats that follow;
  // clients must not append telemetry/trace history for these frames.
  paused?: boolean;
  // Observation contract of the served policy: "vector61" (hand-crafted 61-D
  // champions) or "raster31v2" (ego-raster stack). Drives the raster viewer.
  obs_spec?: string;
  mechanics_version?: number;
  // Present only on the raster31v2 serving path (else null / absent).
  hero_raster?: HeroRasterDTO | null;
  arena: Arena;
  snakes: SnakeDTO[];
  food: Point[];
  stats: Stats;
  session: SessionState;
  play: PlayState | null;
  inspector: InspectorDTO | null;
  netviz: NetvizDTO | null;
}

export type ControlAction =
  | "play"
  | "pause"
  | "reset"
  | "new_game"
  | "set_speed"
  | "set_mode"
  | "set_epsilon"
  | "set_hero"
  | "set_food"
  | "load_checkpoint"
  | "human_input"
  | "human_boost"
  | "set_play_opponents"
  // Saves the live policy weights to saved_snakes/web_train_<ts>.pth; the
  // server replies with an {type:"info"} frame that the client toasts.
  | "save_weights"
  // Per-connection raster streaming subscription: value is {on: boolean}.
  // Sent on Raster tab enter/leave and re-sent on reconnect.
  | "set_raster_stream";

// Non-frame server messages pushed over the stream socket. Both are surfaced
// as toasts (error styling vs neutral).
export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface InfoMessage {
  type: "info";
  message: string;
}

// A toast-worthy server notice, deduped by useGameSocket (a wedged engine
// re-broadcasts the same error every tick). `seq` increments per new notice so
// consumers can effect on identity.
export interface ServerNotice {
  tone: "error" | "info";
  message: string;
  seq: number;
}

export interface LeaderboardEntry {
  rank: number;
  player_name: string;
  score: number;
  length: number;
  food_eaten: number;
  kills: number;
  frames: number;
  created_at: string;
  // Opaque per-browser id recorded with the game (schema v3); preferred over
  // the name match for the "me" highlight. Null/absent on older rows.
  client_id?: string | null;
  // True when the row belongs to this browser's client_id (preferred over the
  // name match for the "me" highlight). Absent on older backends.
  is_me?: boolean;
}

export interface GlobalStats {
  total_players: number;
  total_games: number;
  best_score: number;
  best_player: string | null;
  average_score: number;
}

export interface LeaderboardData {
  leaderboard: LeaderboardEntry[];
  stats: GlobalStats | null;
}

export interface RecentGame {
  id: number;
  player_name: string;
  score: number;
  length: number;
  food_eaten: number;
  kills: number;
  frames: number;
  duration_seconds: number;
  mode: string;
  checkpoint: string | null;
  created_at: string;
  client_id?: string | null;
}

export interface RecentData {
  recent: RecentGame[];
}

export interface PlayerStats {
  name: string;
  games_played: number;
  best_score: number;
  total_score: number;
  average_score: number;
  created_at: string;
  last_played_at: string | null;
}

export interface PlayerResponse {
  found: boolean;
  player: PlayerStats | null;
}

export interface SubmitResponse {
  ok: boolean;
  error?: string;
  result?: {
    player_name: string;
    score: number;
    length: number;
    food_eaten: number;
    kills: number;
  };
  leaderboard?: LeaderboardEntry[];
}

export type SendControl = (action: ControlAction, value?: unknown) => void;

// One lightweight, plottable slice of a frame, accumulated client-side into a
// rolling ring buffer for the live sparklines/telemetry. `loss` is nullable and
// only present intermittently — plots must skip the gaps, never coerce to 0.
export interface MetricSample {
  frame: number;
  loss: number | null;
  epsilon: number;
  foodEaten: number;
  alive: number;
  bestLength: number;
  kills: number;
}

export interface CheckpointInfo {
  name: string;
  size_mb: number;
  // Observation contract detected from the checkpoint ("vector61" |
  // "raster31v2" | "unknown"); cached server-side by (path, mtime).
  obs_spec?: string;
  // Repo-relative path (saved_snakes/... or runs/.../latest_pqn.pth).
  path?: string;
}

export interface EvalRow {
  file: string;
  candidate: string;
  opponent: string;
  frames: number | null;
  n: number;
  // Headline mass for BOTH schemas (mass_integral value on repaired-gate rows,
  // legacy alive-frames mean on old rows); `metric` disambiguates.
  mean_mass: number;
  max_mass: number;
  survival: number;
  kills: number;
  // "mass_integral" (repaired gate) | "legacy_mean_mass" (old gate). Absent on
  // very old backends.
  metric?: string;
  // ISO date of the eval file's mtime.
  date?: string;
  // 95% CI half-width for the paired mass delta, when the repaired gate wrote one.
  mass_ci?: number | null;
  mean_mass_alive?: number | null;
}

export interface DashboardData {
  checkpoints: CheckpointInfo[];
  leaderboard: EvalRow[];
}
