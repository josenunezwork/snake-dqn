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
  chosen: number;
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
  | "set_play_opponents";

export interface LeaderboardEntry {
  rank: number;
  player_name: string;
  score: number;
  length: number;
  food_eaten: number;
  kills: number;
  frames: number;
  created_at: string;
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
}

export interface EvalRow {
  file: string;
  candidate: string;
  opponent: string;
  frames: number | null;
  n: number;
  mean_mass: number;
  max_mass: number;
  survival: number;
  kills: number;
}

export interface DashboardData {
  checkpoints: CheckpointInfo[];
  leaderboard: EvalRow[];
}
