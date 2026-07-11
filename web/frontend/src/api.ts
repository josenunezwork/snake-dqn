// REST + WebSocket endpoints. In dev, Vite proxies these to the FastAPI backend.

import type {
  CheckpointInfo,
  DashboardData,
  LeaderboardData,
  PlayerResponse,
  RecentData,
  SubmitResponse,
} from "./types";

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/stream`;
}

// Single fetch wrapper: turns a non-2xx response into a thrown Error so every
// caller can `.catch` a real failure instead of stalling on an opaque rejection.
async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export async function fetchCheckpoints(): Promise<CheckpointInfo[]> {
  const data = await getJson<{ checkpoints?: CheckpointInfo[] }>("/api/checkpoints");
  return data.checkpoints ?? [];
}

export async function fetchMetrics(): Promise<DashboardData> {
  return getJson<DashboardData>("/api/metrics");
}

export async function fetchLeaderboard(limit = 10): Promise<LeaderboardData> {
  return getJson<LeaderboardData>(`/api/leaderboard?limit=${limit}`);
}

export async function fetchRecent(limit = 8): Promise<RecentData> {
  return getJson<RecentData>(`/api/scores/recent?limit=${limit}`);
}

export async function fetchPlayer(name: string): Promise<PlayerResponse> {
  return getJson<PlayerResponse>(`/api/players/${encodeURIComponent(name)}`);
}

// Server-authoritative: only the name is sent; the score comes from the
// session's finalized run so it cannot be spoofed by the client.
export async function submitScore(name: string): Promise<SubmitResponse> {
  return getJson<SubmitResponse>("/api/scores", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}
