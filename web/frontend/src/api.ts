// REST + WebSocket endpoints. In dev, Vite proxies these to the FastAPI backend.

import type {
  CheckpointInfo,
  DashboardData,
  LeaderboardData,
  PlayerResponse,
  RecentData,
  SubmitResponse,
} from "./types";
import { getClientId, normalizePlayerName } from "./identity";

export { getClientId } from "./identity";

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/stream`;
}

// Single fetch wrapper. A genuine network failure (fetch rejection) throws the
// reserved "Could not reach the server"; a served non-2xx throws the backend's
// HTTPException `detail` (or `error`) message so the UI shows the real reason
// instead of a bare status line.
async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new Error("Could not reach the server");
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body: unknown = await res.json();
      const detail =
        body && typeof body === "object"
          ? ((body as { detail?: unknown; error?: unknown }).detail ??
            (body as { error?: unknown }).error)
          : undefined;
      if (typeof detail === "string" && detail) message = detail;
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

function withClientId(url: string): string {
  const id = getClientId();
  if (!id) return url;
  return url + (url.includes("?") ? "&" : "?") + `client_id=${encodeURIComponent(id)}`;
}

export async function fetchCheckpoints(): Promise<CheckpointInfo[]> {
  const data = await getJson<{ checkpoints?: CheckpointInfo[] }>("/api/checkpoints");
  return data.checkpoints ?? [];
}

export async function fetchMetrics(): Promise<DashboardData> {
  return getJson<DashboardData>("/api/metrics");
}

export async function fetchLeaderboard(limit = 10): Promise<LeaderboardData> {
  return getJson<LeaderboardData>(withClientId(`/api/leaderboard?limit=${limit}`));
}

export async function fetchRecent(limit = 8): Promise<RecentData> {
  return getJson<RecentData>(`/api/scores/recent?limit=${limit}`);
}

export async function fetchPlayer(name: string): Promise<PlayerResponse> {
  return getJson<PlayerResponse>(withClientId(`/api/players/${encodeURIComponent(name)}`));
}

// Server-authoritative: only the name (+ opaque client id) is sent; the score
// comes from the session's finalized run so it cannot be spoofed by the client.
// The name is normalized (trim + collapse internal whitespace) to match the
// server-side normalization.
export async function submitScore(name: string): Promise<SubmitResponse> {
  const clean = normalizePlayerName(name);
  const clientId = getClientId();
  return getJson<SubmitResponse>("/api/scores", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(clientId ? { name: clean, client_id: clientId } : { name: clean }),
  });
}
