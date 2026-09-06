import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchCheckpoints,
  fetchLeaderboard,
  fetchPlayer,
  fetchRecent,
  getClientId,
  submitScore,
} from "./api";

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: `status ${status}`,
    json: async () => body,
  });
}

describe("api getJson error handling", () => {
  beforeEach(() => {
    localStorage.setItem("snake.clientId", "cid-test-1234");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.removeItem("snake.clientId");
  });

  it("returns parsed json on 200", async () => {
    vi.stubGlobal("fetch", mockFetch(200, { leaderboard: [], stats: null }));
    const data = await fetchLeaderboard();
    expect(data.leaderboard).toEqual([]);
  });

  it("throws the backend HTTPException detail from a non-2xx JSON body", async () => {
    vi.stubGlobal("fetch", mockFetch(500, { detail: "scores.db is locked" }));
    await expect(fetchLeaderboard()).rejects.toThrow("scores.db is locked");
  });

  it("throws a legacy {error} field when detail is absent", async () => {
    vi.stubGlobal("fetch", mockFetch(400, { error: "no finished run to submit" }));
    await expect(submitScore("Ada")).rejects.toThrow("no finished run to submit");
  });

  it("falls back to the status line when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "status 500",
        json: async () => {
          throw new Error("not json");
        },
      })
    );
    await expect(fetchLeaderboard()).rejects.toThrow(/500/);
  });

  it("reserves 'Could not reach the server' for genuine network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(fetchLeaderboard()).rejects.toThrow("Could not reach the server");
  });

  it("fetchCheckpoints tolerates a missing key", async () => {
    vi.stubGlobal("fetch", mockFetch(200, {}));
    expect(await fetchCheckpoints()).toEqual([]);
  });

  it("submitScore POSTs the normalized name and the client id", async () => {
    const f = mockFetch(200, { ok: true });
    vi.stubGlobal("fetch", f);
    await submitScore("  Ada   Lovelace ");
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/scores");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ name: "Ada Lovelace", client_id: "cid-test-1234" });
  });

  it("fetchPlayer sends the client id as a query param", async () => {
    const f = mockFetch(200, { found: false, player: null });
    vi.stubGlobal("fetch", f);
    await fetchPlayer("Ada");
    expect(f.mock.calls[0][0]).toBe("/api/players/Ada?client_id=cid-test-1234");
  });

  it("fetchLeaderboard appends the client id after the limit", async () => {
    const f = mockFetch(200, { leaderboard: [], stats: null });
    vi.stubGlobal("fetch", f);
    await fetchLeaderboard(3);
    expect(f.mock.calls[0][0]).toBe("/api/leaderboard?limit=3&client_id=cid-test-1234");
  });

  it("fetchRecent hits the recent endpoint with the limit", async () => {
    const f = mockFetch(200, { recent: [] });
    vi.stubGlobal("fetch", f);
    await fetchRecent(5);
    expect(f.mock.calls[0][0]).toContain("/api/scores/recent?limit=5");
  });
});

describe("getClientId", () => {
  afterEach(() => localStorage.removeItem("snake.clientId"));

  it("generates once and persists", () => {
    localStorage.removeItem("snake.clientId");
    const id = getClientId();
    expect(id).toBeTruthy();
    expect(id.length).toBeLessThanOrEqual(64);
    expect(getClientId()).toBe(id);
    expect(localStorage.getItem("snake.clientId")).toBe(id);
  });

  it("returns the stored id verbatim (capped at 64 chars)", () => {
    localStorage.setItem("snake.clientId", "x".repeat(100));
    expect(getClientId()).toBe("x".repeat(64));
  });
});
