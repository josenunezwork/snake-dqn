import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchCheckpoints, fetchLeaderboard, fetchRecent, submitScore } from "./api";

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: `status ${status}`,
    json: async () => body,
  });
}

describe("api getJson error handling", () => {
  afterEach(() => vi.restoreAllMocks());

  it("returns parsed json on 200", async () => {
    vi.stubGlobal("fetch", mockFetch(200, { leaderboard: [], stats: null }));
    const data = await fetchLeaderboard();
    expect(data.leaderboard).toEqual([]);
  });

  it("throws on a non-2xx response instead of returning junk", async () => {
    vi.stubGlobal("fetch", mockFetch(500, { detail: "boom" }));
    await expect(fetchLeaderboard()).rejects.toThrow(/500/);
  });

  it("fetchCheckpoints tolerates a missing key", async () => {
    vi.stubGlobal("fetch", mockFetch(200, {}));
    expect(await fetchCheckpoints()).toEqual([]);
  });

  it("submitScore POSTs the name as a JSON body", async () => {
    const f = mockFetch(200, { ok: true });
    vi.stubGlobal("fetch", f);
    await submitScore("Ada");
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/scores");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ name: "Ada" });
  });

  it("fetchRecent hits the recent endpoint with the limit", async () => {
    const f = mockFetch(200, { recent: [] });
    vi.stubGlobal("fetch", f);
    await fetchRecent(5);
    expect(f.mock.calls[0][0]).toContain("/api/scores/recent?limit=5");
  });

  it("fetchLeaderboard passes the limit through", async () => {
    const f = mockFetch(200, { leaderboard: [], stats: null });
    vi.stubGlobal("fetch", f);
    await fetchLeaderboard(3);
    expect(f.mock.calls[0][0]).toContain("limit=3");
  });
});
