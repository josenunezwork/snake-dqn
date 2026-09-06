import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useGameSocket } from "./useGameSocket";

type Handler = ((ev: unknown) => void) | null;

// Stands in for the browser WebSocket at the global boundary: the hook is
// exercised for real, only the wire is faked.
class MockWS {
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  static instances: MockWS[] = [];

  static last(): MockWS {
    const ws = MockWS.instances[MockWS.instances.length - 1];
    if (!ws) throw new Error("no socket constructed");
    return ws;
  }

  readyState = 0;
  sent: string[] = [];
  onopen: Handler = null;
  onclose: Handler = null;
  onerror: Handler = null;
  onmessage: Handler = null;
  url: string;

  constructor(url: string) {
    this.url = url;
    MockWS.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  // A real close() also delivers onclose; the hook's unmount guard depends on
  // tolerating that, so model it rather than silently swallowing the event.
  close() {
    if (this.readyState === MockWS.CLOSED) return;
    this.readyState = MockWS.CLOSED;
    this.onclose?.({});
  }
}

function openLast() {
  const ws = MockWS.last();
  act(() => {
    ws.readyState = MockWS.OPEN;
    ws.onopen?.({});
  });
}

function dropLast() {
  const ws = MockWS.last();
  act(() => ws.close());
}

function emit(msg: unknown) {
  const ws = MockWS.last();
  act(() => ws.onmessage?.({ data: JSON.stringify(msg) }));
}

function emitRaw(data: string) {
  const ws = MockWS.last();
  act(() => ws.onmessage?.({ data }));
}

function advance(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

// snake_case stats keys mirror the backend payload that `accumulate` reads.
function mkFrame(n: number, playing = true, checkpoint: string | null = "a") {
  return {
    type: "frame",
    frame: n,
    session: { playing, checkpoint },
    stats: {
      frame: n,
      alive: 3,
      food_eaten: n,
      deaths: 0,
      kills: 1,
      best_length: 10 + n,
      loss: 0.5,
      epsilon: 0.1,
    },
  };
}

describe("useGameSocket", () => {
  beforeEach(() => {
    MockWS.instances = [];
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", MockWS);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("connects to the stream endpoint on mount", () => {
    renderHook(() => useGameSocket());
    expect(MockWS.instances).toHaveLength(1);
    expect(MockWS.last().url).toMatch(/^ws:\/\/.+\/ws\/stream$/);
  });

  it("reports offline before the first connect and live once open", () => {
    const { result } = renderHook(() => useGameSocket());
    expect(result.current.link).toBe("offline");
    expect(result.current.connected).toBe(false);

    openLast();
    expect(result.current.link).toBe("live");
    expect(result.current.connected).toBe(true);
  });

  it("reports reconnecting (not offline) after a drop, and retries", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    dropLast();

    expect(result.current.connected).toBe(false);
    expect(result.current.link).toBe("reconnecting");

    advance(999);
    expect(MockWS.instances).toHaveLength(1);
    advance(1);
    expect(MockWS.instances).toHaveLength(2);

    openLast();
    expect(result.current.link).toBe("live");
  });

  it("closes the socket on error so the retry path runs", () => {
    renderHook(() => useGameSocket());
    openLast();
    const ws = MockWS.last();
    act(() => ws.onerror?.({}));

    expect(ws.readyState).toBe(MockWS.CLOSED);
    advance(1000);
    expect(MockWS.instances).toHaveLength(2);
  });

  it("goes stale after STALE_MS of silence while the engine is playing", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1));
    expect(result.current.link).toBe("live");

    advance(2000);
    expect(result.current.link).toBe("live");

    advance(1000);
    expect(result.current.link).toBe("stale");
  });

  it("recovers to live when frames resume", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1));
    advance(3000);
    expect(result.current.link).toBe("stale");

    emit(mkFrame(2));
    expect(result.current.link).toBe("live");
  });

  it("never calls a legitimately paused engine stale", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1, false));

    advance(5000);
    expect(result.current.link).toBe("live");
  });

  it("bounds history at the cap and drops the oldest samples", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    for (let i = 1; i <= 300; i++) emit(mkFrame(i));

    const history = result.current.history;
    expect(history).toHaveLength(240);
    expect(history[0].frame).toBe(61);
    expect(history[history.length - 1].frame).toBe(300);
  });

  it("maps backend stats onto metric samples", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(7));

    expect(result.current.history[0]).toEqual({
      frame: 7,
      loss: 0.5,
      epsilon: 0.1,
      foodEaten: 7,
      alive: 3,
      bestLength: 17,
      kills: 1,
    });
  });

  it("clears history when the checkpoint changes", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    for (let i = 1; i <= 5; i++) emit(mkFrame(i, true, "a"));
    expect(result.current.history).toHaveLength(5);

    emit(mkFrame(6, true, "b"));
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].frame).toBe(6);
  });

  it("skips history for paused heartbeat frames but still updates the frame", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1));
    expect(result.current.history).toHaveLength(1);

    emit({ ...mkFrame(2), paused: true });
    expect(result.current.history).toHaveLength(1);
    expect(result.current.frame?.frame).toBe(2);
  });

  it("skips history when the frame counter has not advanced", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(3));
    emit(mkFrame(3));
    expect(result.current.history).toHaveLength(1);

    emit(mkFrame(4));
    expect(result.current.history).toHaveLength(2);
  });

  it("changes the history array identity only when a sample is appended", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1));
    const h1 = result.current.history;

    emit({ ...mkFrame(2), paused: true }); // no append
    expect(result.current.history).toBe(h1);

    emit(mkFrame(2)); // append
    expect(result.current.history).not.toBe(h1);
  });

  it("surfaces server error messages as a notice without touching the frame", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1));
    emit({ type: "error", message: "engine wedged" });

    expect(result.current.notice).toEqual({ tone: "error", message: "engine wedged", seq: 1 });
    expect(result.current.frame?.frame).toBe(1);
  });

  it("dedupes the per-tick spam of one repeated error, but not new errors", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit({ type: "error", message: "boom" });
    const first = result.current.notice;
    emit({ type: "error", message: "boom" });
    emit({ type: "error", message: "boom" });
    expect(result.current.notice).toBe(first);
    expect(result.current.notice?.seq).toBe(1);

    emit({ type: "error", message: "different boom" });
    expect(result.current.notice?.seq).toBe(2);
    expect(result.current.notice?.message).toBe("different boom");
  });

  it("always surfaces info confirmations, even identical repeats", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit({ type: "info", message: "Saved web_train_x.pth" });
    expect(result.current.notice).toEqual({
      tone: "info",
      message: "Saved web_train_x.pth",
      seq: 1,
    });

    emit({ type: "info", message: "Saved web_train_x.pth" });
    expect(result.current.notice?.seq).toBe(2);
  });

  it("ignores malformed and non-frame messages without clobbering the last frame", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    emit(mkFrame(1));

    expect(() => emitRaw("not json{")).not.toThrow();
    emit({ type: "pong" });

    expect(result.current.frame?.frame).toBe(1);
    expect(result.current.history).toHaveLength(1);
  });

  it("drops control messages while the socket is not open", () => {
    const { result } = renderHook(() => useGameSocket());
    act(() => result.current.send("pause"));
    expect(MockWS.last().sent).toEqual([]);
  });

  it("sends control messages once open", () => {
    const { result } = renderHook(() => useGameSocket());
    openLast();
    act(() => result.current.send("set_speed", 3));

    expect(MockWS.last().sent).toHaveLength(1);
    expect(JSON.parse(MockWS.last().sent[0])).toEqual({
      type: "control",
      action: "set_speed",
      value: 3,
    });
  });

  it("closes the socket and leaks no timers on unmount", () => {
    const { unmount } = renderHook(() => useGameSocket());
    openLast();
    unmount();

    expect(MockWS.instances[0].readyState).toBe(MockWS.CLOSED);
    advance(5000);
    expect(MockWS.instances).toHaveLength(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("schedules no reconnect after unmount with a retry already pending", () => {
    const { unmount } = renderHook(() => useGameSocket());
    openLast();
    dropLast();
    unmount();

    advance(5000);
    expect(MockWS.instances).toHaveLength(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});
