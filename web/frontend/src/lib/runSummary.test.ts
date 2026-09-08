import { describe, expect, it } from "vitest";
import { runSummary } from "./runSummary";
import type { Frame } from "../types";

function frame(over: Partial<Frame>): Frame {
  return {
    type: "frame",
    frame: 100,
    arena: { width: 800, height: 600, segment: 10, wall: 5, arena_type: "rect" },
    snakes: [],
    food: [],
    stats: {
      alive: 4,
      frame: 100,
      food_eaten: 12,
      deaths: 3,
      kills: 1,
      best_length: 27,
      loss: 0.0123,
      epsilon: 0.05,
    },
    session: {
      playing: true,
      speed: 12,
      mode: "watch",
      training: false,
      hero_id: 0,
      checkpoint: "champion.pth",
      config: "free_space_v2.yaml",
      input_size: 61,
      num_snakes: 6,
      epsilon: 0.05,
      food_target: 300,
      food_count: 280,
      error: null,
    },
    play: null,
    inspector: null,
    netviz: null,
    ...over,
  };
}

describe("runSummary", () => {
  it("handles a null frame", () => {
    expect(runSummary(null)).toContain("not connected");
  });

  it("summarises a watch run with stats + session", () => {
    const out = runSummary(frame({}));
    expect(out).toContain("watch run");
    expect(out).toContain("best length: 27");
    expect(out).toContain("loss: 0.0123");
    expect(out).toContain("champion.pth");
    expect(out).toContain("61-D");
  });

  it("summarises a Play run from play state", () => {
    const out = runSummary(
      frame({
        play: {
          active: true,
          human_id: 0,
          opponents: 5,
          human_alive: true,
          length: 9,
          food_eaten: 4,
          kills: 2,
          frames: 210,
          score: 88,
          run_started: true,
          run_over: false,
          submitted: false,
          pending: null,
        },
      })
    );
    expect(out).toContain("Play run");
    expect(out).toContain("score: 88");
    expect(out).not.toContain("best length");
  });
});
