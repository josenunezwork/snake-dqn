import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Frame } from "./types";

const socket = vi.fn();

vi.mock("./hooks/useGameSocket", () => ({ useGameSocket: () => socket() }));
vi.mock("./hooks/useViewSettings", () => ({
  useViewSettings: () => [{ density: "comfortable" }, vi.fn()],
}));
vi.mock("./api", () => ({ fetchCheckpoints: vi.fn().mockResolvedValue([]) }));
vi.mock("./components/GameCanvas", () => ({ default: () => null }));
vi.mock("./components/Inspector", () => ({ default: () => null }));
vi.mock("./components/EgoRasterViewer", () => ({ default: () => null }));
vi.mock("./components/NetworkVisualizer", () => ({ default: () => null }));
vi.mock("./components/Dashboard", () => ({ default: () => null }));
vi.mock("./components/Controls", () => ({ default: () => null }));
vi.mock("./components/Play", () => ({ default: () => null }));
vi.mock("./components/TelemetryBand", () => ({ default: () => null }));
vi.mock("./components/HelpOverlay", () => ({ default: () => null }));
vi.mock("./components/CommandPalette", () => ({ default: () => null }));
vi.mock("./components/WelcomeTour", () => ({ default: () => null }));
vi.mock("./components/ViewMenu", () => ({ default: () => null }));
vi.mock("./components/Toast", () => ({ default: () => null }));

import App from "./App";

function frame(scope: string | null = "watch_opponents"): Frame {
  return {
    type: "frame",
    frame: 5,
    arena: { width: 800, height: 600, segment: 10, wall: 10, arena_type: "rectangular" },
    snakes: [],
    food: [],
    stats: { frame: 5, alive: 6, food_eaten: 0, deaths: 0, kills: 0, best_length: 1, loss: null, epsilon: 0 },
    session: {
      playing: true,
      speed: 12,
      mode: "watch",
      training: false,
      hero_id: 0,
      checkpoint: "b8_s3_t7_20261016.pth",
      config: "free_space_v2.yaml",
      input_size: 61,
      num_snakes: 6,
      epsilon: 0,
      food_target: 1,
      food_count: 1,
      error: null,
      watch_roster: scope === "watch_opponents" ? "mixed" : "all_b8",
      safety_veto: scope
        ? {
            active: true,
            scope,
            wrapped_snake_ids: [1, 2],
            counters: {
              "1": { decisions: 8, vetoes_applied: 2 },
              "2": { decisions: 5, vetoes_applied: 1 },
            },
          }
        : null,
    },
    play: null,
    inspector: null,
    netviz: null,
  };
}

afterEach(() => vi.clearAllMocks());

describe("mixed Watch veto counters", () => {
  it("shows opponent-only total and per-opponent veto counters", () => {
    socket.mockReturnValue({ frame: frame(), connected: true, link: "live", history: [], send: vi.fn() });
    render(<App />);
    expect(screen.getByText(/opponents v8.*3 vetoes.*13 decisions/i)).toBeInTheDocument();
    expect(screen.getByText(/#1 2\/8/)).toBeInTheDocument();
    expect(screen.getByText(/#2 1\/5/)).toBeInTheDocument();
    expect(screen.queryByText(/#0/)).not.toBeInTheDocument();
  });

  it("does not show the mixed counter pill outside opponent scope", () => {
    socket.mockReturnValue({ frame: frame("watch_hero"), connected: true, link: "live", history: [], send: vi.fn() });
    render(<App />);
    expect(screen.queryByText(/opponents v8/i)).not.toBeInTheDocument();
  });

  it("labels a failed mixed request as a Watch roster fallback", () => {
    const failed = frame();
    failed.session.watch_roster = "all_b8";
    failed.session.watch_roster_reason = "FRP3 pin failed";
    if (failed.session.safety_veto) failed.session.safety_veto.active = false;
    socket.mockReturnValue({ frame: failed, connected: true, link: "live", history: [], send: vi.fn() });
    render(<App />);
    expect(screen.getByText("Watch roster fallback")).toBeInTheDocument();
    expect(screen.queryByText(/opponents v8/i)).not.toBeInTheDocument();
  });
});

describe("Path B assisted-hero identity", () => {
  it("shows the literal assisted label while the enabled controller falls back", () => {
    const assisted = frame();
    assisted.session.pathb = {
      active: true,
      label: "Path B advisory controller enabled",
      counts: { applied: 0, fallback: 4, deadline_miss: 4, stale_discarded: 0 },
      last: { proposed_action: 1, executed_action: 1, fallback_reason: "deadline_miss" },
    };
    socket.mockReturnValue({ frame: assisted, connected: true, link: "live", history: [], send: vi.fn() });
    render(<App />);
    expect(screen.getByText("assisted hero")).toBeInTheDocument();
  });

  it("hides the assisted label when the flag is off or absent", () => {
    socket.mockReturnValue({ frame: frame(), connected: true, link: "live", history: [], send: vi.fn() });
    render(<App />);
    expect(screen.queryByText("assisted hero")).not.toBeInTheDocument();
  });
});
