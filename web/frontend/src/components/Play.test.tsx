import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PlayState } from "../types";

// Mock the REST layer so the component's effects and submit flow are exercised
// without a network.
vi.mock("../api", () => ({
  fetchLeaderboard: vi.fn().mockResolvedValue({ leaderboard: [], stats: null }),
  fetchRecent: vi.fn().mockResolvedValue({ recent: [] }),
  fetchPlayer: vi.fn().mockResolvedValue({ found: false, player: null }),
  submitScore: vi.fn(),
}));

import { fetchLeaderboard, submitScore } from "../api";
import Play from "./Play";

function playState(over: Partial<PlayState> = {}): PlayState {
  return {
    active: true,
    human_id: 0,
    opponents: 5,
    human_alive: true,
    length: 3,
    food_eaten: 1,
    kills: 0,
    frames: 120,
    score: 40,
    run_started: true,
    run_over: false,
    submitted: false,
    pending: null,
    ...over,
  };
}

const noop = () => {};

beforeEach(() => {
  localStorage.clear();
  vi.mocked(fetchLeaderboard).mockResolvedValue({ leaderboard: [], stats: null });
  vi.mocked(submitScore).mockReset();
});

afterEach(() => vi.clearAllMocks());

describe("Play component", () => {
  it("shows a switching message when play is inactive", () => {
    render(<Play play={null} send={noop} speed={12} />);
    expect(screen.getByText(/switching to play mode/i)).toBeInTheDocument();
  });

  it("shows the session error instead of hanging on the spinner", () => {
    render(<Play play={null} send={noop} speed={12} error="Cannot enter play mode: bad checkpoint" />);
    expect(screen.getByText(/cannot enter play mode/i)).toBeInTheDocument();
    expect(screen.queryByText(/switching to play mode/i)).not.toBeInTheDocument();
  });

  it("lets the player pick the opponent count before starting", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Play play={playState({ run_started: false, opponents: 5 })} send={send} speed={12} />);
    await user.click(screen.getByRole("button", { name: /8 AI opponents/i }));
    expect(send).toHaveBeenCalledWith("set_play_opponents", 8);
  });

  it("prompts to start before the first input", () => {
    render(<Play play={playState({ run_started: false })} send={noop} speed={12} />);
    expect(screen.getByText(/press an arrow key to start/i)).toBeInTheDocument();
  });

  it("shows the live HUD including time while alive", () => {
    render(<Play play={playState({ frames: 120 })} send={noop} speed={12} />);
    expect(screen.getByText("Time")).toBeInTheDocument();
    // 120 frames / 12 fps = 10.0s
    expect(screen.getByText("10.0s")).toBeInTheDocument();
    expect(screen.getByText(/you're alive/i)).toBeInTheDocument();
  });

  it("shows the game-over recap with duration when dead", () => {
    const play = playState({
      human_alive: false,
      run_over: true,
      pending: {
        score: 74,
        length: 4,
        food_eaten: 3,
        kills: 0,
        frames: 300,
        duration_seconds: 25,
        checkpoint: null,
      },
    });
    render(<Play play={play} send={noop} speed={12} />);
    expect(screen.getByText(/final score 74/i)).toBeInTheDocument();
    expect(screen.getByText(/survived 25\.0s/i)).toBeInTheDocument();
  });

  it("submits the name and shows placement feedback", async () => {
    const user = userEvent.setup();
    vi.mocked(submitScore).mockResolvedValue({
      ok: true,
      result: { player_name: "Ada", score: 74, length: 4, food_eaten: 3, kills: 0 },
      leaderboard: [
        {
          rank: 1,
          player_name: "Ada",
          score: 74,
          length: 4,
          food_eaten: 3,
          kills: 0,
          frames: 300,
          created_at: "t",
        },
      ],
    });
    const play = playState({
      human_alive: false,
      run_over: true,
      pending: {
        score: 74,
        length: 4,
        food_eaten: 3,
        kills: 0,
        frames: 300,
        duration_seconds: 25,
        checkpoint: null,
      },
    });
    render(<Play play={play} send={noop} speed={12} />);
    await user.type(screen.getByLabelText(/your name/i), "Ada");
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(submitScore).toHaveBeenCalledWith("Ada");
    await waitFor(() => expect(screen.getByText(/new #1/i)).toBeInTheDocument());
  });

  it("does not claim a rank when the run scored below the player's best", async () => {
    const user = userEvent.setup();
    // This run scored 10, but the leaderboard row for Ada is her earlier best (74).
    vi.mocked(submitScore).mockResolvedValue({
      ok: true,
      result: { player_name: "Ada", score: 10, length: 2, food_eaten: 0, kills: 0 },
      leaderboard: [
        {
          rank: 1,
          player_name: "Ada",
          score: 74,
          length: 4,
          food_eaten: 3,
          kills: 0,
          frames: 300,
          created_at: "t",
        },
      ],
    });
    const play = playState({
      human_alive: false,
      run_over: true,
      pending: {
        score: 10,
        length: 2,
        food_eaten: 0,
        kills: 0,
        frames: 10,
        duration_seconds: 1,
        checkpoint: null,
      },
    });
    render(<Play play={play} send={noop} speed={12} />);
    await user.type(screen.getByLabelText(/your name/i), "Ada");
    await user.click(screen.getByRole("button", { name: /submit/i }));
    // Must NOT show "New #1!" for a weaker run; shows the recorded-but-unranked message.
    await waitFor(() => expect(screen.getByText(/recorded/i)).toBeInTheDocument());
    expect(screen.queryByText(/new #1/i)).not.toBeInTheDocument();
  });

  it("surfaces a submit error instead of hanging", async () => {
    const user = userEvent.setup();
    vi.mocked(submitScore).mockRejectedValue(new Error("network"));
    const play = playState({
      human_alive: false,
      run_over: true,
      pending: {
        score: 10,
        length: 2,
        food_eaten: 0,
        kills: 0,
        frames: 10,
        duration_seconds: 1,
        checkpoint: null,
      },
    });
    render(<Play play={play} send={noop} speed={12} />);
    await user.type(screen.getByLabelText(/your name/i), "Bo");
    await user.click(screen.getByRole("button", { name: /submit/i }));
    await waitFor(() => expect(screen.getByText(/could not reach the server/i)).toBeInTheDocument());
  });

  it("highlights the current player's own leaderboard row", async () => {
    localStorage.setItem("snake_player", "Ada");
    vi.mocked(fetchLeaderboard).mockResolvedValue({
      leaderboard: [
        {
          rank: 1,
          player_name: "Ada",
          score: 99,
          length: 5,
          food_eaten: 4,
          kills: 1,
          frames: 200,
          created_at: "t",
        },
      ],
      stats: { total_players: 1, total_games: 1, best_score: 99, best_player: "Ada", average_score: 99 },
    });
    render(<Play play={playState()} send={noop} speed={12} />);
    const cell = await screen.findByText("Ada");
    expect(cell.closest("tr")).toHaveClass("me");
  });
});
