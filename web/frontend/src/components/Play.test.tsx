import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

afterEach(() => {
  vi.clearAllMocks();
  // Tests that fake a coarse pointer stub matchMedia; drop it between tests.
  delete (window as { matchMedia?: unknown }).matchMedia;
});

describe("Play component", () => {
  it("shows an explicit Start run button when play is inactive (no auto-entry)", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Play play={null} send={send} speed={12} />);
    expect(screen.getByText(/race the ai/i)).toBeInTheDocument();
    // Nothing was sent just by rendering the tab.
    expect(send).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /start run/i }));
    expect(send).toHaveBeenCalledWith("set_mode", "play");
  });

  it("routes Start run through onStartRun when provided", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    const onStartRun = vi.fn();
    render(<Play play={null} send={send} speed={12} onStartRun={onStartRun} />);
    await user.click(screen.getByRole("button", { name: /start run/i }));
    expect(onStartRun).toHaveBeenCalled();
    expect(send).not.toHaveBeenCalled();
  });

  it("shows the session error instead of hanging on a spinner", () => {
    render(<Play play={null} send={noop} speed={12} error="Cannot enter play mode: bad checkpoint" />);
    expect(screen.getByText(/cannot enter play mode/i)).toBeInTheDocument();
    expect(screen.queryByText(/switching to play mode/i)).not.toBeInTheDocument();
    // The Start run button stays usable for a retry.
    expect(screen.getByRole("button", { name: /start run/i })).toBeEnabled();
  });

  it("offers a visible End run control while play is active", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Play play={playState()} send={send} speed={12} />);
    await user.click(screen.getByRole("button", { name: /end run/i }));
    expect(send).toHaveBeenCalledWith("set_mode", "watch");
  });

  it("lets the player pick the opponent count before starting", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Play play={playState({ run_started: false, opponents: 5 })} send={send} speed={12} />);
    await user.click(screen.getByRole("button", { name: /8 AI opponents/i }));
    expect(send).toHaveBeenCalledWith("set_play_opponents", 8);
  });

  it("prompts to start before the first input and blanks score/time until then", () => {
    render(<Play play={playState({ run_started: false, score: 10 })} send={noop} speed={12} />);
    expect(screen.getByText(/press an arrow key to start/i)).toBeInTheDocument();
    // score = length*10 would read "10" before any input — show "—" instead.
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("10.0s")).not.toBeInTheDocument();
  });

  it("teaches the boost cost, not just availability", () => {
    render(<Play play={playState()} send={noop} speed={12} />);
    expect(screen.getAllByText(/burns 1 segment \/ 3 frames/i).length).toBeGreaterThanOrEqual(1);
  });

  it("renders a touch d-pad on coarse-pointer devices that steers and boosts", () => {
    (window as { matchMedia?: unknown }).matchMedia = vi
      .fn()
      .mockReturnValue({ matches: true });
    const send = vi.fn();
    render(<Play play={playState({ run_started: false })} send={send} speed={12} />);
    fireEvent.pointerDown(screen.getByRole("button", { name: /steer up/i }));
    expect(send).toHaveBeenCalledWith("human_input", "up");
    const boost = screen.getByRole("button", { name: /hold to boost/i });
    fireEvent.pointerDown(boost);
    expect(send).toHaveBeenCalledWith("human_boost", true);
    fireEvent.pointerUp(boost);
    expect(send).toHaveBeenCalledWith("human_boost", false);
  });

  it("hides the touch d-pad for fine pointers", () => {
    render(<Play play={playState()} send={noop} speed={12} />);
    expect(screen.queryByRole("button", { name: /steer up/i })).not.toBeInTheDocument();
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

  it("submits the normalized name with the client id and shows placement feedback", async () => {
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
    // client_id travels inside api.ts (withClientId), not as an argument.
    expect(submitScore).toHaveBeenCalledWith("Ada");
    await waitFor(() => expect(screen.getByText(/new #1/i)).toBeInTheDocument());
  });

  it("collapses internal whitespace in the name before submitting", async () => {
    const user = userEvent.setup();
    vi.mocked(submitScore).mockResolvedValue({
      ok: true,
      result: { player_name: "jo se", score: 74, length: 4, food_eaten: 3, kills: 0 },
      leaderboard: [],
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
    await user.type(screen.getByLabelText(/your name/i), "jo  se");
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(submitScore).toHaveBeenCalledWith("jo se");
    await waitFor(() => expect(localStorage.getItem("snake_player")).toBe("jo se"));
  });

  it("remains playable when browser storage throws during name load and save", async () => {
    const user = userEvent.setup();
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    vi.mocked(submitScore).mockResolvedValue({
      ok: true,
      result: { player_name: "Ada", score: 74, length: 4, food_eaten: 3, kills: 0 },
      leaderboard: [],
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
    await waitFor(() => expect(screen.getByText(/recorded/i)).toBeInTheDocument());
    getItem.mockRestore();
    setItem.mockRestore();
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

  it("keeps the Submit button usable after a failed submit (retry)", async () => {
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
    await waitFor(() => expect(screen.getByText(/network/i)).toBeInTheDocument());
    // The run was NOT claimed: the form and Submit stay for a retry.
    const submit = screen.getByRole("button", { name: /submit/i });
    expect(submit).toBeEnabled();
    vi.mocked(submitScore).mockResolvedValue({
      ok: true,
      result: { player_name: "Bo", score: 10, length: 2, food_eaten: 0, kills: 0 },
      leaderboard: [],
    });
    await user.click(submit);
    await waitFor(() => expect(screen.getByText(/recorded/i)).toBeInTheDocument());
  });

  it("shows a Retry control when the leaderboard fetch fails", async () => {
    vi.mocked(fetchLeaderboard).mockRejectedValueOnce(new Error("down"));
    render(<Play play={playState()} send={noop} speed={12} />);
    expect(await screen.findByText(/leaderboard unavailable/i)).toBeInTheDocument();
    const user = userEvent.setup();
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
      stats: null,
    });
    await user.click(screen.getByRole("button", { name: /retry/i }));
    expect(await screen.findByText("Ada")).toBeInTheDocument();
  });

  it("highlights the current player's own leaderboard row by name fallback", async () => {
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

  it("prefers the client id over the display name for the me-highlight", async () => {
    localStorage.setItem("snake.clientId", "cid-mine");
    localStorage.setItem("snake_player", "alex");
    vi.mocked(fetchLeaderboard).mockResolvedValue({
      leaderboard: [
        // Same display name but a DIFFERENT browser: must not light up as "me".
        {
          rank: 1,
          player_name: "alex",
          client_id: "cid-other",
          score: 99,
          length: 5,
          food_eaten: 4,
          kills: 1,
          frames: 200,
          created_at: "t",
        },
        // Different display name but THIS browser's id: is "me".
        {
          rank: 2,
          player_name: "alex the second",
          client_id: "cid-mine",
          score: 50,
          length: 3,
          food_eaten: 2,
          kills: 0,
          frames: 100,
          created_at: "t",
        },
      ] as never,
      stats: null,
    });
    render(<Play play={playState()} send={noop} speed={12} />);
    const other = await screen.findByText("alex");
    expect(other.closest("tr")).not.toHaveClass("me");
    const mine = screen.getByText("alex the second");
    expect(mine.closest("tr")).toHaveClass("me");
  });
});
