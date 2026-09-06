import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SessionState, SnakeDTO } from "../types";

vi.mock("../api", () => ({
  fetchCheckpoints: vi.fn(),
}));

import { fetchCheckpoints } from "../api";
import Controls from "./Controls";

function sessionState(over: Partial<SessionState> = {}): SessionState {
  return {
    playing: true,
    speed: 12,
    mode: "watch",
    training: false,
    hero_id: 0,
    checkpoint: "champ.pth",
    config: "free_space_v2.yaml",
    input_size: 61,
    num_snakes: 6,
    epsilon: 0.05,
    food_target: 300,
    food_count: 450,
    error: null,
    ...over,
  };
}

const SNAKES: SnakeDTO[] = [
  {
    id: 0,
    color: [255, 0, 0],
    name: "Crimson",
    alive: true,
    boosting: false,
    length: 10,
    segments: [],
    head: [0, 0],
    is_hero: true,
  },
  {
    id: 1,
    color: [0, 0, 255],
    name: "Azure",
    alive: true,
    boosting: false,
    length: 7,
    segments: [],
    head: [0, 0],
    is_hero: false,
  },
];

const baseProps = {
  snakes: SNAKES,
  ghosts: false,
  onToggleGhosts: () => {},
};

beforeEach(() => {
  vi.mocked(fetchCheckpoints).mockResolvedValue([
    { name: "champ.pth", size_mb: 4.2, obs_spec: "vector61" } as never,
    { name: "latest_pqn.pth", size_mb: 6.1, obs_spec: "raster31v2" } as never,
  ]);
});

afterEach(() => vi.clearAllMocks());

describe("Controls", () => {
  it("browsing the model select does NOT load; the Load button commits", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Controls {...baseProps} session={sessionState()} send={send} />);
    const select = await screen.findByLabelText(/model checkpoint/i);
    await user.selectOptions(select, "latest_pqn.pth");
    expect(send).not.toHaveBeenCalledWith("load_checkpoint", expect.anything());
    await user.click(screen.getByRole("button", { name: /load selected checkpoint/i }));
    expect(send).toHaveBeenCalledWith("load_checkpoint", "latest_pqn.pth");
  });

  it("disables Load while the selection equals the loaded checkpoint", async () => {
    render(<Controls {...baseProps} session={sessionState()} send={vi.fn()} />);
    await screen.findByLabelText(/model checkpoint/i);
    expect(screen.getByRole("button", { name: /load selected checkpoint/i })).toBeDisabled();
  });

  it("routes checkpoint loads through onLoadCheckpoint when provided", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    const onLoadCheckpoint = vi.fn();
    render(
      <Controls
        {...baseProps}
        session={sessionState()}
        send={send}
        onLoadCheckpoint={onLoadCheckpoint}
      />
    );
    const select = await screen.findByLabelText(/model checkpoint/i);
    await user.selectOptions(select, "latest_pqn.pth");
    await user.click(screen.getByRole("button", { name: /load selected checkpoint/i }));
    expect(onLoadCheckpoint).toHaveBeenCalledWith("latest_pqn.pth");
    expect(send).not.toHaveBeenCalledWith("load_checkpoint", expect.anything());
  });

  it("shows obs_spec in the checkpoint options", async () => {
    render(<Controls {...baseProps} session={sessionState()} send={vi.fn()} />);
    expect(await screen.findByText(/latest_pqn\.pth \(6\.1 MB · raster31v2\)/)).toBeInTheDocument();
  });

  it("surfaces a checkpoint-list failure with a Retry, keeping the loaded model selectable", async () => {
    vi.mocked(fetchCheckpoints).mockRejectedValueOnce(new Error("down"));
    const user = userEvent.setup();
    render(<Controls {...baseProps} session={sessionState()} send={vi.fn()} />);
    expect(await screen.findByText(/couldn't fetch the checkpoint list/i)).toBeInTheDocument();
    // The select never renders blank: the loaded checkpoint is kept as an option.
    expect(screen.getByText(/champ\.pth \(loaded\)/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /retry/i }));
    await waitFor(() =>
      expect(screen.queryByText(/couldn't fetch the checkpoint list/i)).not.toBeInTheDocument()
    );
    expect(await screen.findByText(/latest_pqn\.pth/)).toBeInTheDocument();
  });

  it("presents Watch/Train/Play as one mode control routed through onModeChange", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    const onModeChange = vi.fn();
    render(
      <Controls {...baseProps} session={sessionState()} send={send} onModeChange={onModeChange} />
    );
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(onModeChange).toHaveBeenCalledWith("play");
    await user.click(screen.getByRole("button", { name: "Train" }));
    expect(onModeChange).toHaveBeenCalledWith("train");
    expect(send).not.toHaveBeenCalledWith("set_mode", expect.anything());
  });

  it("falls back to raw set_mode sends without the guard prop", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Controls {...baseProps} session={sessionState()} send={send} />);
    await user.click(screen.getByRole("button", { name: "Train" }));
    expect(send).toHaveBeenCalledWith("set_mode", "train");
  });

  it("has a Save weights button", async () => {
    const user = userEvent.setup();
    const send = vi.fn();
    render(<Controls {...baseProps} session={sessionState()} send={send} />);
    await user.click(screen.getByRole("button", { name: /save weights/i }));
    expect(send).toHaveBeenCalledWith("save_weights");
  });

  it("uses honest ARIA for the roster: toggle buttons, no listbox", async () => {
    const { container } = render(<Controls {...baseProps} session={sessionState()} send={vi.fn()} />);
    await screen.findByText(/latest_pqn\.pth/); // settle the checkpoint fetch
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    // No composite roles inside the roster — its rows are plain toggle buttons.
    const roster = container.querySelector(".roster")!;
    expect(roster.querySelectorAll("[role]")).toHaveLength(0);
    const hero = screen.getByRole("button", { name: /crimson/i });
    expect(hero).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /azure/i })).toHaveAttribute("aria-pressed", "false");
  });

  it("describes the raster input contract when serving a raster model", async () => {
    const session = {
      ...sessionState(),
      obs_spec: "raster31v2",
    } as SessionState;
    render(<Controls {...baseProps} session={session} send={vi.fn()} />);
    await screen.findByText(/latest_pqn\.pth/); // settle the checkpoint fetch
    expect(screen.getByText(/raster 31×31 \+ 25×25 \+ 26 scalars/)).toBeInTheDocument();
    expect(screen.getByText("raster31v2")).toBeInTheDocument();
    expect(screen.queryByText(/61-D vector/)).not.toBeInTheDocument();
  });

  it("formats the food slider as count · target with the cap-exemption note", async () => {
    render(<Controls {...baseProps} session={sessionState()} send={vi.fn()} />);
    await screen.findByText(/latest_pqn\.pth/); // settle the checkpoint fetch
    expect(screen.getByText("450 · target 300")).toBeInTheDocument();
    expect(screen.getByText(/corpse drops are cap-exempt/i)).toBeInTheDocument();
  });

  it("shows a visible Copy run summary button when App provides the handler", async () => {
    const user = userEvent.setup();
    const onCopySummary = vi.fn();
    render(
      <Controls {...baseProps} session={sessionState()} send={vi.fn()} onCopySummary={onCopySummary} />
    );
    await user.click(screen.getByRole("button", { name: /copy run summary/i }));
    expect(onCopySummary).toHaveBeenCalled();
  });
});
