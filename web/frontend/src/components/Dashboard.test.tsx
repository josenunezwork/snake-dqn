import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DashboardData } from "../types";

vi.mock("../api", () => ({
  fetchMetrics: vi.fn(),
}));

import { fetchMetrics } from "../api";
import Dashboard from "./Dashboard";

function metrics(over: Partial<DashboardData> = {}): DashboardData {
  return {
    checkpoints: [{ name: "champ.pth", size_mb: 4.2 }],
    leaderboard: [
      {
        file: "eval_new.json",
        candidate: "latest_pqn.pth",
        opponent: "scripted mix",
        frames: 3000,
        n: 10,
        mean_mass: 42.5,
        max_mass: 80,
        survival: 0.91,
        kills: 0.2,
        metric: "mass_integral",
        date: "2026-07-18T10:00:00",
      } as never,
      {
        file: "eval_old.json",
        candidate: "champ.pth",
        opponent: "best_apex_pre_fs.pth",
        frames: 1500,
        n: 5,
        mean_mass: 60.1,
        max_mass: 90,
        survival: 0.853,
        kills: 0.1,
        metric: "legacy_mean_mass",
        date: "2026-06-20T10:00:00",
      } as never,
    ],
    ...over,
  };
}

beforeEach(() => {
  vi.mocked(fetchMetrics).mockResolvedValue(metrics());
});

afterEach(() => vi.clearAllMocks());

describe("Dashboard", () => {
  it("shows the loading skeleton first, not the empty-state flash", async () => {
    const { container } = render(<Dashboard />);
    expect(container.querySelector(".skeleton")).toBeInTheDocument();
    expect(screen.queryByText(/no metrics yet/i)).not.toBeInTheDocument();
    await screen.findByText("latest_pqn.pth"); // settle the fetch
  });

  it("renders opponent and date columns and per-row metric badges", async () => {
    render(<Dashboard />);
    expect(await screen.findByText("scripted mix")).toBeInTheDocument();
    expect(screen.getByText("best_apex_pre_fs.pth")).toBeInTheDocument();
    expect(screen.getByText("2026-07-18")).toBeInTheDocument();
    expect(screen.getByText("mass integral")).toBeInTheDocument();
    // Legacy rows are honestly labelled as scored by the old gate.
    expect(screen.getByText("old gate metric")).toBeInTheDocument();
    // Column header for the opponent is visible (not tooltip-only).
    expect(screen.getByRole("columnheader", { name: /opponent/i })).toBeInTheDocument();
  });

  it("keeps the populate guidance in the true empty state (fresh install)", async () => {
    vi.mocked(fetchMetrics).mockResolvedValue(metrics({ leaderboard: [] }));
    render(<Dashboard />);
    expect(
      await screen.findByText(/run a tournament eval to populate the leaderboard/i)
    ).toBeInTheDocument();
  });

  it("shows a visible warning when Refresh fails after data has loaded, keeping stale data", async () => {
    const user = userEvent.setup();
    render(<Dashboard />);
    expect(await screen.findByText("latest_pqn.pth")).toBeInTheDocument();
    vi.mocked(fetchMetrics).mockRejectedValueOnce(new Error("down"));
    await user.click(screen.getByRole("button", { name: /refresh/i }));
    expect(await screen.findByText(/refresh failed — showing previous results/i)).toBeInTheDocument();
    // The previous table is still there (stale but visible).
    expect(screen.getByText("latest_pqn.pth")).toBeInTheDocument();
    // A successful refresh clears the warning.
    await user.click(screen.getByRole("button", { name: /refresh/i }));
    await waitFor(() =>
      expect(screen.queryByText(/refresh failed/i)).not.toBeInTheDocument()
    );
  });

  it("explains the abbreviated metric headers", async () => {
    render(<Dashboard />);
    const surv = await screen.findByRole("columnheader", { name: /surv/i });
    expect(surv).toHaveAttribute("title", expect.stringMatching(/fraction of eval frames/i));
    const mass = screen.getByRole("columnheader", { name: /mass/i });
    expect(mass).toHaveAttribute("title", expect.stringMatching(/mean per-frame/i));
  });
});
