import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TelemetryBand from "./TelemetryBand";
import type { MetricSample } from "../types";

const sample = (over: Partial<MetricSample>): MetricSample => ({
  frame: 0,
  loss: null,
  epsilon: 0,
  foodEaten: 0,
  alive: 0,
  bestLength: 0,
  kills: 0,
  ...over,
});

describe("TelemetryBand", () => {
  it("renders nothing until there are at least two samples", () => {
    const { container } = render(<TelemetryBand history={[sample({ frame: 1 })]} />);
    expect(container.querySelector(".telemetry")).toBeNull();
  });

  it("renders a tile per metric with the latest value", () => {
    const history = [
      sample({ frame: 1, bestLength: 10, alive: 6, foodEaten: 100, kills: 2 }),
      sample({ frame: 2, bestLength: 42, alive: 5, foodEaten: 250, kills: 3 }),
    ];
    const { container, getByText } = render(<TelemetryBand history={history} />);
    expect(container.querySelectorAll(".tele-tile").length).toBe(6);
    expect(getByText("42")).toBeInTheDocument(); // best length
    expect(getByText("250")).toBeInTheDocument(); // food eaten (< 1000, not compacted)
  });

  it("shows an em dash for loss when it is never present", () => {
    const history = [sample({ frame: 1 }), sample({ frame: 2 })];
    const { getAllByText } = render(<TelemetryBand history={history} />);
    // loss (and, given zeros, nothing else forces it) renders the placeholder
    expect(getAllByText("—").length).toBeGreaterThanOrEqual(1);
  });

  it("compacts large food counts to k", () => {
    const history = [
      sample({ frame: 1, foodEaten: 900 }),
      sample({ frame: 2, foodEaten: 4200 }),
    ];
    const { getByText } = render(<TelemetryBand history={history} />);
    expect(getByText("4.2k")).toBeInTheDocument();
  });
});
