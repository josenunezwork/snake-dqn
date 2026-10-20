import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import Inspector from "./Inspector";
import type { InspectorDTO } from "../types";

const inspector: InspectorDTO = {
  input_size: 61,
  state: Array(61).fill(0),
  groups: [],
  q_values: [0, 1, 2, 3, 4, 5],
  action_labels: ["left", "straight", "right", "boost left", "boost straight", "boost right"],
  chosen: 1,
  free_space: null,
};

afterEach(cleanup);

describe("Path B inspector identity", () => {
  it("shows intervention counters and distinguishes proposed from executed action", () => {
    render(
      <Inspector
        inspector={inspector}
        pathb={{
          active: true,
          label: "Path B advisory controller enabled",
          counts: { applied: 3, fallback: 5, deadline_miss: 2, stale_discarded: 1 },
          last: { proposed_action: 2, executed_action: 1, fallback_reason: "deadline_miss" },
        }}
      />,
    );

    expect(screen.getByLabelText("Assisted hero status")).toHaveTextContent("Interventions 3");
    expect(screen.getByLabelText("Assisted hero status")).toHaveTextContent("fallbacks 5");
    expect(screen.getByLabelText("Assisted hero status")).toHaveTextContent("Proposed action: right");
    expect(screen.getByLabelText("Assisted hero status")).toHaveTextContent("executed action: straight");
    expect(screen.getByLabelText("Assisted hero status")).toHaveTextContent("fallback: deadline_miss");
  });

  it("does not render controller details while Path B is inactive", () => {
    render(<Inspector inspector={inspector} pathb={{ active: false, label: "", counts: {} }} />);
    expect(screen.queryByLabelText("Assisted hero status")).not.toBeInTheDocument();
  });
});
