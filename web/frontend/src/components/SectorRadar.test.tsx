import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SectorRadar from "./SectorRadar";

const zeros = (n: number) => Array.from({ length: n }, () => 0);

describe("SectorRadar", () => {
  it("draws one value wedge per non-zero sector and skips empties", () => {
    const values = zeros(16);
    values[8] = 1; // right
    values[4] = 0.5; // up
    const { container } = render(
      <SectorRadar label="Food" color={[245, 158, 11]} values={values} />
    );
    // value wedges carry a <title> "sector i: v"; empty sectors render nothing.
    const titles = container.querySelectorAll("path title");
    expect(titles.length).toBe(2);
    // background grid still covers all 16 sectors so empties read as a ring.
    expect(container.querySelectorAll("path.radar-bg").length).toBe(16);
  });

  it("summarises the total in the accessible label", () => {
    const values = zeros(16);
    values[0] = 0.25;
    values[1] = 0.75;
    const { getByRole } = render(
      <SectorRadar label="Danger" color={[248, 113, 113]} values={values} />
    );
    expect(getByRole("img").getAttribute("aria-label")).toContain("1.0 across 16 sectors");
  });

  it("renders the heading arrow only when a direction is one-hot set", () => {
    const values = zeros(16);
    const withHeading = render(
      <SectorRadar label="Food" color={[1, 2, 3]} values={values} heading={[0, 1, 0, 0]} />
    );
    expect(withHeading.container.querySelector(".radar-head")).toBeTruthy();

    const noHeading = render(
      <SectorRadar label="Food" color={[1, 2, 3]} values={values} heading={[0, 0, 0, 0]} />
    );
    expect(noHeading.container.querySelector(".radar-head")).toBeNull();
  });
});
