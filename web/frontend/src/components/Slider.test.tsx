import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import Slider from "./Slider";

describe("Slider", () => {
  it("associates the label with the input via htmlFor/id", () => {
    render(
      <Slider label="Speed" value={12} min={1} max={60} format={(v) => `${v} fps`} onChange={vi.fn()} />
    );
    // getByLabelText only resolves when the <label htmlFor> points at the input id.
    const input = screen.getByLabelText("Speed");
    expect(input).toHaveAttribute("type", "range");
  });

  it("formats and displays the current value", () => {
    render(
      <Slider label="Speed" value={30} min={1} max={60} format={(v) => `${v} fps`} onChange={vi.fn()} />
    );
    expect(screen.getByText("30 fps")).toBeInTheDocument();
  });

  it("fires onChange with the numeric value", () => {
    const onChange = vi.fn();
    render(
      <Slider label="Food" value={0} min={0} max={10} format={(v) => `${v}`} onChange={onChange} />
    );
    const input = screen.getByLabelText("Food") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "5" } });
    expect(onChange).toHaveBeenCalledWith(5);
  });
});
