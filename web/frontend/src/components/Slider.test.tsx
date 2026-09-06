import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Slider from "./Slider";

describe("Slider", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

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

  it("fires onChange with the numeric value (trailing-edge throttled)", () => {
    const onChange = vi.fn();
    render(
      <Slider label="Food" value={0} min={0} max={10} format={(v) => `${v}`} onChange={onChange} />
    );
    const input = screen.getByLabelText("Food") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "5" } });
    expect(onChange).not.toHaveBeenCalled(); // coalesced, not per-step
    vi.advanceTimersByTime(40);
    expect(onChange).toHaveBeenCalledWith(5);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("coalesces a fast sweep into one send with the latest value", () => {
    const onChange = vi.fn();
    render(
      <Slider label="Epsilon" value={0} min={0} max={100} format={(v) => `${v}`} onChange={onChange} />
    );
    const input = screen.getByLabelText("Epsilon") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "10" } });
    fireEvent.change(input, { target: { value: "20" } });
    fireEvent.change(input, { target: { value: "30" } });
    vi.advanceTimersByTime(40);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(30);
  });

  it("flushes the pending send immediately on release", () => {
    const onChange = vi.fn();
    render(
      <Slider label="Fine" value={12} min={1} max={60} format={(v) => `${v} fps`} onChange={onChange} />
    );
    const input = screen.getByLabelText("Fine") as HTMLInputElement;
    fireEvent.pointerDown(input);
    fireEvent.change(input, { target: { value: "30" } });
    fireEvent.pointerUp(input); // release before the 30ms trailing send
    expect(onChange).toHaveBeenCalledWith(30);
    expect(onChange).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(100); // no duplicate from the (cleared) timer
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("does not snap back to the stale prop on release; converges on the echo", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <Slider label="Fine" value={12} min={1} max={60} format={(v) => `${v} fps`} onChange={onChange} />
    );
    const input = screen.getByLabelText("Fine") as HTMLInputElement;
    fireEvent.pointerDown(input);
    fireEvent.change(input, { target: { value: "3" } });
    fireEvent.pointerUp(input);
    // The prop is still the stale pre-drag value for up to a full frame period.
    rerender(
      <Slider label="Fine" value={12} min={1} max={60} format={(v) => `${v} fps`} onChange={onChange} />
    );
    expect(screen.getByText("3 fps")).toBeInTheDocument(); // no snap-back
    // The server echoes what we set: adopt it.
    rerender(
      <Slider label="Fine" value={3} min={1} max={60} format={(v) => `${v} fps`} onChange={onChange} />
    );
    expect(screen.getByText("3 fps")).toBeInTheDocument();
    // A later genuine server-side change is adopted normally.
    rerender(
      <Slider label="Fine" value={45} min={1} max={60} format={(v) => `${v} fps`} onChange={onChange} />
    );
    expect(screen.getByText("45 fps")).toBeInTheDocument();
  });

  it("adopts a server-side clamp that differs from the sent value", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <Slider label="Food" value={100} min={0} max={600} format={(v) => `${v}`} onChange={onChange} />
    );
    const input = screen.getByLabelText("Food") as HTMLInputElement;
    fireEvent.pointerDown(input);
    fireEvent.change(input, { target: { value: "600" } });
    fireEvent.pointerUp(input);
    // Server clamps to 500 and echoes that — anything off the stale baseline wins.
    rerender(
      <Slider label="Food" value={500} min={0} max={600} format={(v) => `${v}`} onChange={onChange} />
    );
    expect(screen.getByText("500")).toBeInTheDocument();
  });
});
