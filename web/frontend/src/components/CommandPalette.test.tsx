import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import CommandPalette, { type Command } from "./CommandPalette";

// jsdom has no layout, so it does not implement scrollIntoView — the palette calls
// it on every selection change.
beforeAll(() => {
  if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {};
});

const noop = () => {};

function cmds(): Command[] {
  return [
    { id: "play", title: "Play", group: "Sim", run: vi.fn() },
    { id: "pause", title: "Pause", group: "Sim", run: vi.fn() },
    { id: "reset", title: "Reset game", group: "Sim", run: vi.fn() },
  ];
}

const input = () => screen.getByRole("combobox", { name: /search commands/i });

/** The id the input points assistive tech at, and the option that is aria-selected. */
function activeDescendant() {
  return input().getAttribute("aria-activedescendant");
}
function selectedOption() {
  return screen.getAllByRole("option").find((o) => o.getAttribute("aria-selected") === "true");
}

describe("CommandPalette accessibility contract", () => {
  it("wires the input as a combobox controlling the listbox", () => {
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    const listbox = screen.getByRole("listbox");
    expect(listbox).toHaveAttribute("id", "palette-listbox");
    expect(input()).toHaveAttribute("aria-controls", "palette-listbox");
    expect(input()).toHaveAttribute("aria-autocomplete", "list");
    expect(input()).toHaveAttribute("aria-expanded", "true");
  });

  it("points aria-activedescendant at the highlighted option on mount", () => {
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    expect(activeDescendant()).toBe("palette-opt-0");
    expect(selectedOption()).toHaveAttribute("id", "palette-opt-0");
  });

  it("advances aria-activedescendant with ArrowDown / ArrowUp", () => {
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    fireEvent.keyDown(input(), { key: "ArrowDown" });
    expect(activeDescendant()).toBe("palette-opt-1");
    expect(selectedOption()).toHaveAttribute("id", "palette-opt-1");

    fireEvent.keyDown(input(), { key: "ArrowUp" });
    expect(activeDescendant()).toBe("palette-opt-0");
    expect(selectedOption()).toHaveAttribute("id", "palette-opt-0");
  });

  it("keeps aria-activedescendant in sync with Home / End and the list bounds", () => {
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    fireEvent.keyDown(input(), { key: "End" });
    expect(activeDescendant()).toBe("palette-opt-2");
    // Already at the end: must not dangle past the last option.
    fireEvent.keyDown(input(), { key: "ArrowDown" });
    expect(activeDescendant()).toBe("palette-opt-2");

    fireEvent.keyDown(input(), { key: "Home" });
    expect(activeDescendant()).toBe("palette-opt-0");
    fireEvent.keyDown(input(), { key: "ArrowUp" });
    expect(activeDescendant()).toBe("palette-opt-0");
  });

  it("never references an id that is not in the document", async () => {
    const user = userEvent.setup();
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    await user.type(input(), "zzzznope");

    expect(screen.queryAllByRole("option")).toHaveLength(0);
    expect(screen.getByText(/no matching command/i)).toBeInTheDocument();
    // A dangling aria-activedescendant is itself an a11y error, so it must be absent.
    expect(activeDescendant()).toBeNull();
    expect(input()).toHaveAttribute("aria-expanded", "false");
  });

  it("keeps the filtered list's activedescendant on a real option", async () => {
    const user = userEvent.setup();
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    await user.type(input(), "reset");
    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(activeDescendant()).toBe("palette-opt-0");
    expect(document.getElementById(activeDescendant()!)).toHaveTextContent("Reset game");
  });

  it("keeps focus on the input and options out of the tab order", async () => {
    const user = userEvent.setup();
    render(<CommandPalette commands={cmds()} onClose={noop} />);
    expect(document.activeElement).toBe(input());
    for (const opt of screen.getAllByRole("option")) {
      expect(opt).toHaveAttribute("tabindex", "-1");
    }
    // Tab must not escape into the content aria-modal declares inert.
    await user.tab();
    expect(document.activeElement).toBe(input());
  });

  it("returns focus to the opener when it closes", () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const { unmount } = render(<CommandPalette commands={cmds()} onClose={noop} />);
    expect(document.activeElement).toBe(input());

    unmount();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  it("still runs the highlighted command on Enter", () => {
    const commands = cmds();
    const onClose = vi.fn();
    render(<CommandPalette commands={commands} onClose={onClose} />);
    fireEvent.keyDown(input(), { key: "ArrowDown" });
    fireEvent.keyDown(input(), { key: "Enter" });
    expect(commands[1].run).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
