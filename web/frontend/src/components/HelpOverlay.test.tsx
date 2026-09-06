import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import HelpOverlay from "./HelpOverlay";

/** Mounts the dialog with a real, focused opener button outside it, as App does. */
function renderWithOpener(props: Partial<React.ComponentProps<typeof HelpOverlay>> = {}) {
  const opener = document.createElement("button");
  opener.textContent = "opener";
  document.body.appendChild(opener);
  opener.focus();
  const view = render(<HelpOverlay onClose={props.onClose ?? (() => {})} {...props} />);
  return { ...view, opener, cleanupOpener: () => opener.remove() };
}

const dialog = () => screen.getByRole("dialog", { name: /keyboard shortcuts/i });

/** Tab stops inside the dialog only — `screen` would also match the opener outside it. */
const dialogButtons = () => within(dialog()).getAllByRole("button");

describe("HelpOverlay focus contract", () => {
  it("moves focus into the dialog on open", () => {
    const { cleanupOpener } = renderWithOpener();
    expect(dialog()).toHaveAttribute("aria-modal", "true");
    expect(document.activeElement).toBe(dialog());
    cleanupOpener();
  });

  it("does not park focus on a button, so Space cannot dismiss the cheat sheet", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { cleanupOpener } = renderWithOpener({ onClose });
    await user.keyboard(" ");
    expect(onClose).not.toHaveBeenCalled();
    cleanupOpener();
  });

  it("restores focus to the opener when it closes", () => {
    const { unmount, opener, cleanupOpener } = renderWithOpener();
    expect(document.activeElement).not.toBe(opener);
    unmount();
    expect(document.activeElement).toBe(opener);
    cleanupOpener();
  });

  it("does not throw when the opener unmounted while the dialog was up", () => {
    const { unmount, opener, cleanupOpener } = renderWithOpener();
    opener.remove();
    expect(() => unmount()).not.toThrow();
    cleanupOpener();
  });

  it("wraps Tab from the last focusable back to the first", async () => {
    const user = userEvent.setup();
    const { cleanupOpener } = renderWithOpener({ onReplayTour: () => {} });
    const buttons = dialogButtons();
    const first = buttons[0];
    const last = buttons[buttons.length - 1];
    expect(first).not.toBe(last);

    last.focus();
    await user.tab();
    expect(document.activeElement).toBe(first);
    cleanupOpener();
  });

  it("wraps Shift+Tab from the first focusable back to the last", async () => {
    const user = userEvent.setup();
    const { cleanupOpener } = renderWithOpener({ onReplayTour: () => {} });
    const buttons = dialogButtons();
    const first = buttons[0];
    const last = buttons[buttons.length - 1];

    first.focus();
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(last);
    cleanupOpener();
  });

  it("keeps Tab inside the dialog rather than reaching background content", async () => {
    const user = userEvent.setup();
    const { cleanupOpener } = renderWithOpener({ onReplayTour: () => {} });
    for (let i = 0; i < 6; i++) {
      await user.tab();
      expect(dialog().contains(document.activeElement)).toBe(true);
    }
    cleanupOpener();
  });

  it("closes on Escape from inside the dialog", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { cleanupOpener } = renderWithOpener({ onClose });
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
    cleanupOpener();
  });

  it("documents keys per the keyboard contract: Space/R/[ ]/O are watch & train, P pauses in play", () => {
    const { cleanupOpener } = renderWithOpener();
    const d = dialog();

    // Three accurate sections.
    expect(within(d).getByText("Anywhere")).toBeInTheDocument();
    expect(within(d).getByText("Watch & train")).toBeInTheDocument();
    expect(within(d).getByText("Play mode")).toBeInTheDocument();

    // Section membership: rows between "Anywhere" and "Watch & train" must not
    // claim Space (in play mode Space is boost, not pause).
    const text = d.textContent ?? "";
    const anywhere = text.slice(text.indexOf("Anywhere"), text.indexOf("Watch & train"));
    expect(anywhere).not.toContain("Space");

    const watch = text.slice(text.indexOf("Watch & train"), text.indexOf("Play mode"));
    expect(watch).toContain("Play / pause the simulation");
    expect(watch).toContain("Cycle the inspected snake");
    expect(watch).toContain("Toggle the intent overlay");

    const play = text.slice(text.indexOf("Play mode"));
    expect(play).toContain("Hold to boost");
    expect(play).toContain("Pause / resume the run");
    cleanupOpener();
  });

  it("mentions the Raster tab in the tips", () => {
    const { cleanupOpener } = renderWithOpener();
    expect(within(dialog()).getByText(/Raster tab/i)).toBeInTheDocument();
    cleanupOpener();
  });

  it("still closes via the close button and the backdrop", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container, cleanupOpener } = renderWithOpener({ onClose });
    await user.click(screen.getByRole("button", { name: /close help/i }));
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.click(container.querySelector(".overlay-backdrop") as HTMLElement);
    expect(onClose).toHaveBeenCalledTimes(2);

    // Clicking inside the card must not close it.
    await user.click(screen.getByText(/keyboard shortcuts/i, { selector: "h2" }));
    expect(onClose).toHaveBeenCalledTimes(2);
    cleanupOpener();
  });
});
