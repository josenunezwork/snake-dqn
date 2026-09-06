import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import WelcomeTour from "./WelcomeTour";

/** Mounts the tour with a real, focused opener button outside it, as App does. */
function renderWithOpener(onClose: () => void = () => {}) {
  const opener = document.createElement("button");
  opener.textContent = "opener";
  document.body.appendChild(opener);
  opener.focus();
  const view = render(<WelcomeTour onClose={onClose} />);
  return { ...view, opener, cleanupOpener: () => opener.remove() };
}

const welcomeDialog = () => screen.getByRole("dialog", { name: /welcome to snake-dqn/i });

beforeEach(() => {
  localStorage.clear();
});

describe("WelcomeTour welcome card (modal contract)", () => {
  it("is a named aria-modal dialog and receives focus on open", () => {
    const { cleanupOpener } = renderWithOpener();
    expect(welcomeDialog()).toHaveAttribute("aria-modal", "true");
    expect(document.activeElement).toBe(welcomeDialog());
    cleanupOpener();
  });

  it("starts the tour on Enter from the card itself", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { cleanupOpener } = renderWithOpener(onClose);
    await user.keyboard("{Enter}");
    expect(screen.getByText("The arena")).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(localStorage.getItem("snake_onboarded")).toBeNull();
    cleanupOpener();
  });

  it("lets Enter activate a focused button instead of hijacking it to advance", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { cleanupOpener } = renderWithOpener(onClose);
    // Tab: card -> "Take a quick tour" -> "Explore on my own"
    await user.tab();
    await user.tab();
    expect(document.activeElement).toHaveTextContent(/explore on my own/i);
    await user.keyboard("{Enter}");
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem("snake_onboarded")).toBe("1");
    cleanupOpener();
  });

  it("traps Tab inside the card rather than reaching background content", async () => {
    const user = userEvent.setup();
    const { cleanupOpener } = renderWithOpener();
    for (let i = 0; i < 5; i++) {
      await user.tab();
      expect(welcomeDialog().contains(document.activeElement)).toBe(true);
    }
    cleanupOpener();
  });

  it("closes on Escape (marking onboarded) and restores focus to the opener", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { unmount, opener, cleanupOpener } = renderWithOpener(onClose);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem("snake_onboarded")).toBe("1");
    unmount();
    expect(document.activeElement).toBe(opener);
    cleanupOpener();
  });
});

describe("WelcomeTour spotlight steps", () => {
  it("moves focus to the tooltip on each step and keeps Tab trapped", async () => {
    const user = userEvent.setup();
    const { container, cleanupOpener } = renderWithOpener();
    await user.click(screen.getByRole("button", { name: /take a quick tour/i }));
    const tip = container.querySelector(".tour-tip") as HTMLElement;
    expect(document.activeElement).toBe(tip);
    for (let i = 0; i < 4; i++) {
      await user.tab();
      expect(tip.contains(document.activeElement)).toBe(true);
    }
    cleanupOpener();
  });

  it("says six panels and introduces the Raster tab", async () => {
    const user = userEvent.setup();
    const { cleanupOpener } = renderWithOpener();
    await user.keyboard("{Enter}"); // welcome card -> step 0
    await user.keyboard("{Enter}"); // -> step 1
    await user.keyboard("{Enter}"); // -> step 2 (the tab strip)
    expect(screen.getByText("Six panels")).toBeInTheDocument();
    expect(screen.getByText(/Raster shows what the new conv model literally sees/)).toBeInTheDocument();
    cleanupOpener();
  });

  it("advances (not dismisses) when the fallback veil is clicked with the target missing", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container, cleanupOpener } = renderWithOpener(onClose);
    await user.keyboard("{Enter}"); // start the tour; jsdom has no .stage, so the veil renders
    expect(screen.getByText("The arena")).toBeInTheDocument();
    const veil = container.querySelector(".tour-veil") as HTMLElement;
    expect(veil).not.toBeNull();
    await user.click(veil);
    expect(screen.getByText("Live telemetry")).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(localStorage.getItem("snake_onboarded")).toBeNull();
    cleanupOpener();
  });

  it("finishes from the last step via Enter and marks onboarded", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { cleanupOpener } = renderWithOpener(onClose);
    await user.keyboard("{Enter}"); // -> step 0
    await user.keyboard("{Enter}"); // -> step 1
    await user.keyboard("{Enter}"); // -> step 2
    await user.keyboard("{Enter}"); // -> step 3 (last)
    expect(onClose).not.toHaveBeenCalled();
    await user.keyboard("{Enter}"); // done
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem("snake_onboarded")).toBe("1");
    cleanupOpener();
  });
});
