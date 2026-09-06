import { useCallback, useEffect, useState } from "react";

// Persisted display preferences. Accent + theme drive CSS custom properties on
// <html> (so the whole UI follows); the boolean flags are read by the canvas
// renderer. Stored as one JSON blob under `snake_view`.
export interface ViewSettings {
  accent: string;
  theme: "dark" | "light";
  // True once the user explicitly picks a theme; until then the theme follows
  // the OS prefers-color-scheme (including live changes).
  themeChosen?: boolean;
  grid: boolean;
  labels: boolean; // floating snake name labels
  crown: boolean; // crown on the current leader
  trails: boolean; // boost speed trails
  vignette: boolean; // arena edge darkening
}

export const ACCENTS: { name: string; value: string }[] = [
  { name: "Cyan", value: "#38bdf8" },
  { name: "Violet", value: "#a78bfa" },
  { name: "Emerald", value: "#34d399" },
  { name: "Amber", value: "#fbbf24" },
  { name: "Rose", value: "#fb7185" },
];

const DEFAULTS: ViewSettings = {
  accent: "#38bdf8",
  theme: "dark",
  themeChosen: false,
  grid: true,
  // Names on by default: color alone can't identify a snake (colorblind users,
  // similar hues) — the labels already draw with a high-contrast stroke.
  labels: true,
  crown: true,
  trails: true,
  vignette: true,
};

const KEY = "snake_view";

function osTheme(): "dark" | "light" {
  try {
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function load(): ViewSettings {
  const defaults = { ...DEFAULTS, theme: osTheme() };
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Partial<ViewSettings>;
    const merged = { ...defaults, ...parsed };
    if (parsed.themeChosen === undefined) {
      // Legacy blob (pre-themeChosen): "light" was never the default, so a
      // stored "light" is an explicit choice; a stored "dark" follows the OS.
      merged.themeChosen = parsed.theme === "light";
      if (!merged.themeChosen) merged.theme = defaults.theme;
    }
    return merged;
  } catch {
    return defaults;
  }
}

export function useViewSettings() {
  const [settings, setSettings] = useState<ViewSettings>(load);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(settings));
    } catch {
      /* storage full / disabled — non-fatal */
    }
  }, [settings]);

  // Reflect accent + theme onto the document so all CSS-var-driven surfaces update.
  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty("--accent", settings.accent);
    root.setAttribute("data-theme", settings.theme);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", settings.theme === "light" ? "#eef2f7" : "#0b0f17");
  }, [settings.accent, settings.theme]);

  // Until the user explicitly picks a theme, track the OS preference live.
  useEffect(() => {
    if (settings.themeChosen) return;
    let mq: MediaQueryList | undefined;
    try {
      mq = window.matchMedia?.("(prefers-color-scheme: light)");
    } catch {
      return;
    }
    if (!mq?.addEventListener) return;
    const onChange = (e: MediaQueryListEvent) =>
      setSettings((s) => (s.themeChosen ? s : { ...s, theme: e.matches ? "light" : "dark" }));
    mq.addEventListener("change", onChange);
    return () => mq?.removeEventListener("change", onChange);
  }, [settings.themeChosen]);

  const update = useCallback((patch: Partial<ViewSettings>) => {
    // An explicit theme pick (View menu) pins the theme against the OS default.
    setSettings((s) => ({ ...s, ...patch, ...(patch.theme ? { themeChosen: true } : null) }));
  }, []);

  return [settings, update] as const;
}
