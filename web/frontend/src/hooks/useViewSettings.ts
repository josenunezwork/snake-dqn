import { useCallback, useEffect, useState } from "react";

// Persisted display preferences. Accent + theme drive CSS custom properties on
// <html> (so the whole UI follows); the boolean flags are read by the canvas
// renderer. Stored as one JSON blob under `snake_view`.
export interface ViewSettings {
  accent: string;
  theme: "dark" | "light";
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
  grid: true,
  labels: false,
  crown: true,
  trails: true,
  vignette: true,
};

const KEY = "snake_view";

function load(): ViewSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
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

  const update = useCallback((patch: Partial<ViewSettings>) => {
    setSettings((s) => ({ ...s, ...patch }));
  }, []);

  return [settings, update] as const;
}
