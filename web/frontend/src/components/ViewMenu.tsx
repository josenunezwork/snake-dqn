import { useEffect, useRef, useState } from "react";
import { ACCENTS, type ViewSettings } from "../hooks/useViewSettings";

interface Props {
  settings: ViewSettings;
  update: (patch: Partial<ViewSettings>) => void;
}

const TOGGLES: { key: keyof ViewSettings; label: string }[] = [
  { key: "labels", label: "Snake name labels" },
  { key: "crown", label: "Leader crown" },
  { key: "grid", label: "Grid" },
  { key: "trails", label: "Boost trails" },
  { key: "vignette", label: "Vignette" },
];

// Topbar "View" popover: accent + light/dark theme + arena display toggles.
// Persists via useViewSettings; closes on outside-click / Esc.
export default function ViewMenu({ settings, update }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="view-menu" ref={ref}>
      <button
        className={"pill pill-btn" + (open ? " on" : "")}
        onClick={() => setOpen((v) => !v)}
        aria-label="View settings"
        aria-expanded={open}
        title="View settings"
      >
        ⚙
      </button>
      {open && (
        <div className="view-pop" role="dialog" aria-label="View settings">
          <div className="view-row">
            <span className="view-label">Theme</span>
            <div className="seg view-seg">
              <button
                className={"btn" + (settings.theme === "dark" ? " active" : "")}
                onClick={() => update({ theme: "dark" })}
              >
                Dark
              </button>
              <button
                className={"btn" + (settings.theme === "light" ? " active" : "")}
                onClick={() => update({ theme: "light" })}
              >
                Light
              </button>
            </div>
          </div>

          <div className="view-row">
            <span className="view-label">Accent</span>
            <div className="accent-swatches">
              {ACCENTS.map((a) => (
                <button
                  key={a.value}
                  className={"accent-swatch" + (settings.accent === a.value ? " sel" : "")}
                  style={{ background: a.value }}
                  aria-label={a.name}
                  aria-pressed={settings.accent === a.value}
                  title={a.name}
                  onClick={() => update({ accent: a.value })}
                />
              ))}
            </div>
          </div>

          <div className="view-divider" />

          {TOGGLES.map((t) => (
            <button
              key={t.key}
              className="view-toggle"
              role="switch"
              aria-checked={!!settings[t.key]}
              onClick={() => update({ [t.key]: !settings[t.key] } as Partial<ViewSettings>)}
            >
              <span>{t.label}</span>
              <span className={"switch" + (settings[t.key] ? " on" : "")}>
                <span className="switch-knob" />
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
