import { useEffect, useRef, useState } from "react";

interface Row {
  keys: string[];
  desc: string;
}

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

const GLOBAL: Row[] = [
  { keys: [IS_MAC ? "⌘" : "Ctrl", "K"], desc: "Open the command palette" },
  { keys: ["1", "…", "5"], desc: "Speed presets (0.5× → Max)" },
  { keys: ["?"], desc: "Toggle this help" },
];
const WATCH: Row[] = [
  { keys: ["Space"], desc: "Play / pause the simulation" },
  { keys: ["R"], desc: "Reset the current game" },
  { keys: ["[", "]"], desc: "Cycle the inspected snake" },
  { keys: ["O"], desc: "Toggle the intent overlay on the arena" },
];
const PLAY: Row[] = [
  { keys: ["←", "↑", "↓", "→"], desc: "Steer your snake" },
  { keys: ["W", "A", "S", "D"], desc: "Steer (alternative)" },
  { keys: ["Space"], desc: "Hold to boost (needs length ≥ 5)" },
  { keys: ["P"], desc: "Pause / resume the run" },
  { keys: ["R", "Enter"], desc: "New game after you die" },
];
const TIPS = [
  "Click any snake in the arena to inspect it.",
  "The caption under the arena narrates the AI's live decision.",
  "The Inspector's steering dial + radar show what it senses and how it moves.",
  "The telemetry strip up top plots the run live.",
  "The Raster tab shows what the new conv model literally sees — load a raster31v2 checkpoint from Controls › Model to light it up.",
  "Switch to the Play tab to race the AI for the leaderboard.",
];

function Keys({ keys }: { keys: string[] }) {
  return (
    <span>
      {keys.map((k, i) => (
        <span key={k}>
          <span className="kbd">{k}</span>
          {i < keys.length - 1 ? " " : ""}
        </span>
      ))}
    </span>
  );
}

// Lightweight modal cheat-sheet. Esc / backdrop / the × button all close it
// (Esc is wired in App alongside the "?" toggle).
export default function HelpOverlay({
  onClose,
  onReplayTour,
}: {
  onClose: () => void;
  onReplayTour?: () => void;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [opener] = useState<HTMLElement | null>(() =>
    typeof document === "undefined" ? null : (document.activeElement as HTMLElement | null)
  );

  // aria-modal="true" asserts the rest of the page is inert, so focus has to live
  // inside the dialog for its whole lifetime and go back to the opener on close.
  // The opener can be gone by then (help can be launched from the palette).
  // Focus lands on the card, not the ✕: this is a cheat-sheet to read, and a
  // focused button would make Space close it — a key App otherwise swallows here.
  useEffect(() => {
    dialogRef.current?.focus();
    return () => {
      if (opener?.isConnected) opener.focus();
    };
  }, [opener]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key !== "Tab") return;
    const items = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []);
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === dialogRef.current)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div
        className="overlay-card"
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="overlay-head">
          <h2>Keyboard shortcuts</h2>
          <button className="btn" onClick={onClose} aria-label="Close help">
            ✕
          </button>
        </div>

        <div className="section-title">Anywhere</div>
        {GLOBAL.map((r) => (
          <div className="help-row" key={r.desc}>
            <Keys keys={r.keys} />
            <span className="muted">{r.desc}</span>
          </div>
        ))}

        <div className="section-title" style={{ marginTop: 12 }}>
          Watch &amp; train
        </div>
        {WATCH.map((r) => (
          <div className="help-row" key={r.desc}>
            <Keys keys={r.keys} />
            <span className="muted">{r.desc}</span>
          </div>
        ))}

        <div className="section-title" style={{ marginTop: 12 }}>
          Play mode
        </div>
        {PLAY.map((r) => (
          <div className="help-row" key={r.desc}>
            <Keys keys={r.keys} />
            <span className="muted">{r.desc}</span>
          </div>
        ))}

        <div className="section-title" style={{ marginTop: 12 }}>
          Tips
        </div>
        <ul className="help-tips">
          {TIPS.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>

        {onReplayTour && (
          <button className="btn" style={{ marginTop: 12, width: "100%" }} onClick={onReplayTour}>
            ↻ Replay the welcome tour
          </button>
        )}
      </div>
    </div>
  );
}
