interface Row {
  keys: string[];
  desc: string;
}

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

const GLOBAL: Row[] = [
  { keys: [IS_MAC ? "⌘" : "Ctrl", "K"], desc: "Open the command palette" },
  { keys: ["Space"], desc: "Play / pause the simulation" },
  { keys: ["R"], desc: "Reset the current game" },
  { keys: ["1", "…", "5"], desc: "Speed presets (0.5× → Max)" },
  { keys: ["[", "]"], desc: "Cycle the inspected snake" },
  { keys: ["O"], desc: "Toggle the intent overlay on the arena" },
  { keys: ["?"], desc: "Toggle this help" },
];
const PLAY: Row[] = [
  { keys: ["←", "↑", "↓", "→"], desc: "Steer your snake" },
  { keys: ["W", "A", "S", "D"], desc: "Steer (alternative)" },
  { keys: ["Space"], desc: "Hold to boost (needs length ≥ 5)" },
  { keys: ["R", "Enter"], desc: "New game after you die" },
];
const TIPS = [
  "Click any snake in the arena to inspect it.",
  "The caption under the arena narrates the AI's live decision.",
  "The Inspector's steering dial + radar show what it senses and how it moves.",
  "The telemetry strip up top plots the run live.",
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
  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div
        className="overlay-card"
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
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
