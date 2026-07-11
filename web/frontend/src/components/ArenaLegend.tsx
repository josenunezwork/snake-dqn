// A pixel-legend for the arena's visual language, opened from a hover control on
// the stage. Small hand-drawn SVG swatches keep it accurate to what the canvas
// renderer actually paints.

interface Item {
  swatch: React.ReactNode;
  label: string;
  desc: string;
}

const S = 26;

const ITEMS: Item[] = [
  {
    label: "Snake",
    desc: "Body ribbon, bright head → dim tail. Eyes point where it's heading.",
    swatch: (
      <svg viewBox="0 0 40 26" width="40" height={S}>
        <defs>
          <linearGradient id="lg-body" x1="0" x2="1">
            <stop offset="0" stopColor="#7dd3fc" />
            <stop offset="1" stopColor="#1d5f80" />
          </linearGradient>
        </defs>
        <path d="M4 20 Q16 6 36 8" fill="none" stroke="url(#lg-body)" strokeWidth="7" strokeLinecap="round" />
        <circle cx="35" cy="8" r="3" fill="#0b1220" />
        <circle cx="35" cy="8" r="1.4" fill="#f8fbff" />
      </svg>
    ),
  },
  {
    label: "Hero ring",
    desc: "The snake you're inspecting (Inspector / Network follow it).",
    swatch: (
      <svg viewBox="0 0 40 26" width="40" height={S}>
        <circle cx="20" cy="13" r="8" fill="#1d5f80" />
        <circle cx="20" cy="13" r="10" fill="none" stroke="#e6edf6" strokeWidth="1.6" />
      </svg>
    ),
  },
  {
    label: "Boost trail",
    desc: "A snake spending length for extra speed leaves a glowing wake.",
    swatch: (
      <svg viewBox="0 0 40 26" width="40" height={S}>
        <path d="M4 13 H34" stroke="#34d399" strokeWidth="12" strokeLinecap="round" opacity="0.25" />
        <path d="M4 13 H34" stroke="#7dffcf" strokeWidth="5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    label: "Food",
    desc: "Glowing pellets. Eat them to grow longer and score.",
    swatch: (
      <svg viewBox="0 0 40 26" width="40" height={S}>
        <circle cx="20" cy="13" r="10" fill="#f59e0b" opacity="0.25" />
        <circle cx="20" cy="13" r="4.5" fill="#ffe28c" />
      </svg>
    ),
  },
  {
    label: "Intent cones",
    desc: "Hero's candidate moves (L/S/R). Green = safe, red = dangerous; the pick is brightest. Toggle with O.",
    swatch: (
      <svg viewBox="0 0 40 26" width="40" height={S}>
        <path d="M20 13 L8 4 L10 13 Z" fill="#34d399" opacity="0.7" />
        <path d="M20 13 L20 2 L24 4 Z" fill="#fbbf24" opacity="0.5" />
        <path d="M20 13 L33 5 L31 13 Z" fill="#f87171" opacity="0.5" />
      </svg>
    ),
  },
];

export default function ArenaLegend({ onClose }: { onClose: () => void }) {
  return (
    <div className="legend-pop" role="dialog" aria-label="Arena legend" onClick={(e) => e.stopPropagation()}>
      <div className="legend-pop-head">
        <span className="section-title" style={{ margin: 0 }}>
          What am I looking at?
        </span>
        <button className="toast-x" onClick={onClose} aria-label="Close legend">
          ✕
        </button>
      </div>
      {ITEMS.map((it) => (
        <div className="legend-item" key={it.label}>
          <span className="legend-swatch">{it.swatch}</span>
          <span className="legend-text">
            <b>{it.label}</b>
            <span className="muted">{it.desc}</span>
          </span>
        </div>
      ))}
    </div>
  );
}
