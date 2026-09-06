import { useRef, useState } from "react";
import { GLOSSARY } from "../lib/glossary";

// Width must match the .infodot-pop CSS width so clamping is accurate.
const POP_WIDTH = 230;
const GAP = 8;

interface PopPos {
  left: number;
  top: number;
  below: boolean;
}

// A tiny, dependency-free "ⓘ" affordance that reveals a glossary blurb on hover
// or keyboard focus. The popover renders position:fixed at coordinates measured
// from the trigger, so the scrolling panel's overflow can never clip it; it
// opens downward when the trigger sits near the top of the viewport/panel and
// upward otherwise, clamped to the viewport horizontally.
export default function InfoDot({ term, text }: { term?: string; text?: string }) {
  const body = text ?? (term ? GLOSSARY[term] : "") ?? "";
  const ref = useRef<HTMLSpanElement | null>(null);
  const [pos, setPos] = useState<PopPos | null>(null);
  if (!body) return null;

  const open = () => {
    const el = ref.current;
    if (!el || typeof window === "undefined") return;
    const r = el.getBoundingClientRect();
    let left = r.left + r.width / 2 - POP_WIDTH / 2;
    left = Math.max(GAP, Math.min(left, window.innerWidth - POP_WIDTH - GAP));
    // Flip below when there is not comfortably enough room above (the blurbs
    // run ~70–100px tall); otherwise keep the classic above-the-dot placement.
    const below = r.top < 140;
    const top = below ? r.bottom + GAP : r.top - GAP;
    setPos({ left, top, below });
  };
  const close = () => setPos(null);

  return (
    <span
      ref={ref}
      className="infodot"
      tabIndex={0}
      role="note"
      aria-label={body}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
    >
      <span className="infodot-mark" aria-hidden="true">
        i
      </span>
      {pos && (
        <span
          className="infodot-pop"
          role="tooltip"
          style={{
            position: "fixed",
            left: pos.left,
            top: pos.top,
            bottom: "auto",
            transform: pos.below ? "none" : "translateY(-100%)",
            opacity: 1,
          }}
        >
          {body}
        </span>
      )}
    </span>
  );
}
