import { useEffect, useRef, useState } from "react";
import type { InspectorDTO } from "../types";
import { narrate } from "../lib/narrate";

// A cinematic "subtitle" for the agent's thinking, overlaid at the foot of the
// arena. Debounced: the sentence only changes when it actually changes AND has
// held for a beat, so it reads calmly instead of strobing at the frame rate.
const MIN_HOLD_MS = 260;

export default function DecisionNarrator({ inspector }: { inspector: InspectorDTO | null }) {
  const next = narrate(inspector);
  const [shown, setShown] = useState<typeof next>(null);
  const committedAt = useRef(0);

  useEffect(() => {
    if (!next) {
      if (shown) setShown(null);
      return;
    }
    if (!shown || shown.text !== next.text) {
      const now = performance.now();
      if (!shown || now - committedAt.current >= MIN_HOLD_MS) {
        committedAt.current = now;
        setShown(next);
      } else {
        const id = setTimeout(() => {
          committedAt.current = performance.now();
          setShown(next);
        }, MIN_HOLD_MS - (now - committedAt.current));
        return () => clearTimeout(id);
      }
    }
  }, [next?.text, shown]);

  if (!shown) return null;
  return (
    <div className={"narrator narrator-" + shown.tone} role="status" aria-live="polite">
      <span className="narrator-dot" />
      {shown.text}
    </div>
  );
}
