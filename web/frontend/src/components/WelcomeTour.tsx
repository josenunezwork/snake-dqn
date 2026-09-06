import { useEffect, useLayoutEffect, useRef, useState } from "react";

// First-run onboarding: a one-sentence welcome, then an optional 3–4 step
// spotlight tour that dims the app and frames each key surface in turn. Gated by
// a localStorage flag so it shows once; replayable from Help / the palette.

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

interface Step {
  selector: string;
  title: string;
  body: string;
}

const STEPS: Step[] = [
  {
    selector: ".stage",
    title: "The arena",
    body: "A trained agent plays snake against copies of itself. Click any snake to inspect it — the one wearing the ring is selected. The caption narrates its live decision.",
  },
  {
    selector: ".telemetry",
    title: "Live telemetry",
    body: "Best length, food, kills, training loss and ε (exploration) update every frame, so you can watch the run unfold.",
  },
  {
    selector: ".tabs",
    title: "Six panels",
    body: "Inspector shows the AI's decision and what it senses, Raster shows what the new conv model literally sees, Network peeks inside the net, Dashboard ranks checkpoints — and Play lets you race the AI yourself.",
  },
  {
    selector: ".cmdk-pill",
    title: "Command palette",
    body: `Press ${IS_MAC ? "⌘K" : "Ctrl-K"} anytime for a searchable list of every control — play, speed, models, inspect a snake, and more.`,
  },
];

const TOOLTIP_W = 330;

export default function WelcomeTour({ onClose }: { onClose: () => void }) {
  // -1 = welcome card; 0..n = spotlight steps
  const [step, setStep] = useState(-1);
  const [rect, setRect] = useState<DOMRect | null>(null);
  // The dialog surface (the welcome card, then the tour tooltip): initial focus
  // target and Tab-trap boundary. aria-modal="true" asserts the rest of the page
  // is inert, so focus must live in here for the overlay's whole lifetime and go
  // back to whatever was focused before it opened.
  const dialogRef = useRef<HTMLDivElement>(null);
  const [opener] = useState<HTMLElement | null>(() =>
    typeof document === "undefined" ? null : (document.activeElement as HTMLElement | null)
  );

  const finish = () => {
    localStorage.setItem("snake_onboarded", "1");
    onClose();
  };

  // Welcome card → step 0; steps advance; the last step finishes.
  const advance = () => {
    if (step < 0) setStep(0);
    else if (step >= STEPS.length - 1) finish();
    else setStep(step + 1);
  };

  // Move focus into the dialog on open and on every step change (re-announcing
  // the step to assistive tech); hand it back to the opener on close.
  useEffect(() => {
    dialogRef.current?.focus();
  }, [step]);
  useEffect(() => {
    return () => {
      if (opener?.isConnected) opener.focus();
    };
  }, [opener]);

  // Measure the current target (re-measuring on resize + a slow poll, since the
  // arena and telemetry can resize as the first frames arrive).
  useLayoutEffect(() => {
    if (step < 0) return;
    const measure = () => {
      const el = document.querySelector(STEPS[Math.min(step, STEPS.length - 1)].selector);
      setRect(el ? el.getBoundingClientRect() : null);
    };
    measure();
    window.addEventListener("resize", measure);
    const id = window.setInterval(measure, 400);
    return () => {
      window.removeEventListener("resize", measure);
      window.clearInterval(id);
    };
  }, [step]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Tab") {
        // Trap Tab inside the dialog (mirrors HelpOverlay). If focus somehow
        // escaped (e.g. a veil click landed it on <body>), pull it back in.
        const dialog = dialogRef.current;
        const items = Array.from(dialog?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []);
        if (!dialog || items.length === 0) {
          e.preventDefault();
          return;
        }
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        if (!dialog.contains(active)) {
          e.preventDefault();
          first.focus();
        } else if (e.shiftKey && (active === first || active === dialog)) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && active === last) {
          e.preventDefault();
          first.focus();
        }
      } else if (e.key === "Escape") {
        e.preventDefault();
        finish();
      } else if (e.key === "Enter" && (e.target as HTMLElement)?.closest?.("button")) {
        // A focused button (Take a quick tour / Explore on my own / Skip / Back /
        // Next) must activate natively — don't hijack Enter to advance over it.
        return;
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault();
        advance();
      } else if (step > 0 && e.key === "ArrowLeft") {
        e.preventDefault();
        setStep((s) => Math.max(0, s - 1));
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  // ---- welcome card -----------------------------------------------------
  if (step < 0) {
    return (
      <div className="overlay-backdrop" onClick={finish}>
        <div
          className="overlay-card welcome-card"
          ref={dialogRef}
          tabIndex={-1}
          role="dialog"
          aria-modal="true"
          aria-label="Welcome to snake-dqn"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="welcome-badge">🐍 snake-dqn</div>
          <h2 style={{ margin: "6px 0 4px" }}>An AI is learning to play snake.</h2>
          <p className="muted" style={{ marginTop: 0, lineHeight: 1.6 }}>
            Watch it think in real time — see what it senses, how confident it is, and which
            move it picks. Or jump into the arena and race it yourself.
          </p>
          <div className="welcome-actions">
            <button className="btn primary" onClick={() => setStep(0)}>
              Take a quick tour
            </button>
            <button className="btn" onClick={finish}>
              Explore on my own
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---- spotlight step ---------------------------------------------------
  // Clamp defensively: batched updates or a stale index must never read past the end.
  const s = STEPS[Math.min(step, STEPS.length - 1)];
  const last = step === STEPS.length - 1;
  const pad = 8;

  // spotlight hole geometry (fallbacks to a centred card if the target is gone)
  const hole = rect
    ? {
        top: Math.max(0, rect.top - pad),
        left: Math.max(0, rect.left - pad),
        width: rect.width + pad * 2,
        height: rect.height + pad * 2,
      }
    : null;

  // tooltip placement: below the target if it fits, else above, else (a target too
  // tall for either — e.g. the arena) pinned near its top. Always fully on-screen.
  const TIP_H = 170;
  let tipStyle: React.CSSProperties;
  if (rect) {
    const left = Math.min(
      Math.max(12, rect.left + rect.width / 2 - TOOLTIP_W / 2),
      window.innerWidth - TOOLTIP_W - 12
    );
    let top: number;
    if (rect.bottom + 14 + TIP_H < window.innerHeight) top = rect.bottom + 14;
    else if (rect.top - 14 - TIP_H > 0) top = rect.top - 14 - TIP_H;
    else top = Math.min(rect.top + 16, window.innerHeight - TIP_H - 12);
    top = Math.max(12, top);
    tipStyle = { top, left };
  } else {
    tipStyle = { top: "40%", left: "50%", transform: "translate(-50%, -50%)" };
  }

  return (
    <div className="tour" role="dialog" aria-modal="true" aria-label={`Tour: ${s.title}`}>
      {hole ? (
        <div className="tour-hole" style={hole} />
      ) : (
        // No spotlight target (e.g. the stream hasn't produced the element yet):
        // clicking the veil advances like "Next" — it must NOT dismiss the tour.
        <div className="tour-veil" onClick={advance} />
      )}
      <div className="tour-tip" ref={dialogRef} tabIndex={-1} style={{ width: TOOLTIP_W, ...tipStyle }}>
        <div className="tour-tip-title">{s.title}</div>
        <div className="tour-tip-body">{s.body}</div>
        <div className="tour-tip-foot">
          <span className="tour-dots">
            {STEPS.map((_, i) => (
              <span key={i} className={"tour-dot" + (i === step ? " on" : "")} />
            ))}
          </span>
          <span className="tour-btns">
            <button className="btn tour-skip" onClick={finish}>
              Skip
            </button>
            {step > 0 && (
              <button className="btn" onClick={() => setStep((v) => Math.max(0, v - 1))}>
                Back
              </button>
            )}
            <button
              className="btn primary"
              onClick={() => (last ? finish() : setStep((v) => Math.min(v + 1, STEPS.length - 1)))}
            >
              {last ? "Done" : "Next"}
            </button>
          </span>
        </div>
      </div>
    </div>
  );
}
