import { useEffect, useRef, useState } from "react";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  accent?: string;
  format: (v: number) => string;
  onChange: (v: number) => void;
}

// How long we coalesce live drag sends: trailing-edge, so a fast sweep sends the
// latest value at most every ~30ms instead of one control message per step.
const SEND_THROTTLE_MS = 30;

// Reusable labelled range control. The slider value round-trips through the
// server (WS), and server state re-renders every frame — so while the user is
// interacting we hold a LOCAL value and ignore incoming frames, otherwise the
// thumb fights the per-frame updates and jitters. After the user changes the
// value we KEEP the local value until the server echoes something new: the prop
// still holds the pre-change (stale) value until the next broadcast frame, and
// snapping back to it on release makes the control feel ignored. Any prop value
// different from the stale baseline (the echo of what we sent, a server-side
// clamp, or another client's change) is adopted immediately.
export default function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  accent,
  format,
  onChange,
}: Props) {
  const [local, setLocal] = useState(value);
  const dragging = useRef(false);
  // Trailing-edge send throttle state.
  const sendTimer = useRef<number | null>(null);
  const latest = useRef(value);
  const unsent = useRef(false);
  // While awaiting the server echo of a value we sent, the prop is stale; adopt
  // prop updates only once they move off the baseline captured at edit time.
  const awaitingEcho = useRef(false);
  const staleBaseline = useRef(value);
  const propRef = useRef(value);
  propRef.current = value;
  const id = "slider-" + label.toLowerCase().replace(/\s+/g, "-");

  useEffect(() => {
    if (dragging.current) return;
    if (awaitingEcho.current) {
      if (value === staleBaseline.current) return; // still the pre-edit value
      awaitingEcho.current = false;
    }
    setLocal(value);
  }, [value]);

  const doSend = () => {
    if (!unsent.current) return;
    unsent.current = false;
    onChange(latest.current);
  };

  const flushSend = () => {
    if (sendTimer.current != null) {
      window.clearTimeout(sendTimer.current);
      sendTimer.current = null;
    }
    doSend();
  };

  // Never leave a queued send dangling past unmount.
  useEffect(
    () => () => {
      if (sendTimer.current != null) window.clearTimeout(sendTimer.current);
    },
    []
  );

  const release = () => {
    dragging.current = false;
    flushSend(); // the final drag position goes out immediately
    // Deliberately no setLocal(value) here: `value` is the stale pre-drag prop
    // until the server echoes; the sync effect converges once it does.
  };

  return (
    <div className="row">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={local}
        aria-valuetext={format(local)}
        style={accent ? { accentColor: accent } : undefined}
        onPointerDown={() => (dragging.current = true)}
        onPointerUp={release}
        onPointerCancel={release}
        onBlur={release}
        onChange={(e) => {
          const v = Number(e.target.value);
          setLocal(v);
          latest.current = v;
          unsent.current = true;
          if (!awaitingEcho.current) {
            awaitingEcho.current = true;
            staleBaseline.current = propRef.current;
          }
          if (sendTimer.current == null) {
            sendTimer.current = window.setTimeout(() => {
              sendTimer.current = null;
              doSend();
            }, SEND_THROTTLE_MS);
          }
        }}
      />
      <span className="mono slider-val">{format(local)}</span>
    </div>
  );
}
