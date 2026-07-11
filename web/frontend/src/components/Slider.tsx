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

// Reusable labelled range control. The slider value round-trips through the
// server (WS), and server state re-renders every frame — so while the user is
// actively dragging we hold a LOCAL value and ignore incoming frames, otherwise
// the thumb fights the per-frame updates and jitters. We re-sync to the server
// value as soon as the drag ends.
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
  const id = "slider-" + label.toLowerCase().replace(/\s+/g, "-");

  useEffect(() => {
    if (!dragging.current) setLocal(value);
  }, [value]);

  const release = () => {
    dragging.current = false;
    setLocal(value);
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
          onChange(v);
        }}
      />
      <span className="mono slider-val">{format(local)}</span>
    </div>
  );
}
