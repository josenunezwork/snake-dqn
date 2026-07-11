import { useEffect, useState } from "react";
import { fetchCheckpoints } from "../api";
import type { CheckpointInfo, SendControl, SessionState, SnakeDTO } from "../types";
import { SPEED_PRESETS, activePresetIndex } from "../lib/presets";
import Slider from "./Slider";

const MODE_BLURB: Record<string, string> = {
  watch: "Frozen weights — a pure demo of the trained agent.",
  train: "Learning live — watch loss and ε move as it improves.",
};

interface Props {
  session: SessionState;
  snakes: SnakeDTO[];
  send: SendControl;
  ghosts: boolean;
  onToggleGhosts: () => void;
}

export default function Controls({ session, snakes, send, ghosts, onToggleGhosts }: Props) {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);

  useEffect(() => {
    fetchCheckpoints()
      .then(setCheckpoints)
      .catch(() => setCheckpoints([]));
  }, []);

  return (
    <div className="panel">
      <div className="row">
        <button
          className="btn primary"
          aria-label={session.playing ? "Pause" : "Play"}
          onClick={() => send(session.playing ? "pause" : "play")}
        >
          {session.playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button className="btn" aria-label="Reset game" onClick={() => send("reset")}>
          ⟲ Reset
        </button>
        <span className="kbd-hint">space · R</span>
      </div>

      <div className="row" style={{ marginBottom: 4 }}>
        <label>Mode</label>
        <div className="seg">
          <button
            className={"btn" + (session.mode === "watch" ? " active" : "")}
            onClick={() => send("set_mode", "watch")}
          >
            Watch
          </button>
          <button
            className={"btn" + (session.mode === "train" ? " active" : "")}
            onClick={() => send("set_mode", "train")}
          >
            Train
          </button>
        </div>
      </div>
      {MODE_BLURB[session.mode] && (
        <div className="mode-blurb muted">{MODE_BLURB[session.mode]}</div>
      )}

      <div className="row" style={{ marginBottom: 4 }}>
        <label>Speed</label>
        <div className="seg">
          {SPEED_PRESETS.map((p, i) => (
            <button
              key={p.fps}
              className={"btn" + (activePresetIndex(session.speed) === i ? " active" : "")}
              title={`${p.fps} fps · key ${i + 1}`}
              onClick={() => send("set_speed", p.fps)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
      <Slider
        label="Fine"
        value={session.speed}
        min={1}
        max={60}
        format={(v) => `${Math.round(v)} fps`}
        onChange={(v) => send("set_speed", v)}
      />
      <Slider
        label="Epsilon"
        value={session.epsilon}
        min={0}
        max={1}
        step={0.01}
        format={(v) => v.toFixed(2)}
        onChange={(v) => send("set_epsilon", v)}
      />
      <Slider
        label="Food"
        value={session.food_target}
        min={0}
        max={600}
        step={10}
        accent="#f59e0b"
        format={(v) => `${session.food_count} / ${v}`}
        onChange={(v) => send("set_food", v)}
      />

      <div className="row">
        <label>Overlay</label>
        <div className="seg">
          <button
            className={"btn" + (ghosts ? " active" : "")}
            aria-pressed={ghosts}
            title="Project the hero's candidate moves onto the arena"
            onClick={onToggleGhosts}
          >
            {ghosts ? "◈ Intent on" : "◇ Intent off"}
          </button>
        </div>
        <span className="kbd-hint">O</span>
      </div>

      <div className="section-title" style={{ marginTop: 8 }}>
        Snakes · tap to inspect
      </div>
      <div className="roster" role="listbox" aria-label="Snakes — select the hero to inspect">
        {[...snakes]
          .sort((a, b) => Number(b.alive) - Number(a.alive) || b.length - a.length)
          .map((s) => {
            const hero = s.id === session.hero_id;
            const rgb = `rgb(${s.color[0]},${s.color[1]},${s.color[2]})`;
            return (
              <button
                key={s.id}
                role="option"
                aria-selected={hero}
                className={"roster-row" + (hero ? " hero" : "") + (s.alive ? "" : " dead")}
                onClick={() => send("set_hero", s.id)}
              >
                <span className="swatch" style={{ background: rgb, color: rgb }} />
                <span className="rname">{s.name}</span>
                {s.boosting && <span className="rbadge">boost</span>}
                {!s.alive && <span className="rbadge dead">dead</span>}
                <span className="rlen mono">{s.length}</span>
              </button>
            );
          })}
      </div>

      <div className="row" style={{ alignItems: "flex-start" }}>
        <label>Model</label>
        <select
          aria-label="Model checkpoint"
          value={session.checkpoint ?? ""}
          onChange={(e) => send("load_checkpoint", e.target.value)}
        >
          {!session.checkpoint && <option value="">— none —</option>}
          {checkpoints.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} ({c.size_mb} MB)
            </option>
          ))}
        </select>
      </div>

      <div className="card" style={{ marginTop: 4 }}>
        <div className="section-title">Session</div>
        <div className="muted mono" style={{ fontSize: 12, lineHeight: 1.7 }}>
          config: {session.config}
          <br />
          input: {session.input_size}-D · snakes: {session.num_snakes}
          <br />
          training: {session.training ? "on" : "off"}
        </div>
        {session.error && <div className="error">{session.error}</div>}
      </div>
    </div>
  );
}
