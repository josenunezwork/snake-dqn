import { useEffect, useState } from "react";
import { fetchCheckpoints } from "../api";
import type { CheckpointInfo, SendControl, SessionState, SnakeDTO } from "../types";
import { SPEED_PRESETS, activePresetIndex } from "../lib/presets";
import Slider from "./Slider";

const MODE_BLURB: Record<string, string> = {
  watch: "Frozen weights — a pure demo of the trained agent.",
  train: "Learning live — watch loss and ε move as it improves.",
  play: "You steer a snake against the AI — scored runs land on the leaderboard.",
};

// Human-readable description of what the served network actually consumes,
// keyed by the session's observation contract.
function inputLabel(obsSpec: string | undefined, inputSize: number): string {
  if (obsSpec === "raster31v2") return "raster 31×31 + 25×25 + 26 scalars";
  return `${inputSize}-D vector`;
}

interface Props {
  session: SessionState;
  snakes: SnakeDTO[];
  send: SendControl;
  ghosts: boolean;
  onToggleGhosts: () => void;
  // False while the socket is reconnecting — engine actions are disabled so a
  // click can't silently no-op into the void.
  connected?: boolean;
  // Guarded routes provided by App (mode changes and checkpoint loads may need a
  // "training progress is in memory" confirm). Fall back to a raw send so the
  // component still works standalone (tests, storybook-style mounting).
  onModeChange?: (mode: "watch" | "train" | "play") => void;
  onLoadCheckpoint?: (path: string) => void;
  // Visible "Copy run summary" affordance; App passes its clipboard helper.
  onCopySummary?: () => void;
}

export default function Controls({
  session,
  snakes,
  send,
  ghosts,
  onToggleGhosts,
  connected,
  onModeChange,
  onLoadCheckpoint,
  onCopySummary,
}: Props) {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [ckptError, setCkptError] = useState(false);
  // The select is for BROWSING; nothing loads until the explicit Load action.
  const [selected, setSelected] = useState<string>(session.checkpoint ?? "");

  const loadCheckpoints = () => {
    setCkptError(false);
    fetchCheckpoints()
      .then((cs) => {
        setCheckpoints(cs);
        setCkptError(false);
      })
      .catch(() => setCkptError(true));
  };

  useEffect(loadCheckpoints, []);

  // Adopt the server's loaded checkpoint as the browsing baseline whenever it
  // changes (initial frame, load elsewhere, palette load).
  useEffect(() => {
    setSelected(session.checkpoint ?? "");
  }, [session.checkpoint]);

  const changeMode = (mode: "watch" | "train" | "play") => {
    if (onModeChange) onModeChange(mode);
    else send("set_mode", mode);
  };

  const commitLoad = (path: string) => {
    if (!path) return;
    if (onLoadCheckpoint) onLoadCheckpoint(path);
    else send("load_checkpoint", path);
  };

  const obsSpec = session.obs_spec;
  const checkpointSpec = (c: CheckpointInfo) => c.obs_spec;

  const loadedName = session.checkpoint ?? "";
  const canLoad = !!selected && selected !== loadedName;
  const offline = connected === false;

  return (
    <div className="panel">
      {offline && (
        <div className="muted" role="status" style={{ fontSize: 12, marginBottom: 6 }}>
          Reconnecting — controls are paused until the server is back.
        </div>
      )}
      <div className="row">
        <button
          className="btn primary"
          aria-label={session.playing ? "Pause" : "Play"}
          disabled={offline}
          onClick={() => send(session.playing ? "pause" : "play")}
        >
          {session.playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button className="btn" aria-label="Reset game" disabled={offline} onClick={() => send("reset")}>
          ⟲ Reset
        </button>
        <button
          className="btn"
          aria-label="Save weights"
          title="Save the current policy weights to saved_snakes/ (train-mode progress lives only in memory until saved)"
          disabled={offline}
          onClick={() => send("save_weights")}
        >
          💾 Save weights
        </button>
        <span className="kbd-hint">space · R</span>
      </div>

      <div className="row" style={{ marginBottom: 4 }}>
        <label>Mode</label>
        <div className="seg">
          <button
            className={"btn" + (session.mode === "watch" ? " active" : "")}
            disabled={offline}
            onClick={() => changeMode("watch")}
          >
            Watch
          </button>
          <button
            className={"btn" + (session.mode === "train" ? " active" : "")}
            disabled={offline}
            onClick={() => changeMode("train")}
          >
            Train
          </button>
          <button
            className={"btn" + (session.mode === "play" ? " active" : "")}
            disabled={offline}
            onClick={() => changeMode("play")}
          >
            Play
          </button>
        </div>
      </div>
      {MODE_BLURB[session.mode] && (
        <div className="mode-blurb muted">
          {MODE_BLURB[session.mode]}
          {session.mode === "train" && session.reward_override_active && (
            <>
              {" "}
              <span title="This checkpoint recorded different reward economics than the arena runs; its weights were warm-started and are training under the current rewards.">
                Fine-tuning across reward versions — warm start, not a resume.
              </span>
            </>
          )}
        </div>
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
      <div title="Live food count · ambient target. Corpse drops are exempt from the target, so the count can legitimately exceed it.">
        <Slider
          label="Food"
          value={session.food_target}
          min={0}
          max={600}
          step={10}
          accent="#f59e0b"
          format={(v) => `${session.food_count} · target ${v}`}
          onChange={(v) => send("set_food", v)}
        />
        <div className="muted" style={{ fontSize: 11, marginTop: -2, marginBottom: 6 }}>
          Corpse drops are cap-exempt — the live count can exceed the target.
        </div>
      </div>

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
      <div className="roster" role="group" aria-label="Snakes — select the hero to inspect">
        {[...snakes]
          .sort((a, b) => Number(b.alive) - Number(a.alive) || b.length - a.length)
          .map((s) => {
            const hero = s.id === session.hero_id;
            const rgb = `rgb(${s.color[0]},${s.color[1]},${s.color[2]})`;
            return (
              <button
                key={s.id}
                aria-pressed={hero}
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
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {!loadedName && !selected && <option value="">— none —</option>}
          {/* Keep the loaded checkpoint selectable even if the listing failed or
              rotated it out, so the select never renders blank. */}
          {loadedName && !checkpoints.some((c) => c.name === loadedName) && (
            <option value={loadedName}>{loadedName} (loaded)</option>
          )}
          {checkpoints.map((c) => {
            const spec = checkpointSpec(c);
            return (
              <option key={c.name} value={c.name}>
                {c.name} ({c.size_mb} MB{spec && spec !== "unknown" ? ` · ${spec}` : ""})
              </option>
            );
          })}
        </select>
        <button
          className="btn"
          aria-label="Load selected checkpoint"
          disabled={!canLoad}
          title={
            canLoad
              ? "Load the selected checkpoint (rebuilds the game around it)"
              : "Browse the list, then load — the selected model is already active"
          }
          onClick={() => commitLoad(selected)}
        >
          Load
        </button>
      </div>
      {ckptError && (
        <div className="row" style={{ marginTop: -4 }}>
          <span className="error" style={{ flex: 1 }}>
            Couldn't fetch the checkpoint list.
          </span>
          <button className="btn" onClick={loadCheckpoints}>
            Retry
          </button>
        </div>
      )}

      <div className="card" style={{ marginTop: 4 }}>
        <div className="section-title">
          Session
          {obsSpec && (
            <span className="rbadge" style={{ marginLeft: 6 }} title="Observation contract of the served policy">
              {obsSpec}
            </span>
          )}
        </div>
        <div className="muted mono" style={{ fontSize: 12, lineHeight: 1.7 }}>
          config: {session.config}
          <br />
          input: {inputLabel(obsSpec, session.input_size)} · snakes: {session.num_snakes}
          <br />
          training: {session.training ? "on" : "off"}
        </div>
        {onCopySummary && (
          <button
            className="btn"
            style={{ marginTop: 8 }}
            title="Copy a text summary of the current run to the clipboard"
            onClick={onCopySummary}
          >
            ⧉ Copy run summary
          </button>
        )}
        {session.error && <div className="error">{session.error}</div>}
      </div>
    </div>
  );
}
