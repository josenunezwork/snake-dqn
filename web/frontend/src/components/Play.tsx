import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { fetchLeaderboard, fetchPlayer, fetchRecent, submitScore } from "../api";
import {
  getClientId,
  getStoredPlayerName,
  normalizePlayerName,
  setStoredPlayerName,
} from "../identity";
import type {
  LeaderboardData,
  PlayerStats,
  PlayState,
  RecentData,
  SendControl,
  SubmitResponse,
} from "../types";
import Confetti from "./Confetti";

interface Props {
  play: PlayState | null;
  send: SendControl;
  speed: number;
  error?: string | null;
  // False while the socket is reconnecting — the panel disables engine actions.
  connected?: boolean;
  // App's authoritative pending-mode state (cleared on the first frame after a
  // (re)connect). When provided it supersedes the local "starting…" latch, so a
  // backend restart can never strand an eternal "Switching to play mode…".
  pendingMode?: string | null;
  // Guarded routes provided by App: starting a run may need the train-progress
  // confirm, ending one returns the engine to watch. Fall back to raw set_mode
  // sends so the component works standalone (tests).
  onStartRun?: () => void;
  onEndRun?: () => void;
}

const MEDALS = ["🥇", "🥈", "🥉"];
const OPPONENT_PRESETS = [3, 5, 8];
const MIN_BOOST_LEN = 5; // boost requires length >= this (see game mechanics)
// Boost's price, surfaced in the HUD so players don't discover it by shrinking
// (game_config.py: boost_length_cost_frames = 3).
const BOOST_COST = "burns 1 segment / 3 frames";

// True for touch-first devices (phones/tablets), where the keyboard steering
// keys don't exist — they get an on-screen pad instead.
function hasCoarsePointer(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}

function fmtDuration(seconds: number): string {
  if (seconds >= 60) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${seconds.toFixed(1)}s`;
}

// On-screen steering for coarse-pointer devices: a d-pad plus a hold-to-boost
// button, sending the same human_input / human_boost controls the keyboard does.
function TouchPad({ send }: { send: SendControl }) {
  const steer = (dir: string) => (e: ReactPointerEvent) => {
    e.preventDefault();
    send("human_input", dir);
  };
  const boost = (on: boolean) => (e: ReactPointerEvent) => {
    e.preventDefault();
    send("human_boost", on);
  };
  const cell = (dir: string, glyph: string, area: string) => (
    <button
      className="btn dpad-btn"
      aria-label={`Steer ${dir}`}
      style={{ gridArea: area, touchAction: "none", minWidth: 52, minHeight: 44, fontSize: 16 }}
      onPointerDown={steer(dir)}
      onContextMenu={(e) => e.preventDefault()}
    >
      {glyph}
    </button>
  );
  return (
    <div className="dpad" style={{ marginTop: 10 }}>
      <div
        style={{
          display: "grid",
          gridTemplateAreas: `". up ." "left down right"`,
          gap: 6,
          justifyContent: "center",
          touchAction: "none",
        }}
      >
        {cell("up", "▲", "up")}
        {cell("left", "◀", "left")}
        {cell("down", "▼", "down")}
        {cell("right", "▶", "right")}
      </div>
      <button
        className="btn dpad-boost"
        aria-label="Hold to boost"
        title={`Hold to boost — ${BOOST_COST}`}
        style={{ marginTop: 6, width: "100%", touchAction: "none", minHeight: 40 }}
        onPointerDown={boost(true)}
        onPointerUp={boost(false)}
        onPointerCancel={boost(false)}
        onPointerLeave={boost(false)}
        onContextMenu={(e) => e.preventDefault()}
      >
        ⚡ Hold to boost
      </button>
    </div>
  );
}

// The Play tab: an explicit start-run gate, live HUD for the human run, a
// game-over score-submit form with placement feedback, and the persisted
// leaderboard + recent runs. Movement keys are captured globally in App.tsx;
// coarse-pointer devices get the on-screen pad.
export default function Play({
  play,
  send,
  speed,
  error: sessionError,
  connected,
  pendingMode,
  onStartRun,
  onEndRun,
}: Props) {
  const [board, setBoard] = useState<LeaderboardData | null>(null);
  const [recent, setRecent] = useState<RecentData | null>(null);
  const [player, setPlayer] = useState<PlayerStats | null>(null);
  const [playerStale, setPlayerStale] = useState(false);
  const [boardError, setBoardError] = useState(false);
  const [name, setName] = useState<string>(getStoredPlayerName);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [placed, setPlaced] = useState<SubmitResponse["result"] | null>(null);
  const [placedRank, setPlacedRank] = useState<number | null>(null);
  // Latch a successful submit locally so the form is replaced by the placement
  // feedback immediately, without waiting for the server's next frame to flip
  // play.submitted. Only set AFTER the DB write succeeds — a failed submit keeps
  // the form (and its Submit button) usable for a retry.
  const [didSubmit, setDidSubmit] = useState(false);
  const [celebrate, setCelebrate] = useState(false);
  const [scoreBumped, setScoreBumped] = useState(false);
  const [starting, setStarting] = useState(false);
  const [clientId] = useState(getClientId);
  const [touch] = useState(hasCoarsePointer);
  const bestBeforeRun = useRef(0); // known personal best snapshotted at run start
  const prevScoreRef = useRef(0);

  const myName = normalizePlayerName(name);

  // Prefer the opaque per-browser id for "is this row mine?" when the server
  // ships one; fall back to display-name match for older rows.
  const isMe = useCallback(
    (e: { player_name: string; client_id?: string | null }) => {
      return e.client_id ? e.client_id === clientId : e.player_name === myName;
    },
    [clientId, myName]
  );

  const refresh = useCallback(async () => {
    try {
      const [b, r] = await Promise.all([fetchLeaderboard(10), fetchRecent(8)]);
      setBoard(b);
      setRecent(r);
      setBoardError(false);
    } catch {
      setBoardError(true);
    }
    const nm = normalizePlayerName(getStoredPlayerName());
    if (nm) {
      try {
        // api.ts appends the persisted client_id itself (withClientId).
        const p = await fetchPlayer(nm);
        setPlayer(p.found ? p.player : null);
        setPlayerStale(false);
      } catch {
        setPlayerStale(true); // keep prior stats but mark them as possibly stale
      }
    }
  }, [clientId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Clear last-run feedback when a fresh run is underway, and snapshot the known
  // personal best so we can detect a new PB when this run ends.
  useEffect(() => {
    if (play?.human_alive && !play?.run_over) {
      setPlaced(null);
      setPlacedRank(null);
      setError(null);
      setDidSubmit(false);
      bestBeforeRun.current =
        player?.best_score ?? board?.leaderboard.find(isMe)?.score ?? 0;
    }
  }, [play?.human_alive, play?.run_over, player, board, isMe]);

  // Reset the "starting…" latch once the mode switch lands (or fails).
  useEffect(() => {
    if (play?.active || sessionError) setStarting(false);
  }, [play?.active, sessionError]);

  // Pulse the score readout each time it ticks up.
  useEffect(() => {
    const s = play?.score ?? 0;
    if (s > prevScoreRef.current) {
      setScoreBumped(true);
      const id = setTimeout(() => setScoreBumped(false), 350);
      prevScoreRef.current = s;
      return () => clearTimeout(id);
    }
    prevScoreRef.current = s;
  }, [play?.score]);

  const runOver = !!play?.run_over;
  const submitted = !!play?.submitted || didSubmit;
  const canSubmit = runOver && !submitted && !!play?.pending;

  const onSubmit = useCallback(async () => {
    const trimmed = normalizePlayerName(name) || "anonymous";
    setStoredPlayerName(trimmed);
    setName(trimmed);
    setSubmitting(true);
    setError(null);
    try {
      // api.ts attaches the persisted client_id to the submission itself.
      const resp = await submitScore(trimmed);
      if (resp.ok) {
        const result = resp.result ?? null;
        // The server echoes the canonical stored name (its sanitizer may differ);
        // use it for the rank lookup and future "me" matching.
        const canonical = result?.player_name ?? trimmed;
        if (canonical !== trimmed) {
          setStoredPlayerName(canonical);
          setName(canonical);
        }
        const entry = resp.leaderboard?.find(
          (e) => isMe(e) || e.player_name === canonical
        );
        // Only claim a placement if THIS run is the player's best: the leaderboard
        // has one row per player (their best game), so a weaker re-submit under the
        // same name would otherwise show the OLD best's rank/medal. Require the
        // matched row's score to equal this run's score.
        const rank = entry && result && entry.score === result.score ? entry.rank : null;
        setPlaced(result);
        setPlacedRank(rank);
        setDidSubmit(true);
        // Celebrate a leaderboard placement or a new personal best.
        if (rank != null || (result != null && result.score > bestBeforeRun.current)) {
          setCelebrate(true);
        }
        await refresh();
      } else {
        setError(resp.error ?? "Submit failed");
      }
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : "Could not reach the server");
    } finally {
      setSubmitting(false);
    }
  }, [name, refresh, clientId, isMe]);

  const newGame = useCallback(() => {
    setPlaced(null);
    setPlacedRank(null);
    setError(null);
    send("new_game");
    refresh();
  }, [send, refresh]);

  const startRun = useCallback(() => {
    setStarting(true);
    if (onStartRun) onStartRun();
    else send("set_mode", "play");
  }, [onStartRun, send]);

  const endRun = useCallback(() => {
    if (onEndRun) onEndRun();
    else send("set_mode", "watch");
  }, [onEndRun, send]);

  // App's pending-mode state (cleared on the first post-connect frame) is
  // authoritative when provided; the local latch covers standalone mounting.
  const switching = pendingMode !== undefined ? pendingMode === "play" : starting;

  if (!play?.active) {
    // Pure navigation brought us here — the engine is still in watch/train mode.
    // Nothing rebuilds until the player explicitly starts a run.
    return (
      <div className="panel">
        <div className="card">
          <div className="section-title">🎮 Race the AI</div>
          <div className="muted" style={{ fontSize: 12, lineHeight: 1.7 }}>
            Steer your own snake in the live arena — eat food, dodge the AI, climb the
            leaderboard. Starting a run rebuilds the game around your snake.
          </div>
          {sessionError ? (
            <>
              <div className="error" style={{ marginTop: 8 }}>
                {sessionError}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                Play mode couldn't start. Try another model in Controls, then try again.
              </div>
            </>
          ) : connected === false ? (
            <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
              Reconnecting to the server…
            </div>
          ) : (
            switching && (
              <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
                Switching to play mode…
              </div>
            )
          )}
          <button
            className="btn primary"
            style={{ marginTop: 10 }}
            disabled={connected === false || (switching && !sessionError)}
            onClick={startRun}
          >
            ▶ Start run
          </button>
        </div>
      </div>
    );
  }

  const myBest = board?.leaderboard.find(isMe)?.score ?? null;
  const preRun = !play.run_started && !play.run_over;
  const timeAlive = fmtDuration(play.frames / Math.max(1, speed));
  const finalScore = play.pending?.score ?? play.score;
  const isNewPB = !play.human_alive && finalScore > bestBeforeRun.current;
  const boostReady = play.length >= MIN_BOOST_LEN;
  const boostPct = Math.min(1, play.length / MIN_BOOST_LEN) * 100;

  return (
    <div className="panel">
      {celebrate && <Confetti onDone={() => setCelebrate(false)} />}
      <div className="statgrid" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
        <div className="stat">
          <div className="k">Score</div>
          <div className={"v" + (scoreBumped ? " bump" : "")} style={{ color: "var(--green)" }}>
            {preRun ? "—" : play.score}
          </div>
        </div>
        <div className="stat">
          <div className="k">Length</div>
          <div className="v">{play.length}</div>
        </div>
        <div className="stat">
          <div className="k">Food</div>
          <div className="v">{play.food_eaten}</div>
        </div>
        <div className="stat">
          <div className="k">Kills</div>
          <div className="v">{play.kills}</div>
        </div>
        <div className="stat">
          <div className="k">Time</div>
          <div className="v">{preRun ? "—" : timeAlive}</div>
        </div>
      </div>

      {play.human_alive ? (
        <div className="card" style={{ marginTop: 12 }}>
          {play.run_started ? (
            <>
              <div className="section-title">You're alive — good luck</div>
              <div className="muted" style={{ fontSize: 12, lineHeight: 1.7 }}>
                <span className="kbd">← ↑ ↓ →</span> or <span className="kbd">W A S D</span> to
                steer
                <br />
                <span className="kbd">Space</span> to boost ({BOOST_COST}) · eat food to grow ·
                outlast the AI
              </div>
            </>
          ) : (
            <>
              <div className="section-title" style={{ color: "var(--accent)" }}>
                {touch ? "Tap the pad (or press an arrow key) to start" : "Press an arrow key to start"}
              </div>
              <div className="muted" style={{ fontSize: 12, lineHeight: 1.7 }}>
                <span className="kbd">← ↑ ↓ →</span> or <span className="kbd">W A S D</span> to
                steer · <span className="kbd">Space</span> to boost ({BOOST_COST})
              </div>
              <div className="row" style={{ marginTop: 10, marginBottom: 0 }}>
                <label style={{ width: "auto" }}>Opponents</label>
                <div className="seg">
                  {OPPONENT_PRESETS.map((n) => (
                    <button
                      key={n}
                      className={"btn" + (play.opponents === n ? " active" : "")}
                      aria-label={`${n} AI opponents`}
                      onClick={() => send("set_play_opponents", n)}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
          <div className="row" style={{ marginTop: 10, marginBottom: 0, alignItems: "center" }}>
            <label style={{ width: 44 }}>Boost</label>
            <div
              className="meter"
              title={
                boostReady
                  ? `Boost ready — holding it ${BOOST_COST}`
                  : `Grow to unlock boost (needs length ≥ ${MIN_BOOST_LEN})`
              }
            >
              <div
                className="meter-fill"
                style={{ width: `${boostPct}%`, background: boostReady ? "var(--green)" : "var(--amber)" }}
              />
            </div>
            <span
              className="mono"
              style={{ fontSize: 11, width: 64, textAlign: "right", color: boostReady ? "var(--green)" : "var(--muted)" }}
            >
              {boostReady ? "ready" : `+${MIN_BOOST_LEN - play.length}`}
            </span>
          </div>
          {touch && <TouchPad send={send} />}
          {player ? (
            <div className="muted mono" style={{ fontSize: 11, marginTop: 8 }}>
              your best: {player.best_score} · {player.games_played} games · avg{" "}
              {Math.round(player.average_score)}
              {playerStale && <span title="Couldn't refresh your stats — showing the last known values."> (may be stale)</span>}
            </div>
          ) : (
            myBest != null && (
              <div className="muted mono" style={{ fontSize: 11, marginTop: 8 }}>
                your best: {myBest}
              </div>
            )
          )}
        </div>
      ) : (
        <div className="card" style={{ marginTop: 12, borderColor: "#6f2929" }}>
          <div className="section-title" style={{ color: "var(--red)" }}>
            Game over — final score {finalScore}
            {isNewPB && <span className="pb-badge">★ personal best</span>}
          </div>
          <div className="muted" style={{ fontSize: 12 }}>
            survived {fmtDuration(play.pending?.duration_seconds ?? play.frames / Math.max(1, speed))}{" "}
            · {play.pending?.food_eaten ?? play.food_eaten} food ·{" "}
            {play.pending?.kills ?? play.kills} kills
          </div>

          {canSubmit ? (
            <div style={{ marginTop: 8 }}>
              <div className="row" style={{ marginBottom: 6 }}>
                <input
                  type="text"
                  aria-label="Your name"
                  value={name}
                  maxLength={32}
                  placeholder="your name"
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onSubmit();
                  }}
                  style={{
                    flex: 1,
                    background: "var(--panel-2)",
                    color: "var(--text)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    padding: "7px 10px",
                  }}
                />
                <button className="btn primary" disabled={submitting} onClick={onSubmit}>
                  {submitting ? "…" : "Submit"}
                </button>
              </div>
              {error && (
                <div className="error">
                  {error} — your run is safe, submit again to retry.
                </div>
              )}
            </div>
          ) : (
            <div style={{ marginTop: 6 }}>
              {placedRank != null ? (
                <div className={"placement-banner tier-" + Math.min(placedRank, 4)}>
                  {placedRank === 1 ? "🏆 New #1!" : `You placed #${placedRank}!`}{" "}
                  {MEDALS[placedRank - 1] ?? ""}
                </div>
              ) : placed ? (
                <div className="muted" style={{ fontSize: 12 }}>
                  Score {placed.score} recorded — crack the top 10 to rank.
                </div>
              ) : (
                <div className="muted" style={{ fontSize: 12 }}>
                  {submitted ? "Score submitted! 🏆" : "No score to submit."}
                </div>
              )}
            </div>
          )}
          <button className="btn" style={{ marginTop: 10 }} onClick={newGame}>
            ▶ New game <span className="kbd" style={{ marginLeft: 6 }}>R</span>
          </button>
        </div>
      )}

      <div className="row" style={{ marginTop: 10, marginBottom: 0 }}>
        <button className="btn" title="Leave play mode and return the arena to the AI" onClick={endRun}>
          ← End run · back to watch
        </button>
      </div>

      <div className="card">
        <div className="section-title">🏆 Leaderboard</div>
        {boardError ? (
          <div>
            <div className="error">Leaderboard unavailable.</div>
            <button className="btn" style={{ marginTop: 6 }} onClick={refresh}>
              Retry
            </button>
          </div>
        ) : board && board.leaderboard.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Player</th>
                <th className="num">Score</th>
                <th className="num">Length</th>
              </tr>
            </thead>
            <tbody>
              {board.leaderboard.map((e) => (
                <tr key={e.rank} className={isMe(e) ? "me" : undefined}>
                  <td className="num">{MEDALS[e.rank - 1] ?? e.rank}</td>
                  <td>{e.player_name}</td>
                  <td className="num">{e.score}</td>
                  <td className="num">{e.length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="muted" style={{ fontSize: 12 }}>
            No scores yet — be the first to make the board.
          </div>
        )}
        {board?.stats && board.stats.total_games > 0 && (
          <div className="muted mono" style={{ fontSize: 11, marginTop: 8 }}>
            {board.stats.total_players} players · {board.stats.total_games} games · best{" "}
            {board.stats.best_score} ({board.stats.best_player})
          </div>
        )}
      </div>

      {recent && recent.recent.length > 0 && (
        <div className="card">
          <div className="section-title">Recent runs</div>
          <div className="mono" style={{ fontSize: 11, lineHeight: 1.9 }}>
            {recent.recent.map((g) => (
              <div
                key={g.id}
                className={isMe(g) ? "me" : undefined}
                style={{ display: "flex", justifyContent: "space-between" }}
              >
                <span className="muted">{g.player_name}</span>
                <span>
                  {g.score} · len {g.length}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
