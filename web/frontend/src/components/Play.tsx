import { useCallback, useEffect, useRef, useState } from "react";
import { fetchLeaderboard, fetchPlayer, fetchRecent, submitScore } from "../api";
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
}

const MEDALS = ["🥇", "🥈", "🥉"];
const OPPONENT_PRESETS = [3, 5, 8];
const MIN_BOOST_LEN = 5; // boost requires length >= this (see game mechanics)

function fmtDuration(seconds: number): string {
  if (seconds >= 60) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${seconds.toFixed(1)}s`;
}

// The Play tab: live HUD for the human run, a game-over score-submit form with
// placement feedback, and the persisted leaderboard + recent runs. Movement keys
// are captured globally in App.tsx.
export default function Play({ play, send, speed, error: sessionError }: Props) {
  const [board, setBoard] = useState<LeaderboardData | null>(null);
  const [recent, setRecent] = useState<RecentData | null>(null);
  const [player, setPlayer] = useState<PlayerStats | null>(null);
  const [boardError, setBoardError] = useState(false);
  const [name, setName] = useState<string>(() => localStorage.getItem("snake_player") || "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [placed, setPlaced] = useState<SubmitResponse["result"] | null>(null);
  const [placedRank, setPlacedRank] = useState<number | null>(null);
  // Latch a successful submit locally so the form is replaced by the placement
  // feedback immediately, without waiting for the server's next frame to flip
  // play.submitted.
  const [didSubmit, setDidSubmit] = useState(false);
  const [celebrate, setCelebrate] = useState(false);
  const [scoreBumped, setScoreBumped] = useState(false);
  const bestBeforeRun = useRef(0); // known personal best snapshotted at run start
  const prevScoreRef = useRef(0);

  const myName = name.trim();

  const refresh = useCallback(async () => {
    try {
      const [b, r] = await Promise.all([fetchLeaderboard(10), fetchRecent(8)]);
      setBoard(b);
      setRecent(r);
      setBoardError(false);
    } catch {
      setBoardError(true);
    }
    const nm = (localStorage.getItem("snake_player") || "").trim();
    if (nm) {
      try {
        const p = await fetchPlayer(nm);
        setPlayer(p.found ? p.player : null);
      } catch {
        /* leave prior stats */
      }
    }
  }, []);

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
        player?.best_score ??
        board?.leaderboard.find((e) => e.player_name === myName)?.score ??
        0;
    }
  }, [play?.human_alive, play?.run_over, player, board, myName]);

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
    const trimmed = name.trim() || "anonymous";
    localStorage.setItem("snake_player", trimmed);
    setName(trimmed);
    setSubmitting(true);
    setError(null);
    try {
      const resp = await submitScore(trimmed);
      if (resp.ok) {
        const result = resp.result ?? null;
        const entry = resp.leaderboard?.find((e) => e.player_name === trimmed);
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
    } catch {
      setError("Could not reach the server");
    } finally {
      setSubmitting(false);
    }
  }, [name, refresh]);

  const newGame = useCallback(() => {
    setPlaced(null);
    setPlacedRank(null);
    setError(null);
    send("new_game");
    refresh();
  }, [send, refresh]);

  if (!play?.active) {
    return (
      <div className="panel">
        {sessionError ? (
          <div className="card">
            <div className="error">{sessionError}</div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              Play mode couldn't start. Try another model in Controls, then reopen Play.
            </div>
          </div>
        ) : (
          <div className="card muted">Switching to play mode…</div>
        )}
      </div>
    );
  }

  const myBest = board?.leaderboard.find((e) => e.player_name === myName)?.score ?? null;
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
            {play.score}
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
          <div className="v">{timeAlive}</div>
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
                <span className="kbd">Space</span> to boost · eat food to grow · outlast the AI
              </div>
            </>
          ) : (
            <>
              <div className="section-title" style={{ color: "var(--accent)" }}>
                Press an arrow key to start
              </div>
              <div className="muted" style={{ fontSize: 12, lineHeight: 1.7 }}>
                <span className="kbd">← ↑ ↓ →</span> or <span className="kbd">W A S D</span> to
                steer · <span className="kbd">Space</span> to boost
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
            <div className="meter" title={boostReady ? "Boost ready" : "Grow to unlock boost"}>
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
          {player ? (
            <div className="muted mono" style={{ fontSize: 11, marginTop: 8 }}>
              your best: {player.best_score} · {player.games_played} games · avg{" "}
              {Math.round(player.average_score)}
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
              {error && <div className="error">{error}</div>}
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

      <div className="card">
        <div className="section-title">🏆 Leaderboard</div>
        {boardError ? (
          <div className="error">Leaderboard unavailable.</div>
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
                <tr key={e.rank} className={e.player_name === myName ? "me" : undefined}>
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
                className={g.player_name === myName ? "me" : undefined}
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
