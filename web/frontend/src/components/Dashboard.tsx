import { useEffect, useState } from "react";
import { fetchMetrics } from "../api";
import type { DashboardData } from "../types";

export default function Dashboard({ checkpoint }: { checkpoint?: string | null }) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  const load = () => {
    setLoading(true);
    setFailed(false);
    fetchMetrics()
      .then(setData)
      .catch(() => setFailed(true))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  if (!data) {
    if (loading) {
      return (
        <div className="panel">
          <div className="skeleton skeleton-row" />
          <div className="skeleton skeleton-row" />
          <div className="skeleton skeleton-row" style={{ width: "70%" }} />
        </div>
      );
    }
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-glyph">{failed ? "⚠" : "▦"}</div>
          <div className="empty-title">{failed ? "Metrics unavailable" : "No metrics yet"}</div>
          <div className="muted" style={{ fontSize: 12 }}>
            {failed
              ? "Couldn't reach the metrics endpoint."
              : "Run a tournament eval to populate the leaderboard."}
          </div>
          <button className="btn" style={{ marginTop: 8 }} onClick={load}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  const maxMass = Math.max(...data.leaderboard.map((r) => r.mean_mass), 1);
  // Match the currently-loaded checkpoint so the promotion surfaces point at "you".
  const isLoaded = (name?: string | null) =>
    !!checkpoint && !!name && (name === checkpoint || name.endsWith("/" + checkpoint));

  return (
    <div className="panel">
      <div className="panel-intro">How trained checkpoints score head-to-head in frozen-opponent evals.</div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="section-title" style={{ margin: 0 }}>
          Eval leaderboard ({data.leaderboard.length})
        </div>
        <button className="btn" onClick={load}>
          {loading ? "…" : "Refresh"}
        </button>
      </div>
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <table>
          <thead>
            <tr>
              <th>Candidate</th>
              <th className="num">mass</th>
              <th className="num">max</th>
              <th className="num">surv</th>
              <th className="num">n</th>
            </tr>
          </thead>
          <tbody>
            {data.leaderboard.map((r, i) => (
              <tr key={i} className={isLoaded(r.file) || isLoaded(r.candidate) ? "me" : undefined}>
                <td>
                  <div title={`${r.candidate} vs ${r.opponent} (${r.file})`}>
                    {r.candidate}
                    {(isLoaded(r.file) || isLoaded(r.candidate)) && (
                      <span className="rbadge" style={{ marginLeft: 6 }}>
                        loaded
                      </span>
                    )}
                  </div>
                  <div className="massbar" style={{ width: `${(r.mean_mass / maxMass) * 100}%`, marginTop: 3 }} />
                </td>
                <td className="num">{r.mean_mass}</td>
                <td className="num">{r.max_mass}</td>
                <td className="num">{r.survival}</td>
                <td className="num">{r.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="section-title">Checkpoints ({data.checkpoints.length})</div>
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th className="num">MB</th>
            </tr>
          </thead>
          <tbody>
            {data.checkpoints.map((c) => (
              <tr key={c.name} className={isLoaded(c.name) ? "me" : undefined}>
                <td className="mono">
                  {c.name}
                  {isLoaded(c.name) && (
                    <span className="rbadge" style={{ marginLeft: 6 }}>
                      loaded
                    </span>
                  )}
                </td>
                <td className="num">{c.size_mb}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
