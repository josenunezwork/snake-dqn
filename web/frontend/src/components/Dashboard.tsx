import { useEffect, useState } from "react";
import { fetchMetrics } from "../api";
import type { DashboardData, EvalRow } from "../types";
import InfoDot from "./InfoDot";

// EvalRow already carries the dual-schema fields (metric/date/mass_ci).
type EnrichedRow = EvalRow;

const HEADER_HELP = {
  candidate: "The checkpoint being evaluated.",
  opponent:
    "Who the candidate played against. Difficulty depends on the opponent — rows with different opponents are not directly comparable.",
  mass:
    "Headline metric: mean per-frame snake length over the WHOLE eval horizon, dead frames counting 0. Rows tagged “old gate metric” instead average over alive frames only (the pre-redesign gate, which could reward dying rich).",
  max: "Best single-frame mass the candidate reached during the eval.",
  surv: "Survival: fraction of eval frames the candidate was alive (0 – 1).",
  n: "Number of eval games aggregated into this row.",
  date: "When the eval log file was written.",
};

function fmtDate(iso?: string | null): string {
  if (!iso) return "—";
  return iso.slice(0, 10);
}

export default function Dashboard({ checkpoint }: { checkpoint?: string | null }) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  const load = () => {
    setLoading(true);
    fetchMetrics()
      .then((d) => {
        setData(d);
        setFailed(false);
      })
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

  const rows = data.leaderboard as EnrichedRow[];
  // Mass bars are only comparable against the same opponent under the same
  // metric: normalize within each (metric, opponent) group, not globally.
  const groupMax = new Map<string, number>();
  for (const r of rows) {
    const k = `${r.metric ?? "legacy_mean_mass"}|${r.opponent}`;
    groupMax.set(k, Math.max(groupMax.get(k) ?? 1, r.mean_mass));
  }
  // Match the currently-loaded checkpoint so the promotion surfaces point at "you".
  const isLoaded = (name?: string | null) =>
    !!checkpoint && !!name && (name === checkpoint || name.endsWith("/" + checkpoint));

  return (
    <div className="panel">
      <div className="panel-intro">
        How trained checkpoints score head-to-head against frozen (fixed-weights) opponents.{" "}
        <InfoDot text="Each row is one candidate-vs-opponent eval aggregated from a logs/eval_*.json file. Opponents' weights are frozen so the score measures only the candidate. Rows with different opponents are not directly comparable." />
      </div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="section-title" style={{ margin: 0 }}>
          Eval leaderboard ({rows.length})
        </div>
        <button className="btn" onClick={load}>
          {loading ? "…" : "Refresh"}
        </button>
      </div>
      {failed && (
        <div className="error" role="alert" style={{ marginBottom: 6 }}>
          Refresh failed — showing previous results (may be stale).
        </div>
      )}
      <div className="card" style={{ padding: 0, overflow: "hidden", opacity: failed ? 0.7 : 1 }}>
        {rows.length === 0 ? (
          <div className="muted" style={{ fontSize: 12, padding: 12 }}>
            No eval results yet — run a tournament eval to populate the leaderboard.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th title={HEADER_HELP.candidate}>Candidate</th>
                <th title={HEADER_HELP.opponent}>Opponent</th>
                <th className="num" title={HEADER_HELP.mass}>
                  mass <InfoDot text={HEADER_HELP.mass} />
                </th>
                <th className="num" title={HEADER_HELP.max}>
                  max
                </th>
                <th className="num" title={HEADER_HELP.surv}>
                  surv
                </th>
                <th className="num" title={HEADER_HELP.n}>
                  n
                </th>
                <th className="num" title={HEADER_HELP.date}>
                  date
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const legacy = (r.metric ?? "legacy_mean_mass") !== "mass_integral";
                const max = groupMax.get(`${r.metric ?? "legacy_mean_mass"}|${r.opponent}`) ?? 1;
                return (
                  <tr key={i} className={isLoaded(r.file) || isLoaded(r.candidate) ? "me" : undefined}>
                    <td>
                      <div title={`${r.candidate} vs ${r.opponent} (${r.file})`}>
                        {r.candidate}
                        {(isLoaded(r.file) || isLoaded(r.candidate)) && (
                          <span className="rbadge" style={{ marginLeft: 6 }}>
                            loaded
                          </span>
                        )}
                        <span
                          className="rbadge"
                          style={{ marginLeft: 6, opacity: legacy ? 0.7 : 1 }}
                          title={
                            legacy
                              ? "Scored by the pre-redesign gate (mass averaged over alive frames only) — not comparable with mass-integral rows"
                              : "Scored by the repaired gate's mass integral (dead frames count 0)"
                          }
                        >
                          {legacy ? "old gate metric" : "mass integral"}
                        </span>
                      </div>
                      <div
                        className="massbar"
                        style={{ width: `${(r.mean_mass / max) * 100}%`, marginTop: 3 }}
                      />
                    </td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {r.opponent}
                    </td>
                    <td className="num" title={r.mass_ci != null ? `±${r.mass_ci} (95% CI)` : undefined}>
                      {r.mean_mass}
                    </td>
                    <td className="num">{r.max_mass}</td>
                    <td className="num">{r.survival}</td>
                    <td className="num">{r.n}</td>
                    <td className="num" title={r.date ?? undefined}>
                      {fmtDate(r.date)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="section-title">Checkpoints ({data.checkpoints.length})</div>
      <div className="card" style={{ padding: 0, overflow: "hidden", opacity: failed ? 0.7 : 1 }}>
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
