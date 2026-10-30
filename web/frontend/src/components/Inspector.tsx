import { memo } from "react";
import type { InspectorDTO, PathBState, StateGroup } from "../types";
import SectorRadar from "./SectorRadar";
import SteeringWheel from "./SteeringWheel";
import DecisionTrace from "./DecisionTrace";
import InfoDot from "./InfoDot";

function valueColor(v: number): string {
  // diverging: negative -> red, ~0 -> grey, positive -> teal
  if (v >= 0) return `rgba(56,189,248,${Math.min(0.25 + Math.abs(v), 1)})`;
  return `rgba(248,113,113,${Math.min(0.25 + Math.abs(v), 1)})`;
}

function FeatureRow({ idx, value }: { idx: number; value: number }) {
  const w = Math.min(Math.abs(value), 1) * 100;
  const left = value >= 0 ? 50 : 50 - w / 2;
  return (
    <div className="feat-row">
      <span className="idx">{idx}</span>
      <div className="feat-bar">
        <div
          className="feat-fill"
          style={{ left: `${left}%`, width: `${w / 2}%`, background: valueColor(value) }}
        />
      </div>
      <span className="feat-val">{value.toFixed(2)}</span>
    </div>
  );
}

// A 16-wide state group rendered as a radial windrose instead of 16 flat bars.
function isSectorGroup(g: StateGroup): boolean {
  return g.end - g.start === 16;
}

interface Props {
  inspector: InspectorDTO | null;
  tick?: number;
  heroId?: number;
  // Name of the loaded checkpoint (frame.checkpoint_name / session.checkpoint).
  // Part of the DecisionTrace reset key so loading a different checkpoint of
  // the same obs spec clears the trace instead of splicing two models' histories.
  checkpointName?: string | null;
  // Present only for an enabled Path B Watch controller. The controller can
  // fall back on every request and still remains an assisted hero by identity.
  pathb?: PathBState | null;
}

function actionName(action: number | null | undefined, labels: string[]): string {
  return typeof action === "number" && action >= 0 && action < labels.length
    ? labels[action]
    : "—";
}

function Inspector({
  inspector,
  tick = 0,
  heroId = 0,
  checkpointName = null,
  pathb = null,
}: Props) {
  if (!inspector) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-glyph">◎</div>
          <div className="empty-title">No snake to inspect</div>
          <div className="muted" style={{ fontSize: 12 }}>
            The inspected snake isn't alive. Click a live snake in the arena, or pick one in
            Controls, to watch its decision.
          </div>
        </div>
      </div>
    );
  }
  const { state, groups, q_values, action_labels, chosen, free_space } = inspector;
  const sorted = [...q_values].sort((a, b) => b - a);
  const margin = sorted.length > 1 ? sorted[0] - sorted[1] : 0;

  // The action the hero ACTUALLY took last step (post-masking / exploration),
  // when the backend serves it. Display it as "action taken"; the raw greedy
  // argmax (`chosen`) is shown separately when it differs.
  const executedRaw = inspector.executed_action;
  const acted =
    typeof executedRaw === "number" && executedRaw >= 0 && executedRaw < q_values.length
      ? executedRaw
      : chosen;
  const greedyDiffers = acted !== chosen;
  const pathbLast = pathb?.last ?? null;
  const pathbCounts = pathb?.counts ?? {};
  const pathbHeroId = pathb?.hero_id ?? 0;
  const inspectingPathbHero = pathb?.active === true && heroId === pathbHeroId;
  const pathbLastMatchesInspectedHero =
    inspectingPathbHero && pathbLast?.hero_id === pathbHeroId;
  const pathbAppliedThisTick =
    pathbLastMatchesInspectedHero && pathbLast?.frame === tick && pathbLast.applied === true;

  const foodGroup = groups.find((g) => isSectorGroup(g) && g.name.toLowerCase().includes("food"));
  const dangerGroup = groups.find((g) => isSectorGroup(g) && g.name.toLowerCase().includes("danger"));
  const heading = state.slice(0, 4);
  const foodMarker: [number, number] | null = state.length > 6 ? [state[5], state[6]] : null;

  return (
    <div className="panel">
      <div className="panel-intro">The AI's decision right now — what it sees and how it moves.</div>

      {pathb?.active === true && (
        <div className="card" aria-label="Assisted hero status">
          <div className="section-title">assisted hero #{pathbHeroId}</div>
          <div className="muted" style={{ fontSize: 11, lineHeight: 1.6 }}>
            Interventions {pathbCounts.applied ?? 0} · fallbacks {pathbCounts.fallback ?? 0}
            {" · "}deadline misses {pathbCounts.deadline_miss ?? 0}
            {" · "}stale discarded {pathbCounts.stale_discarded ?? 0}
          </div>
          {pathbLastMatchesInspectedHero ? (
            <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
              Proposed action: <span className="mono">{actionName(pathbLast?.proposed_action, action_labels)}</span>
              {" · "}executed action: <span className="mono">{actionName(pathbLast?.executed_action, action_labels)}</span>
              {pathbLast?.fallback_reason ? (
                <>
                  {" · "}fallback: <span className="mono">{pathbLast.fallback_reason}</span>
                </>
              ) : null}
            </div>
          ) : inspectingPathbHero ? (
            <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
              Waiting for the first assisted-hero decision.
            </div>
          ) : (
            <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
              Path B decision details belong to assisted hero #{pathbHeroId}; selected snake #{heroId}
              {" "}uses its own served controller.
            </div>
          )}
        </div>
      )}

      <div className="card">
        <div className="section-title">
          Steering · Q-values → action
          <InfoDot term="qvalue" />
          <InfoDot term="margin" />
          {greedyDiffers && <InfoDot term="actiontaken" />}
        </div>
        <div className="decision">
          <SteeringWheel
            q={q_values}
            chosen={acted}
            greedy={greedyDiffers ? chosen : null}
            labels={action_labels}
            resolvedMask={inspector.resolved_mask}
          />
          <div className="qgrid">
            {q_values.map((q, i) => (
              <div key={i} className={"qcell" + (i === acted ? " chosen" : "")}>
                <span className="qcell-k">
                  {i === acted ? "▸ " : ""}
                  {action_labels[i]}
                  {greedyDiffers && i === chosen && (
                    <span className="muted" style={{ fontSize: 9 }}> greedy</span>
                  )}
                </span>
                <span className="qcell-v mono">{q.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
        {inspector.replay_search_override && (
          <div role="status" className="muted" style={{ fontSize: 11, marginTop: 6 }}>
            Move from search: <span className="mono">{actionName(acted, action_labels)}</span>
            {" · "}bare action: <span className="mono">{actionName(inspector.replay_bare_action, action_labels)}</span>.
            {" "}Q-values show the bare network.
          </div>
        )}
        {greedyDiffers && !inspector.replay_search_override && (
          <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
            Action taken: <span className="mono">{action_labels[acted]}</span> · greedy pick:{" "}
            <span className="mono">{action_labels[chosen]}</span>{" "}
            {pathbAppliedThisTick
              ? "(assisted lookahead override selected the executed action)."
              : "(exploration or masking overrode the argmax)."}
          </div>
        )}
        {inspector.resolved_mask && (
          <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
            Resolved actions: {inspector.resolved_mask.map((allowed, index) =>
              allowed ? action_labels[index] : null
            ).filter(Boolean).join(", ") || "none"}
          </div>
        )}
        <DecisionTrace
          tick={tick}
          chosen={acted}
          margin={margin}
          labels={action_labels}
          resetKey={`${checkpointName ?? ""}:${heroId}:${inspector.input_size}`}
        />
      </div>

      {free_space && (
        <div className="card">
          <div className="section-title">
            Free-space (don't trap yourself)
            <InfoDot term="freespace" />
          </div>
          <div className="row" style={{ marginBottom: 0 }}>
            {["Left", "Straight", "Right"].map((lbl, i) => (
              <div key={lbl} style={{ flex: 1, textAlign: "center" }}>
                <div className="muted" style={{ fontSize: 11 }}>{lbl}</div>
                <div className="mono" style={{ fontSize: 18 }}>{free_space[i].toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(foodGroup || dangerGroup) && (
        <div className="card">
          <div className="section-title">
            Perception · what the agent senses around it
            <InfoDot term="perception" />
          </div>
          <div className="radar-row">
            {foodGroup && (
              <SectorRadar
                label="Food"
                color={[245, 158, 11]}
                values={state.slice(foodGroup.start, foodGroup.end)}
                heading={heading}
                marker={foodMarker}
              />
            )}
            {dangerGroup && (
              <SectorRadar
                label="Danger"
                color={[248, 113, 113]}
                values={state.slice(dangerGroup.start, dangerGroup.end)}
                heading={heading}
              />
            )}
          </div>
          <div className="muted" style={{ fontSize: 11, marginTop: 8, lineHeight: 1.6 }}>
            Screen-oriented — up on the dial is up in the arena. The pale arrow is the snake's
            heading{foodGroup ? "; the dashed line points at the nearest food" : ""}.
          </div>
        </div>
      )}

      <details className="state-details">
        <summary>
          <span className="section-title" style={{ margin: 0, display: "inline" }}>
            Full state vector ({state.length}-D)
          </span>
        </summary>
        <div className="diverging-legend" style={{ marginTop: 10 }}>
          <span className="muted mono">−1</span>
          <div className="legend-bar" />
          <span className="muted mono">0</span>
          <div className="legend-bar pos" />
          <span className="muted mono">+1</span>
        </div>
        {groups.map((grp) => (
          <div className="card" key={grp.name} style={{ padding: "8px 12px" }}>
            <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
              {grp.name} <span className="mono">[{grp.start}–{grp.end - 1}]</span>
            </div>
            {state.slice(grp.start, grp.end).map((v, j) => (
              <FeatureRow key={grp.start + j} idx={grp.start + j} value={v} />
            ))}
          </div>
        ))}
      </details>
    </div>
  );
}

export default memo(Inspector);
