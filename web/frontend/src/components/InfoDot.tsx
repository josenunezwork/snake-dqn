import { GLOSSARY } from "../lib/glossary";

// A tiny, dependency-free "ⓘ" affordance that reveals a glossary blurb on hover
// or keyboard focus. Positioned via CSS; no JS state so it can't get stuck open.
export default function InfoDot({ term, text }: { term?: string; text?: string }) {
  const body = text ?? (term ? GLOSSARY[term] : "") ?? "";
  if (!body) return null;
  return (
    <span className="infodot" tabIndex={0} role="note" aria-label={body}>
      <span className="infodot-mark" aria-hidden="true">
        i
      </span>
      <span className="infodot-pop" role="tooltip">
        {body}
      </span>
    </span>
  );
}
