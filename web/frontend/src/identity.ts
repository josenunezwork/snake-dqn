const CLIENT_ID_KEY = "snake.clientId";
const PLAYER_NAME_KEY = "snake_player";
const MAX_CLIENT_ID_LENGTH = 64;

// Used only when browser storage is unavailable (privacy settings, embedded
// contexts, or quota failures). It keeps requests from one open page tied to
// the same opaque identity without making persistence a render-time requirement.
let runtimeClientId = "";
let storageUnavailable = false;

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function mintClientId(): string {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/** Collapse internal whitespace to match score-store name normalization. */
export function normalizePlayerName(name: string): string {
  return name.replace(/\s+/g, " ").trim();
}

/** Read the last display name without allowing unavailable storage to break the UI. */
export function getStoredPlayerName(): string {
  try {
    return storage()?.getItem(PLAYER_NAME_KEY) ?? "";
  } catch {
    return "";
  }
}

/** Persist the display name when possible; the caller can still use it if persistence fails. */
export function setStoredPlayerName(name: string): void {
  try {
    storage()?.setItem(PLAYER_NAME_KEY, name);
  } catch {
    // Storage is an optional convenience, not a prerequisite for playing.
  }
}

// Opaque per-browser id for leaderboard identity. Persist it when possible,
// otherwise retain one id for this JavaScript runtime.
export function getClientId(): string {
  const browserStorage = storage();
  if (!browserStorage) {
    storageUnavailable = true;
    if (runtimeClientId) return runtimeClientId;
    runtimeClientId = mintClientId().slice(0, MAX_CLIENT_ID_LENGTH);
    return runtimeClientId;
  }

  try {
    const stored = browserStorage.getItem(CLIENT_ID_KEY);
    if (stored) {
      runtimeClientId = stored.slice(0, MAX_CLIENT_ID_LENGTH);
      return runtimeClientId;
    }
    if (storageUnavailable && runtimeClientId) return runtimeClientId;
  } catch {
    storageUnavailable = true;
    if (runtimeClientId) return runtimeClientId;
  }

  runtimeClientId = mintClientId().slice(0, MAX_CLIENT_ID_LENGTH);
  try {
    browserStorage.setItem(CLIENT_ID_KEY, runtimeClientId);
  } catch {
    storageUnavailable = true;
    // The runtime id still provides stable identity for this open page.
  }
  return runtimeClientId;
}
