"""GameSession: owns one live GameState + shared ApexPolicy and drives it.

The single source of truth for the running game. The FastAPI app steps this on a
background loop and broadcasts serialized frames; control messages mutate it.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Dict, Optional

import torch

from src.core.config_loader import load_config
from src.core.game_config import GameConfig, initialize_config
from src.data.score_store import compute_score
from src.game.game_state import GameState
from src.model.obs_spec import DEFAULT_OBS_SPEC, RASTER31V2, VECTOR61
from src.training.apex_policy import ApexPolicy
from web.backend.checkpoints import resolve_checkpoint_name

# The three ways to drive the shared game.
MODE_WATCH = "watch"  # AI plays itself; we observe.
MODE_TRAIN = "train"  # AI learns online.
MODE_PLAY = "play"  # A human controls snake 0 against the AI (scored).
VALID_MODES = (MODE_WATCH, MODE_TRAIN, MODE_PLAY)

# Version of the frame/control protocol the backend speaks. Serialized into
# every state frame (serialize.py reads it off the session) so the client can
# detect a stale UI build talking to a newer backend.
PROTOCOL_VERSION = 2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_DIR = os.path.join(REPO_ROOT, "saved_snakes")
CONFIG_61 = os.path.join(REPO_ROOT, "configs", "free_space_v2.yaml")
CONFIG_58 = os.path.join(REPO_ROOT, "configs", "default.yaml")
CONFIG_MECHANICS_V2 = os.path.join(REPO_ROOT, "configs", "mechanics_v2.yaml")
DEFAULT_CHECKPOINT = os.path.join(SAVED_DIR, "champion_a5_freespace_20260621.pth")


def _read_input_size(checkpoint_path: str) -> int:
    """Best-effort read of a checkpoint's input_size to pick a matching config."""
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("input_size",):
            if key in blob:
                return int(blob[key])
        cfg = blob.get("config") or blob.get("apex_config") or {}
        if isinstance(cfg, dict) and "input_size" in cfg:
            return int(cfg["input_size"])
    except Exception:
        pass
    return 61


def _reward_contract_mismatch(checkpoint_path: str) -> bool:
    """Whether a checkpoint's recorded reward economics differ from the active config's.

    Mirrors the semantics of ``validate_checkpoint_contract``'s reward check: a
    checkpoint with no recorded version is implicitly v1, and any field-level
    difference counts. Used to (a) refuse a doomed train-mode build up front and
    (b) tell the client the refusal is overridable (deliberate fine-tune), not
    fatal. Best-effort: unreadable metadata reports no mismatch and the real
    validator stays authoritative during the build.
    """
    try:
        from src.core.reward_contract import current_reward_contract
        from src.training.checkpoint_contract import checkpoint_contract_values

        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current = current_reward_contract()
        recorded_maps = [
            m for m in checkpoint_contract_values(blob, "reward_contract") if isinstance(m, dict)
        ]
        if not recorded_maps:
            # Pre-contract checkpoints are implicitly reward v1.
            return float(current.get("version", 1)) != 1.0
        for recorded in recorded_maps:
            for key, expected in current.items():
                raw = recorded.get(key, 1 if key == "version" else None)
                if raw is None:
                    return True
                try:
                    if not math.isclose(float(raw), float(expected), rel_tol=1e-7, abs_tol=1e-9):
                        return True
                except (TypeError, ValueError):
                    return True
        return False
    except Exception:
        return False


def _read_obs_spec(checkpoint_path: str) -> str:
    """Best-effort read of a checkpoint's obs_spec.

    Absent metadata means a pre-contract-v2 vector champion, so we fall back to
    :data:`~src.model.obs_spec.DEFAULT_OBS_SPEC` (``vector61``) — every historical
    checkpoint keeps loading unchanged. Delegates the actual parsing (including
    ``apex_config``/``config`` nesting) to :meth:`InferenceAgent._detect_obs_spec`.
    """
    try:
        from src.model.inference_agent import InferenceAgent

        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return InferenceAgent._detect_obs_spec(blob)
    except Exception:
        return DEFAULT_OBS_SPEC


def _mechanics_v2_enabled() -> bool:
    """Whether the served game should run mechanics v2 (default on).

    Opt out with ``SNAKE_MECHANICS_V2=0`` to serve the legacy v1 mechanics.
    Read per-build so the flag can be flipped without re-importing.
    """
    return os.environ.get("SNAKE_MECHANICS_V2", "1").strip().lower() not in ("0", "false", "no")


def _config_for(input_size: int) -> str:
    """Pick the session config for a checkpoint's input width.

    61-D checkpoints get the mechanics-v2 arena (configs/mechanics_v2.yaml —
    same champion_a5 checkpoint contract, mechanics/reward v2) when present and
    not opted out via ``SNAKE_MECHANICS_V2=0``. Legacy 58-D checkpoints always
    stay on the v1 default config, since mechanics_v2.yaml is a 61-D contract.
    """
    if input_size >= 61:
        if _mechanics_v2_enabled() and os.path.exists(CONFIG_MECHANICS_V2):
            return CONFIG_MECHANICS_V2
        return CONFIG_61
    return CONFIG_58


class GameSession:
    """A single running game the whole server shares (one game, many viewers)."""

    def __init__(self, checkpoint: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self.playing: bool = True
        self.speed: float = 12.0  # frames per second
        self.mode: str = MODE_WATCH
        self.hero_id: int = 0
        self.checkpoint_path: Optional[str] = None
        self.config_path: str = CONFIG_61
        self.obs_spec: str = VECTOR61
        self.last_error: Optional[str] = None
        # True when last_error can be resolved by re-issuing the action with the
        # reward-contract override (deliberate v1->v2 fine-tune), so the client
        # can offer "Fine-tune anyway" instead of a dead-end error.
        self.last_error_overridable: bool = False
        # True while the live training policy was built with the reward override
        # active (surfaced in the Train blurb: warm start, not a resume).
        self.reward_override_active: bool = False

        # Protocol/connection bookkeeping. viewer_count is maintained by the
        # web app (number of connected WebSocket clients). raster_subscribers
        # (count of connections that asked for the raster block, letting the
        # serializer skip the ~3.8k-int payload nobody is looking at) is
        # intentionally NOT initialized here: the serializer defaults a missing
        # attr to "on" (getattr(session, 'raster_subscribers', 1)), so
        # standalone sessions (tests, scripts) keep serving the raster block.
        # It appears once the first set_raster_stream control arrives.
        self.protocol_version: int = PROTOCOL_VERSION
        self.viewer_count: int = 0
        self._raster_conns: set = set()

        # Human-play run state (only meaningful in MODE_PLAY).
        self.human_id: Optional[int] = None
        self.play_opponents: Optional[int] = None  # AI opponent count (None = config default)
        self.run_started: bool = False  # gate movement until the first input
        self.run_frames: int = 0
        self.run_started_at: float = 0.0
        self.run_over: bool = False
        self.last_run: Optional[dict] = None  # finalized, server-authoritative stats
        self.submitted: bool = False
        # Connection that owns the current scored run (set by the web app on
        # new_game / first accepted input); other connections may watch but not
        # steer or destroy the run while it is live.
        self.run_owner: Optional[object] = None

        ckpt = checkpoint or (DEFAULT_CHECKPOINT if os.path.exists(DEFAULT_CHECKPOINT) else None)
        self._build(ckpt, mode=MODE_WATCH)

    # -- construction -------------------------------------------------------
    def _build(
        self, checkpoint: Optional[str], mode: str, override_reward_contract: bool = False
    ) -> None:
        obs_spec = _read_obs_spec(checkpoint) if checkpoint else VECTOR61
        # Raster checkpoints are served forward-only (no online training path
        # through AISnake's vector replay). A train request on a raster champion
        # falls back to watch so the build still succeeds — but say so, instead
        # of silently dropping out of train mode (e.g. when a raster checkpoint
        # is loaded while training).
        if obs_spec == RASTER31V2 and mode == MODE_TRAIN:
            mode = MODE_WATCH
            self.last_error = (
                "Raster (raster31v2) checkpoints are served forward-only; " "dropped to watch mode."
            )
        training = mode == MODE_TRAIN
        human = mode == MODE_PLAY
        input_size = _read_input_size(checkpoint) if checkpoint else 61
        config_path = _config_for(input_size)
        initialize_config(load_config(config_path))

        if obs_spec == RASTER31V2:
            from web.backend.raster_policy import RasterServingPolicy

            if not checkpoint:
                raise RuntimeError("raster31v2 serving requires a checkpoint.")
            policy = RasterServingPolicy.from_checkpoint(checkpoint)
        else:
            policy = ApexPolicy(
                input_size=GameConfig.INPUT_SIZE,
                hidden_size=GameConfig.HIDDEN_SIZE,
                output_size=GameConfig.OUTPUT_SIZE,
                training=training,
                override_reward_contract=training and override_reward_contract,
            )
            if checkpoint:
                if not policy.load_checkpoint(checkpoint):
                    raise RuntimeError(f"Failed to load checkpoint: {checkpoint}")
        policy.epsilon = 0.1 if training else 0.0
        # Record whether this build actually leaned on the reward escape hatch,
        # so the UI can label the run "fine-tune under current rewards", and
        # clear any stale overridable-error flag from the failed attempt.
        self.reward_override_active = bool(
            training
            and override_reward_contract
            and checkpoint
            and _reward_contract_mismatch(checkpoint)
        )
        self.last_error_overridable = False

        # In play mode the arena is 1 human + N AI opponents; N is adjustable for
        # difficulty. Watch/train use the config's snake count.
        num_snakes = GameConfig.NUM_SNAKES
        if human and self.play_opponents is not None:
            num_snakes = max(2, min(12, int(self.play_opponents) + 1))
        game = GameState(
            headless=True,
            num_snakes=num_snakes,
            shared_policy=policy,
            human_mode=human,
        )

        # The raster policy reads the live game each frame to build observations
        # through the shared featurizer; give it the game it now serves.
        if obs_spec == RASTER31V2:
            policy.attach_game(game)

        self.policy = policy
        self.game = game
        self.checkpoint_path = checkpoint
        self.config_path = config_path
        self.obs_spec = obs_spec
        self.mode = mode

        if human:
            human_snake = self._find_human()
            self.human_id = human_snake.id if human_snake else None
            # The human's death ends the run; AI opponents keep respawning.
            if human_snake is not None:
                human_snake.auto_respawn = False
            self.hero_id = self.human_id if self.human_id is not None else 0
            self._start_run()
        else:
            self.human_id = None
            self.hero_id = game.snakes[0].id if game.snakes else 0

    # -- stepping -----------------------------------------------------------
    def step(self) -> None:
        with self._lock:
            if not self.playing:
                return
            if self.mode == MODE_TRAIN:
                # Reset on wipeout, or (mechanics v2) when the training population
                # floor is hit — otherwise lone-survivor frames skew replay
                # (blueprint §1.7). The property is False at v1 and outside train.
                if self.game.alive_snakes <= 0 or self.game.population_floor_reached:
                    self.game.reset()
                self.game.update(train_mode=True, learn=True, allow_respawn=False)
            elif self.mode == MODE_PLAY:
                self._step_play()
            else:
                self.game.update(train_mode=False, learn=False, allow_respawn=True)
            self._enforce_food_target()

    def _enforce_food_target(self) -> None:
        """Trim food back to the live target.

        The game's spawn-on-eat ratchets food above max_food over time
        (maintain_count only tops up, never trims), so the Food control wouldn't
        otherwise hold.
        """
        fm = self.game.food_manager
        # Trim only ambient overage — corpse/boost-trail pellets (mechanics v2)
        # are cap-exempt and must survive so kill economics stay visible.
        fm.trim_ambient(fm.max_food)

    def _step_play(self) -> None:
        """Advance one frame of a human-vs-AI scored game.

        The world stays frozen until the player's first input (so the snake
        doesn't run into a wall before they're ready). The human
        (auto_respawn=False) stays dead on collision so the run can be finalized
        and submitted; AI opponents respawn as usual (allow_respawn).
        """
        if not self.run_started:
            return
        self.game.update(train_mode=False, learn=False, allow_respawn=True)
        human = self._find_human()
        if human is not None and human.is_alive and not self.run_over:
            self.run_frames += 1
        if human is not None and not human.is_alive and not self.run_over:
            self._finalize_run()

    def snapshot(self) -> dict:
        from web.backend.serialize import build_frame

        with self._lock:
            return build_frame(self)

    # -- controls -----------------------------------------------------------
    def set_playing(self, value: bool) -> None:
        self.playing = bool(value)

    def set_speed(self, fps: float) -> None:
        self.speed = max(1.0, min(120.0, float(fps)))

    def set_epsilon(self, eps: float) -> None:
        with self._lock:
            self.policy.epsilon = max(0.0, min(1.0, float(eps)))

    def set_hero(self, snake_id: int) -> None:
        with self._lock:
            ids = [s.id for s in self.game.snakes]
            if int(snake_id) in ids:
                self.hero_id = int(snake_id)

    def set_food(self, target) -> None:
        """Set the live food target (count maintained on the board).

        maintain_count() only tops up to the target, so an immediate decrease
        trims the excess here. Persisted on the GameState so resets keep it.
        """
        with self._lock:
            target = max(0, min(1000, int(target)))
            fm = self.game.food_manager
            fm.max_food = target
            # reset() / new spawns read these effective counts on the GameState.
            self.game._effective_max_food = target
            self.game._effective_initial_food = target
            if fm.ambient_count > target:
                fm.trim_ambient(target)
            else:
                fm.maintain_count(self.game.snakes)

    def reset_game(self) -> None:
        with self._lock:
            self.game.reset()
            # The raster policy caches per-frame observations keyed by the frame
            # counter; a reset rewinds it to 0, so drop the stale cache.
            invalidate = getattr(self.policy, "_invalidate_cache", None)
            if callable(invalidate):
                invalidate()
            if self.mode == MODE_PLAY:
                self._start_run()

    def save_weights(self) -> str:
        """Persist the live policy's weights to a fresh checkpoint in SAVED_DIR.

        The safe counterpart to the destructive rebuild paths (set_mode /
        load_checkpoint), which construct a fresh policy from the on-disk
        checkpoint and would otherwise silently discard everything learned in
        train mode.

        Returns:
            The written checkpoint's basename (``web_train_YYYYMMDD_HHMMSS.pth``).

        Raises:
            RuntimeError: If the served policy is forward-only (raster serving).
        """
        with self._lock:
            get_state_dict = getattr(self.policy, "get_state_dict", None)
            if not callable(get_state_dict):
                raise RuntimeError("the served policy is forward-only and cannot be saved")
            os.makedirs(SAVED_DIR, exist_ok=True)
            stem = time.strftime("web_train_%Y%m%d_%H%M%S")
            for suffix in range(10_000):
                name = f"{stem}{'' if suffix == 0 else f'_{suffix}'}.pth"
                path = os.path.join(SAVED_DIR, name)
                try:
                    # Exclusive creation also protects two GameSession instances
                    # saving in the same second.
                    checkpoint_file = open(path, "xb")
                except FileExistsError:
                    continue
                try:
                    with checkpoint_file:
                        torch.save(get_state_dict(), checkpoint_file)
                    return name
                except BaseException:
                    # A failed torch.save can leave a partial .pth file.  It
                    # must not enter the catalog as a corrupt checkpoint.
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    raise
            raise RuntimeError("could not allocate a unique checkpoint filename")

    # -- per-connection state -----------------------------------------------
    def set_raster_stream(self, conn_id: object, value: object) -> None:
        """Record one connection's raster-stream subscription.

        ``raster_subscribers`` is the count of connections that currently want
        the raster block; the serializer skips the ~3.8k-int payload when it is
        zero. ``value`` is ``{"on": bool}`` (or a bare bool) per the control
        contract. Idempotent per connection — repeated ``on`` messages from the
        same connection count once.
        """
        on = value.get("on") if isinstance(value, dict) else value
        with self._lock:
            if bool(on):
                self._raster_conns.add(conn_id)
            else:
                self._raster_conns.discard(conn_id)
            self.raster_subscribers = len(self._raster_conns)

    def drop_connection(self, conn_id: object) -> None:
        """Forget a disconnected websocket's per-connection state.

        Clears its raster subscription and, if it owned the live scored run,
        releases ownership so the shared session cannot be locked by a ghost.
        """
        with self._lock:
            self._raster_conns.discard(conn_id)
            # Only maintain the counter once a subscription message has ever
            # materialized it — sessions without the attr default to "on".
            if hasattr(self, "raster_subscribers"):
                self.raster_subscribers = len(self._raster_conns)
            if self.run_owner == conn_id:
                self.run_owner = None

    # -- human play ---------------------------------------------------------
    def _find_human(self):
        """Return the human-controlled snake, or None if not in play mode."""
        from src.game.human_snake import HumanSnake

        for snake in self.game.snakes:
            if isinstance(snake, HumanSnake):
                return snake
        return None

    def _start_run(self) -> None:
        """(Re)start a scored human run: reset counters and the finalize latch."""
        human = self._find_human()
        if human is not None:
            human.auto_respawn = False
            if hasattr(human, "start_run"):
                human.start_run()
        self.run_started = False
        self.run_frames = 0
        self.run_started_at = time.time()
        self.run_over = False
        self.last_run = None
        self.submitted = False
        self.run_owner = None

    def _finalize_run(self) -> None:
        """Capture the just-ended human run as the server-authoritative result."""
        human = self._find_human()
        length = int(getattr(human, "length", 1)) if human else 1
        food = int(getattr(human, "run_food_eaten", 0)) if human else 0
        kills = int(getattr(human, "run_kills", 0)) if human else 0
        frames = int(self.run_frames)
        duration = max(0.0, time.time() - self.run_started_at) if self.run_started_at else 0.0
        self.last_run = {
            "score": compute_score(length, food, kills, frames),
            "length": length,
            "food_eaten": food,
            "kills": kills,
            "frames": frames,
            "duration_seconds": round(duration, 2),
            "checkpoint": (
                os.path.basename(self.checkpoint_path) if self.checkpoint_path else None
            ),
            "mechanics_version": int(GameConfig.MECHANICS_VERSION),
        }
        self.run_over = True

    def human_input(self, direction_name: str) -> None:
        """Apply a named absolute-direction input to the human snake.

        The first ACCEPTED input of a run also un-freezes the world (see
        _step_play) and starts the clock, so duration reflects actual play
        time. A rejected input (unknown name, or a 180° reversal of the spawn
        facing) must not launch the snake in a direction the player didn't
        choose.
        """
        with self._lock:
            human = self._find_human()
            if human is None or not human.is_alive or self.run_over:
                return
            accepted = human.apply_direction_input(direction_name)
            if accepted and not self.run_started:
                self.run_started = True
                self.run_started_at = time.time()

    def human_boost(self, boosting) -> None:
        """Toggle the human snake's speed boost."""
        with self._lock:
            human = self._find_human()
            if human is not None and hasattr(human, "set_boost"):
                human.set_boost(bool(boosting))

    def set_play_opponents(self, count) -> None:
        """Set the AI-opponent count for play mode and restart the run.

        Clamped to 1..11 opponents (2..12 total snakes). No-op outside play mode
        except to remember the choice for the next time play mode is entered.
        """
        with self._lock:
            self.play_opponents = max(1, min(11, int(count)))
            if self.mode == MODE_PLAY:
                self._build(self.checkpoint_path, mode=MODE_PLAY)

    def pending_submission(self) -> Optional[dict]:
        """Return the finalized-but-unsubmitted run without consuming it (read-only)."""
        with self._lock:
            if self.run_over and not self.submitted and self.last_run is not None:
                return dict(self.last_run)
            return None

    def claim_submission(self) -> Optional[dict]:
        """Atomically take the finalized run for submission, exactly once.

        Checks *and* latches ``submitted`` under a single lock acquisition so two
        near-simultaneous ``POST /api/scores`` calls (double-click / retry) can't
        both record the same run. Returns the run stats to persist, or ``None`` if
        there is nothing to submit or it was already claimed.
        """
        with self._lock:
            if self.run_over and not self.submitted and self.last_run is not None:
                self.submitted = True
                return dict(self.last_run)
            return None

    def release_submission(self) -> None:
        """Un-claim a run whose persistence failed, so submission can be retried.

        The claim/write/latch order matters: ``claim_submission`` latches
        ``submitted`` up front (double-submit guard), so if the DB write then
        fails the caller must release the claim — otherwise the finished run
        would be permanently unsubmittable and the UI would show a false
        "submitted" state.
        """
        with self._lock:
            if self.run_over and self.last_run is not None:
                self.submitted = False

    def set_mode(self, mode: str, override_reward_contract: bool = False) -> None:
        """Switch between watch / train / play, rebuilding the game as needed.

        A no-op (no rebuild, no arena reset) when the requested mode equals the
        current mode, or when it could not take effect anyway (train on a
        forward-only raster checkpoint, or train on a checkpoint whose reward
        economics predate the active config — both report why via last_error
        instead of silently resetting the arena). The reward case is
        overridable: re-issuing with ``override_reward_contract=True`` performs
        a deliberate fine-tune under the current rewards (warm start, not a
        resume) via the contract validator's escape hatch.
        """
        mode = str(mode) if str(mode) in VALID_MODES else MODE_WATCH
        with self._lock:
            if mode == self.mode:
                return
            if mode == MODE_TRAIN and self.obs_spec == RASTER31V2:
                # Forward-only checkpoint: the rebuild would coerce back to
                # watch, destroying the current game for nothing. Skip it and
                # surface the reason through the existing error pipeline.
                self.last_error = (
                    "Raster (raster31v2) checkpoints are served forward-only; "
                    "train mode is unavailable for this model."
                )
                self.last_error_overridable = False
                return
            if (
                mode == MODE_TRAIN
                and not override_reward_contract
                and self.checkpoint_path
                and _reward_contract_mismatch(self.checkpoint_path)
            ):
                # Refuse up front (no doomed build, arena preserved) and mark the
                # refusal overridable so the client offers "Fine-tune anyway".
                self.last_error = (
                    "This checkpoint was trained under different reward economics "
                    "(reward contract v1) than the arena is running (v2). "
                    "Training would be a fine-tune under the new rewards, not a resume."
                )
                self.last_error_overridable = True
                return
            try:
                self.last_error = None
                self.last_error_overridable = False
                self._build(
                    self.checkpoint_path,
                    mode=mode,
                    override_reward_contract=override_reward_contract,
                )
            except Exception as exc:  # contract mismatch etc. -> fall back to watch
                self.last_error = f"Cannot enter {mode} mode: {exc}"
                self.last_error_overridable = False
                if self.mode != MODE_WATCH:
                    self._build(self.checkpoint_path, mode=MODE_WATCH)

    def load_checkpoint(self, name: str, override_reward_contract: bool = False) -> None:
        """Load a checkpoint named by an untrusted client (a WebSocket control).

        The name reaches ``torch.load``, which unpickles, so only two shapes are
        accepted (matching what ``metrics.list_checkpoints`` advertises):

        * a plain basename naming a real ``.pth`` directly inside SAVED_DIR;
        * a repo-relative ``runs/**/latest_pqn.pth`` training output.

        ``basename``/``realpath`` containment defeats ``..``, absolute paths,
        and symlinks pointing outside the allowed roots. Report only the
        sanitized name, so a rejection cannot confirm arbitrary paths back to
        the client.

        Loading while in train mode enforces the reward contract like
        ``set_mode``; a mismatch is refused with an overridable error unless
        ``override_reward_contract`` is set (deliberate fine-tune).
        """
        raw = name if isinstance(name, str) else ""
        safe = os.path.basename(raw)
        entry = resolve_checkpoint_name(raw, REPO_ROOT, SAVED_DIR, os.path.join(REPO_ROOT, "runs"))
        path = entry.path if entry is not None else ""
        if entry is not None:
            safe = entry.name
        with self._lock:
            if not path or not safe.endswith(".pth"):
                self.last_error = f"Unknown checkpoint: {safe or '(none)'}"
                self.last_error_overridable = False
                return
            if (
                self.mode == MODE_TRAIN
                and not override_reward_contract
                and _reward_contract_mismatch(path)
            ):
                self.last_error = (
                    f"{safe} was trained under different reward economics than the "
                    "arena is running. Loading it in train mode would be a fine-tune "
                    "under the new rewards, not a resume."
                )
                self.last_error_overridable = True
                return
            try:
                self.last_error = None
                self.last_error_overridable = False
                self._build(path, mode=self.mode, override_reward_contract=override_reward_contract)
            except Exception as exc:
                self.last_error = f"Failed to load {safe}: {exc}"
                self.last_error_overridable = False
                # fall back to a clean watch build of the previous/default ckpt
                self._build(self.checkpoint_path, mode=MODE_WATCH)

    # -- state report -------------------------------------------------------
    def control_state(self) -> Dict[str, object]:
        with self._lock:
            return {
                "playing": self.playing,
                "speed": self.speed,
                "mode": self.mode,
                "training": bool(getattr(self.policy, "training", False)),
                "hero_id": self.hero_id,
                "checkpoint": (
                    os.path.basename(self.checkpoint_path) if self.checkpoint_path else None
                ),
                "config": os.path.basename(self.config_path),
                "mechanics_version": int(GameConfig.MECHANICS_VERSION),
                "obs_spec": str(self.obs_spec),
                "input_size": int(GameConfig.INPUT_SIZE),
                "num_snakes": int(len(self.game.snakes)),
                "epsilon": float(getattr(self.policy, "epsilon", 0.0)),
                "food_target": int(self.game.food_manager.max_food),
                "food_count": int(len(self.game.food_manager.food)),
                "error": self.last_error,
                # True when re-issuing the failed action with the reward-contract
                # override would succeed (client offers "Fine-tune anyway").
                "error_overridable": bool(self.last_error_overridable),
                # True while the live training run leans on the reward override
                # (fine-tune under current rewards, not a resume).
                "reward_override_active": bool(self.reward_override_active),
            }

    def play_state(self) -> Optional[Dict[str, object]]:
        """Per-frame human-play status, or None when not in play mode.

        Reports live run stats and, once the human has died, the finalized
        server-authoritative result awaiting name submission.
        """
        with self._lock:
            if self.mode != MODE_PLAY:
                return None
            human = self._find_human()
            alive = bool(getattr(human, "is_alive", False)) if human else False
            length = int(getattr(human, "length", 1)) if human else 1
            food = int(getattr(human, "run_food_eaten", 0)) if human else 0
            kills = int(getattr(human, "run_kills", 0)) if human else 0
            live_score = compute_score(length, food, kills, self.run_frames)
            return {
                "active": True,
                "human_id": self.human_id,
                "opponents": max(0, len(self.game.snakes) - 1),
                "human_alive": alive,
                "length": length,
                "food_eaten": food,
                "kills": kills,
                "frames": int(self.run_frames),
                "score": live_score,
                "run_started": bool(self.run_started),
                "run_over": bool(self.run_over),
                "submitted": bool(self.submitted),
                "pending": self.last_run if (self.run_over and not self.submitted) else None,
            }
