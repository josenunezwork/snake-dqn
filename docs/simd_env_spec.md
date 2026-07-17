# SIMD Env Parity Spec — Exact v1 & v2 Game Dynamics

Authoritative, file:line-precise description of the live simulator's per-frame
dynamics so a from-scratch **batched / vectorized** reimplementation (`src/simd_env/`)
can reproduce them **bit-for-bit**. This is a read-only extraction of the shipped
code as of P0+P1 (mechanics/reward versioning landed). Where v1 and v2 differ, both
are specified.

Primary sources:
- `src/game/game_state.py` — per-frame orchestration (`GameState.update`, `handle_collisions`).
- `src/game/snake.py` — movement, boost, growth, lifecycle (`Snake.move`, `grow`, `die`, `respawn`).
- `src/game/game_logic.py` — collision detection (`GameLogic.check_collisions` and helpers).
- `src/game/food_manager.py` — food pools, spawn, consume, maintain.
- `src/core/mechanics_constants.py` — cell lattice + v2 constants.
- `src/core/reward_events.py` — pure v2 reward (`compute_reward_v2`).
- `src/game/snake_reward.py` — reward dispatch (`calculate_reward` → v2 path).
- `src/game/ai_snake.py` — action decode + `move()` call (`AISnake.update`).

Key config (defaults; `configs/mechanics_v2.yaml` is the v2 training target):
`segment_size=10`, `wall_thickness=10`, `width=1450`, `height=830`,
`min_boost_length=5`, `boost_length_cost_frames=3`, `max_length=150` (v2 cfg) /
`100` (dataclass default), `frame_rate=100` (default) / `1` (v2 cfg),
`initial_food=250`, `max_food=300`, `num_snakes=6` (v2 cfg).
Reward v2 gamma = `GameConfig.APEX_GAMMA` = `apex.gamma` = `0.99`.

> **Scope note.** The full live env couples a neural policy, per-snake state
> tensors, action masking and replay into each step. For a *dynamics* parity
> harness the batched sim replays a **recorded action log** (per snake, per frame,
> the integer action 0–5) rather than reproducing network inference. Everything
> below is the deterministic world-update given those actions. §8 lists the exact
> determinism constraints the harness must impose.

---

## 0. Coordinate system & cell lattice (all versions)

- Positions are integer pixel `(x, y)`. Movement is cardinal in steps of
  `segment_size` (`s`, =10). `direction ∈ {(0,-1),(1,0),(0,1),(-1,0)}` = up,right,down,left.
- **Cell index** = floor division: `cell(p) = (p.x // s, p.y // s)`
  (`mechanics_constants.cell_index`, l.128-141). Floor division, so negative
  (out-of-bounds) coords map to negative cells — they do **not** alias to cell 0.
- **`same_cell(a,b,s)`** = `cell(a)==cell(b)` (l.144-164). This is THE collision /
  pickup predicate at all mechanics versions.
- **`snap_to_cell(p,s)`** = `((p.x//s)*s, (p.y//s)*s)` (l.103-125). Every random
  spawn (food + snake) is snapped so all entities share one lattice; on that
  lattice `same_cell` is exactly equivalent to the legacy `distance < s` radius
  test. **The batched sim must store snapped integer positions and compare by cell
  index — never by float distance** (except the two head-swap / mask paths noted
  in §3.2 and §7 that still use `GameLogic.distance`).

---

## 1. Per-frame STEP ORDER (`GameState.update`, game_state.py l.266-441)

Executed once per `update()` call. Numbered exactly as the code runs (the
docstring's 1–9 numbering is slightly reordered vs. actual execution; the list
below is the **actual** order).

**Inputs:** `train_mode: bool`, `learn: bool`, `allow_respawn: Optional[bool]`.
- `allow_respawn` defaults to `not train_mode` (l.295-296).
- `self._train_mode = bool(train_mode)` (l.297).

0. **Clear per-snake move traces** (l.299-301): for every snake with
   `last_move_positions`, set `snake.last_move_positions = []`. (Note: `move()`
   also clears these; this pre-clear matters for snakes that will NOT move this
   frame, e.g. dead ones, so a stale trace can't leak into collision checks.)

1. **Increment frame** (l.304): `self.frame += 1`.

2. **Maintain food count** (l.307): `food_manager.maintain_count(self.snakes)`.
   Tops ambient food up to `max_food` (see §4.4). **RNG is drawn here** (§5).

3. **Count alive** (l.310): `alive_snakes = sum(is_alive)`.

4. **Respawn handling** — only if `allow_respawn` (l.314-343):
   a. Decrement `respawn_timer` for each dead snake with `timer > 0` (l.315-317).
   b. For each dead snake with `timer <= 0` and `auto_respawn != False`
      (l.320-333): `new_pos = GameLogic.find_empty_position(...)`; if found,
      `snake.respawn(new_pos)`, `alive_snakes += 1`, `any_respawned = True`.
      **RNG is drawn inside `find_empty_position`** (§5). Iteration is over
      `self.snakes` in list order.
   c. If any respawned, invalidate every snake's carried selection cache
      (l.339-343). (No world effect; only affects neural selection — irrelevant
      to a replayed-action harness.)
   > For a **training/replay parity harness** `train_mode=True` ⇒ `allow_respawn=False`,
   > so step 4 is a no-op and no respawn RNG is drawn. **Recommended: parity runs
   > use `allow_respawn=False` (terminal deaths).**

5. **Action selection + MOVE ALL** (l.345-355):
   - Build `pre_move_snapshots` = shallow copies of every snake (segments +
     `last_move_positions` copied) BEFORE anyone moves (l.348, via
     `_snapshot_snake_for_observation`, l.258-264). This makes all snakes observe
     the same pre-frame world (no order bias).
   - For each **alive** snake in list order (l.349-355): call
     `snake.update(observation_snakes, self.food)`. In a replayed-action harness
     this reduces to: decode the logged action → set direction → set boost flag →
     `snake.move()` (see §2, §6a). **Movement mutates segment lists in place; the
     order is list order but movement is independent per snake, so order does not
     affect the world** (collisions are resolved later in step 8).

6. **v2 boost trail pellets** (l.360-367): for each snake, for each cell in
   `snake.pending_trail_pellets` (populated by `move()` under v2 only), if the
   cell is inside the arena (`_segment_inside_arena`), `food_manager.add_food(cell,
   corpse=True)`; then clear `pending_trail_pellets`. **No RNG.** At v1 the list is
   always empty ⇒ no-op.

7. **FOOD consumption** (l.369-391): for each **alive** snake in list order:
   - `ate = check_food_consumption(snake)` (l.483-498): checks every position in
     `snake.last_move_positions` (or `[snake.head]` if empty) via
     `food_manager.consume_at(pos, segment_size)`; on the **first** cell that has
     food, `snake.grow()` and return True. (So a boost 2-step that passes over two
     pellets eats only the first-checked one this frame; the trailing head cell's
     pellet, if any, is checked second — see §4.1.)
   - Record `ate_food_map[snake.id] = ate`. If ate: `episode_food_eaten += 1`,
     update `episode_best_length`.
   - **If `ate and not train_mode`**: `food_manager.spawn(1, self.snakes)` — spawns
     one replacement pellet immediately (**RNG**, §5).
   - `self.frame_ate_food = ate_food_map` (l.385).
   - **If `train_mode and any snake ate`** (l.387-391): a single
     `food_manager.maintain_count(self.snakes)` (**RNG**, §5) to keep post-eat food
     count in the same world the snake will next observe.
   > **Food replacement RNG differs by mode.** Not-train: one `spawn(1)` per eating
   > snake, interleaved in snake order. Train: no per-eat spawn; one bulk
   > `maintain_count` after the loop iff anyone ate. The batched sim MUST branch on
   > `train_mode` here to match the RNG draw sequence.

8. **Collision detection + resolution** (l.394): `frame_collisions = self.handle_collisions()`
   — the single source of truth (see §3). Sets `self.frame_collisions`,
   `self.frame_kills`, `self.frame_death_causes`, updates `episode_deaths`,
   `episode_kills`, `episode_collision_counts`. **Corpse food dropped here** (§4.5).
   **No RNG** in collision handling itself (food drops are deterministic cell copies).

9. **REWARD** (l.397-407): for each snake with `compute_reward_and_train`, compute
   reward from `ate_food_map[id]`, `collided = id in frame_collisions`, and
   `self.frame_kills`. See §6. (For non-AI snakes with no such method, skipped.)

10. **Centralized training** (l.409-428): every `TRAIN_FREQUENCY` frames, one
    `shared_policy.train_step()`. **Irrelevant to world dynamics / parity of the
    environment** (only mutates network weights). A dynamics harness ignores this.

11. **Episode bookkeeping** (l.430-441): update `episode_current_reward`,
    `episode_best_reward`, `episode_best_length`, recompute `alive_snakes`.

**`population_floor_reached`** (l.443-460): property, not a step. True iff
`MECHANICS_VERSION==2` AND last update was train_mode AND `num_snakes >=
POPULATION_FLOOR_V2(=3)` AND `alive_snakes < 3`. Training loops use it as an
episode-end signal. No world mutation.

---

## 2. MOVEMENT (`Snake.move`, snake.py l.134-177)

Called once per alive snake per frame (from `AISnake.update`, l.481). Direction
and `is_boosting` are already set by action decode (§6a) before `move()` runs.

```
move():
  last_move_positions = []
  pending_trail_pellets = []
  new_head = head + direction * s              # single cardinal step (l.145-148)
  last_move_positions.append(new_head)          # step-1 head recorded
  segments.insert(0, new_head)                  # grow at head
  if len(segments) > length: segments.pop()     # drop tail unless mid-growth

  # BOOST 2nd step, only if is_boosting AND length >= MIN_BOOST_LENGTH (l.156)
  if is_boosting and length >= MIN_BOOST_LENGTH:
     new_head2 = head + direction * s           # from the NEW head (l.157-160)
     last_move_positions.append(new_head2)       # step-2 head recorded
     segments.insert(0, new_head2)
     if len(segments) > length: segments.pop()

     boost_frames += 1                           # burn cadence (l.167)
     if boost_frames >= BOOST_LENGTH_COST_FRAMES: # =3
        boost_frames = 0
        if length > 1:
           length -= 1                            # pay 1 segment
           if len(segments) > length:
              burned_tail = segments.pop()
              if BOOST_DROPS_TRAIL_V2 and MECHANICS_VERSION==2:
                 pending_trail_pellets.append(burned_tail)  # v2 mass transfer
```

Key facts:
- **Head advance** is a pure cardinal step of exactly `s` per sub-step. No sub-cell
  motion, no float positions.
- **`last_move_positions`** holds the traversed head cell(s) this frame: 1 entry
  normal, 2 entries when boosting (both new_head cells). Collision + food checks
  scan *all* traversed heads (§3, §4.1), so a boost step can collide/eat at either
  intermediate cell.
- **Body-follow** is implicit: insert at head, pop tail when `len > length`. So the
  body always trails the head by cell-adjacency, and the tail vacates the oldest
  cell each sub-step (unless growing).
- **Growth timing**: `grow()` (l.179-186) only does `length += amount`. The extra
  segment is **not appended in `grow()`** — the body physically lengthens on the
  NEXT `move()` sub-step(s), because that sub-step's `len(segments) > length` check
  becomes False and the tail is NOT popped. I.e. eating on frame *t* → `length`
  incremented at step 7 of frame *t* → the snake keeps one extra segment starting
  from its next head-insert (frame *t*'s food check runs AFTER frame *t*'s move, so
  the retained tail first appears at frame *t+1*'s move). **Batched sim: apply
  `length += 1` at the food step; let the "pop tail iff len(segments) > length"
  rule at the next move do the physical growth.**
- **Boost burn cadence**: `boost_frames` increments **once per boosting frame**
  (not per sub-step). Every `BOOST_LENGTH_COST_FRAMES` (=3) boosting frames,
  `length -= 1` (never below 1) and the tail segment is popped immediately. `move`'s
  `boost_frames` is reset to 0 on that burn. `boost_frames` persists across frames
  and is reset to 0 on `respawn`/`soft_reset` (l.222, l.715) but **not** on a normal
  non-boosting frame — so it counts only frames where a burn-eligible boost occurred.
- **Boost eligibility gate**: the 2nd step and all burn logic require
  `length >= MIN_BOOST_LENGTH(=5)`. Even if `is_boosting=True`, a length-4 snake
  moves 1 step and pays nothing. Action decode also sets `is_boosting=False` when
  `length < MIN_BOOST_LENGTH` (ai_snake.py l.476-479), so the gate is enforced twice.
- **`change_direction`** (l.188-196): rejects a 180° reversal (keeps current dir).
  Note `AISnake.update` sets `self.direction` **directly** (l.472-473) via
  `relative_to_absolute_direction`, which by construction never yields a reversal
  (relative L/S/R from a cardinal dir), so the 180° guard is moot in the AI path.
  A replayed-action harness computing absolute dir from `(prev_dir, relative_action)`
  gets the same result — **use `relative_to_absolute_direction` (§6a), not raw dir**.

---

## 3. COLLISION detection & resolution

### 3.1 Detection order (`GameLogic.check_collisions`, game_logic.py l.130-167)

Iterate snakes by **list index `i`** (skip dead). For snake `i`:
1. **Wall** (`check_wall_collision`): if any traversed head is out of bounds →
   append `(snake, None, "wall")`, **`continue`** (wall wins, skip all else).
2. **Self** (`check_self_collision`): else if head hits own body → append
   `(snake, None, "self")`, **`continue`**.
3. **Other snakes**: else for each `j != i` with `other.is_alive`, in index order:
   - if `check_head_collision(i,j)` → append `(snake, other, "head")`
   - elif `check_body_collision(i,j)` → append `(snake, other, "body")`
   (Both head and body vs the same other can't both fire — `elif`. But snake `i`
   can produce multiple tuples across different `j`.)

Returned collisions are a **flat list in (i, then j) order**. This ordering is
load-bearing for v1 body-collision de-dup (§3.3).

Priority per snake is therefore **wall > self > (head|body) vs others**, and among
others, **head beats body** for a given pair, pairs scanned in `j` order.

**Predicates (all cell-exact via `same_cell`):**
- `_collision_positions(snake)` = `last_move_positions` or `[head]` (l.23-27) — the
  traversed head cell(s).
- **Wall** (l.169-196): for each traversed head, rectangular: out iff
  `x<0 or x>=game_width or y<0 or y>=game_height` (pixel compare, not cell).
  Circular: `dx²+dy² > radius²` (float — see §7).
- **Self** (l.227-251): only if `len(segments) > 3`; for each traversed head, check
  `same_cell(head, seg, s)` against `segments[3:]` (skip head + 2 adjacent).
- **Head-on** (l.253-274): True if any traversed-head cell of snake1 equals any
  traversed-head cell of snake2 (`same_cell`), **OR** `_head_paths_crossed`
  (the head-swap check, §3.2).
- **Body** (l.276-295): True if any traversed head of snake1 shares a cell with any
  `snake2.segments[1:]` (body = all but the other's head).

### 3.2 Head-swap path check (`_head_paths_crossed`, game_logic.py l.44-58)

Handles two heads that passed *through* each other in one frame without landing on
the same cell. For each snake builds `_head_path_positions` = `[previous_head] +
last_move_positions` where `previous_head = first_move_pos - direction*s`
(l.29-42). Then for every consecutive pair `(start1,end1)` in path1 and
`(start2,end2)` in path2, returns True iff
`distance(start1,end2) < threshold AND distance(start2,end1) < threshold`, with
`threshold = min(s1, s2)`. **This uses `GameLogic.distance` (Euclidean), not cell
index** — one of only two float paths in collision (see §7). On the shared lattice
the swap condition still reduces to exact cell coincidence of the swapped endpoints,
so the batched sim can implement it as: "head A's start-cell == head B's end-cell
AND head B's start-cell == head A's end-cell" over the consecutive path segments.

### 3.3 Resolution (`GameState.handle_collisions`, game_state.py l.589-694)

State: `dead_snake_ids: set`, `frame_collisions: dict`, `frame_kills: dict`,
`frame_death_causes: dict`. Helper `record_death(snake, type)` (l.608-621): if
already dead → return False; else add to `dead_snake_ids`, `episode_deaths += 1`,
bump `episode_collision_counts[type]`, tag `snake.last_death_cause`, record cause,
return True.

Iterate the `collisions` list **in the order produced by `check_collisions`**
(l.624). For each `(snake, other, type)`; if `snake.id in dead_snake_ids` → skip.

- **`"wall"` / `"self"`** (l.628-632): `_drop_food_from_snake(snake)` (§4.5),
  `snake.die()`, `record_death`, `frame_collisions[snake.id]=type`.

- **`"head"`** (l.633-664):
  - If `other is None` or `other.id in dead_snake_ids` → skip (l.634).
  - **v2 size resolution** (l.639-646): `length_a=snake._logical_length()`,
    `length_b=other._logical_length()` (`_logical_length = max(1, int(length))`,
    snake.py l.130-132). If `length_a >= 1.15 * length_b` → `winner=snake`; elif
    `length_b >= 1.15 * length_a` → `winner=other`; else `winner=None`. **v1: winner
    always None** (block gated on `MECHANICS_VERSION==2`).
  - If **winner** (l.647-656): `loser = other if winner is snake else snake`;
    drop loser's food, `loser.die()`, `record_death(loser,"head")`,
    `frame_collisions[loser.id]="head"`,
    `frame_kills[winner.id].append(loser.id)`. **Kill credit stays even if the
    winner later dies this frame** from another collision.
  - Else **mutual** (l.657-664): drop food for BOTH, `head_on_collision` (kills
    both, game_logic.py l.198-210), `record_death` both, mark both in
    `frame_collisions`. **No killer credited.**

- **`"body"`** (l.666-687): snake `i` hit `other`'s body.
  - If `other is None` → skip (l.676).
  - **v1 de-dup** (l.678): if `MECHANICS_VERSION != 2` AND `other.id in dead_snake_ids`
    → skip. (Legacy: the already-dead killer gets no credit; only the lower-index
    snake dies in a mutual body clash.)
  - **v2 does NOT skip** on a dead `other` (l.666-675 comment): in a mutual
    same-frame body collision both snakes hit each other's body; the second pair's
    victim death and killer credit must still land even though the killer died from
    the first pair.
  - Then: `_drop_food_from_snake(snake)`, `body_collision(snake, other)` (kills
    `snake`, game_logic.py l.212-225), `record_death(snake,"body")`,
    `frame_collisions[snake.id]="body"`, `frame_kills[other.id].append(snake.id)`
    (**`other` = the snake whose body was hit = the killer**).

Finally set `self.frame_collisions/frame_kills/frame_death_causes`; add
`sum(len(victims))` to `episode_kills` (l.691-693). Returns `frame_collisions`.

**Parity-critical resolution invariants:**
- A snake is resolved at most once (first collision tuple where it's the dying
  `snake`, guarded by `dead_snake_ids`).
- Because a wall/self `continue`s in detection, a snake that hit a wall NEVER also
  appears as a head/body victim → wall/self strictly dominates.
- v1 vs v2 body branch differs ONLY in the dead-killer skip (l.678). This changes
  both which snakes die and kill attribution in simultaneous mutual body clashes.
- `_drop_food_from_snake` is called **before** `die()` and asserts the snake is
  still alive (l.549-550: early return if already dead) — so drop order = death
  resolution order, and each snake drops exactly once.

---

## 4. FOOD (`FoodManager`, food_manager.py)

Two pools tracked in one `self.food: List[Tuple[int,int]]` plus a
`self._corpse_positions: Set` of positions that are corpse-class (v2 only).
`ambient_count = len(food) - corpse_count` where `corpse_count` counts live food
positions that are in `_corpse_positions` (l.285-299).

### 4.1 Consume (`consume_at`, l.213-232)
`eaten = [f for f in food if same_cell(f, position, radius)]`. If non-empty:
`food = [f for f in food if f not in eaten]`, drop those from `_corpse_positions`,
return True. **Cell-exact, radius = segment_size.** A single call removes **every**
pellet sharing the head's cell (normally ≤1 because `add_food` de-dups by cell).
`check_food_consumption` (game_state.py l.483-498) calls `consume_at` per traversed
head **and returns on the first True** — so a boost frame passing two occupied cells
eats only the first (step-1 head checked before step-2 head, matching
`last_move_positions` order), and `grow()` is called once.

### 4.2 add_food (l.131-155)
`if _position_overlaps_food(position): return False` (cell-exact overlap vs existing
food, l.121-129). Else append; if `corpse and CORPSE_EXEMPT_FROM_CAP_V2` add to
`_corpse_positions` **and then enforce the corpse cap, which may evict older corpse
pellets from `self.food` before this call returns — see §4.7.** **Two pellets never
share a cell.** Corpse flag passed True only under v2 (kill corpses + boost trail).
Returns True whenever the new pellet was appended, **even if the cap eviction then
removed a different (older) pellet** — the return value reports the add, not the net
board delta.

### 4.3 Spawn (`spawn`, l.173-191) & `_find_spawn_position` (l.157-171)
`spawn(count, snakes)`: loop `count` times; `pos = _find_spawn_position(snakes)`;
if `pos and add_food(pos)`: `spawned+=1`, else `break` (stop on first failure).
`_find_spawn_position`: up to **100 attempts**; if `snakes` given,
`pos = GameLogic.find_empty_position(...)` (which itself loops up to 100, §5) — if it
returns None, return None immediately; else `pos = _get_random_position()`. Accept
first `pos` that doesn't overlap existing food (cell-exact). **RNG per §5.**

### 4.4 Maintain (`maintain_count`, l.193-211)
`deficit = max_food - ambient_count`; if `deficit > 0`: `return spawn(deficit, snakes)`,
else 0. **Only ambient food is counted** — corpse-class food never suppresses baseline
spawns (v2). At v1 all food is ambient ⇒ this is the legacy `max_food - len(food)` top-up.

### 4.5 Corpse drop (`_drop_food_from_snake`, game_state.py l.538-567)
On any death (called from `handle_collisions`, before `die()`):
`mechanics_v2 = MECHANICS_VERSION==2`; `drop_fraction = 1.0 (v2) | 0.5 (v1)`;
`stride = max(1, round(1.0/drop_fraction))` → **v1 stride=2, v2 stride=1**.
For `i, segment in enumerate(snake.segments)`: if `i % stride == 0`: skip if segment
outside arena (`_segment_inside_arena`, l.569-587); else `add_food(segment, corpse=True)`
(v2) or `add_food(segment)` (v1). **So v1 drops every-other segment as ambient food;
v2 drops the whole corpse as corpse-class food.** Order = `segments` order (head→tail).
**No RNG** (positions are the snake's own cells). Overlaps with existing food are
silently dropped by `add_food`'s de-dup. At v2 each segment's add also runs the corpse
cap (§4.7), so a corpse large enough to cross the cap evicts the oldest corpse pellets
segment-by-segment as it drops — a long corpse can even evict its own earlier segments.

### 4.6 Boost trail pellet (v2)
`move()` records burned tail cells in `pending_trail_pellets`; `GameState.update`
step 6 (l.360-367) turns each in-arena cell into `add_food(cell, corpse=True)`.
Position = the exact vacated tail cell. No RNG.

### 4.7 Corpse cap (v2) — `corpse_food_cap` / `evict_oldest_corpse`
Corpse-class food is exempt from the **ambient** cap (§4.4) but is **not unbounded**.
Both engines share one helper pair in `mechanics_constants.py` so the eviction target —
and therefore the ordered food list — stays byte-identical; **the parity gate depends on
this. A reimplementation MUST use the same rule, not merely a similar one.**

**Cap formula** (`corpse_food_cap`, l.62-64):
```
cap = max(int(max_food * CORPSE_FOOD_CAP_MULTIPLE_V2), CORPSE_FOOD_CAP_FLOOR_V2)
    = max(int(max_food * 0.5), 10)
```
`int()` truncates toward zero (these are non-negative, so it's floor). The floor keeps
low-`max_food` configs (sparse arenas, small unit-test boards) able to hold a normal
corpse drop. Training board: `max_food=300` → **cap = 150**, total food bounded at
`(1 + 0.5) x max_food = 450`.

**Trigger point:** *inside* `add_food` (l.148-154), immediately after the new position
is appended to `self.food` and added to `_corpse_positions` — **not** at frame end and
**not** batched per corpse drop. A whole-corpse drop (§4.5) therefore evicts
incrementally, once per over-cap segment, as each segment is added:
```python
cap = corpse_food_cap(self.max_food)
while len(self._corpse_positions) > cap:
    if evict_oldest_corpse(self.food, self._corpse_positions) is None:
        break
```
The condition is `> cap` (strictly over), so the steady state holds **exactly `cap`**
corpse pellets. The `while` (not `if`) plus the `None` break makes it total: it drains
any pre-existing overage and cannot spin when no corpse pellet remains.

**Eviction order (FIFO by board order):** `evict_oldest_corpse` (l.67-90) scans the
**ordered `food` list front-to-back and removes the FIRST cell that is a member of
`corpse_positions`**, from both structures, returning it. This is "oldest rots first" —
oldest in **food-list append order**, which is *not* necessarily corpse-insertion order,
since ambient pellets are interleaved and `consume_at`/`trim_ambient` remove from the
middle. **The scan order over `self.food` (a list) is load-bearing; `_corpse_positions`
is only membership-tested, never iterated — do not reimplement this by iterating the
set** (§7.1 hazard 1).

**Only corpse-class pellets are ever evicted** — ambient food is untouched by this rule
and an ambient `add_food` never triggers it (the cap is checked only on the
`corpse and CORPSE_EXEMPT_FROM_CAP_V2` branch).

**Batched mirror:** `BatchSim._add_food` (batch_sim.py l.646-663) applies the identical
rule per env against `corpse_cells[e]` / `food_cells[e]` via the same helper. It
additionally maintains a `food_set[e]` membership index, so it must `discard` each
evicted cell from that index too (`food_set[e] == set(food_cells[e])` is an invariant).
A reimplementation carrying any such side index owes it the same discard.

---

## 5. RNG DRAW ORDER (CRITICAL for parity)

All draws use the **global `random` module** (`import random`). To reproduce food /
spawn positions the batched sim must reproduce the **exact sequence and count** of
`random.randint` / `random.uniform` calls, then `snap_to_cell` the result.

### 5.1 Rectangular ambient/food spawn — `FoodManager._get_random_position` (l.93-119)
Per call, **in this order**:
1. `random.randint(wall_thickness, game_width - wall_thickness - segment_size)`  → x
2. `random.randint(wall_thickness, game_height - wall_thickness - segment_size)`  → y

Then `snap_to_cell((x,y), s)`. **Two `randint` draws (x then y) per attempt.**
Called by `_find_spawn_position` when `snakes` is falsy — but in the live loop
`snakes` is always the (non-empty) snake list, so food spawn actually routes through
`find_empty_position` (5.2), NOT `_get_random_position`, EXCEPT `_spawn_initial`
during `reset()` may pass `snakes=None` (see 5.4).

### 5.2 Rectangular empty-position — `GameLogic.find_empty_position` (l.311-349)
Up to **100 attempts**; each attempt draws, **in order**:
1. `random.randint(margin, width - margin - 1)`   → x   (margin = wall_thickness)
2. `random.randint(margin, height - margin - 1)`  → y

Then `snap_to_cell`. If `position_overlaps_snakes` (float `distance < segment_size`
vs every living snake segment, l.351-359) → retry; else return. After 100 failures
prints a warning and returns None. **Two `randint` per attempt; attempts repeat until
a non-overlapping cell is found.** THE NUMBER OF DRAWS DEPENDS ON THE WORLD (rejection
sampling) — the batched sim must run the identical rejection loop with the identical
overlap predicate to stay in RNG lockstep.

Note `_find_spawn_position` wraps this with its OWN up-to-100 loop and an additional
`_position_overlaps_food` (cell-exact) rejection. So a food spawn is a nested
rejection: outer loop (≤100) calls `find_empty_position` (inner ≤100, snake-overlap)
and then re-checks food-overlap. **Draw budget per accepted food pellet is variable.**

### 5.3 Where draws happen per frame (`update`):
1. **Step 2** `maintain_count` → `spawn(deficit)` → per pellet: `_find_spawn_position`
   → `find_empty_position` rejection draws (5.2). deficit = `max_food - ambient_count`.
2. **Step 7 (not train_mode)** per eating snake: `spawn(1)` → same nested draws.
3. **Step 7 (train_mode, if anyone ate)** one `maintain_count` → `spawn(deficit)`.
4. **Step 4 (respawn, allow_respawn only)** per respawning snake:
   `find_empty_position` draws (5.2). (Skipped when `allow_respawn=False`.)

Within a frame the order is exactly: (respawns, in snake list order) →
(maintain_count spawns) → (per-eating-snake replacement spawns in snake order) OR
(one train-mode maintain_count). **Reproduce this exact call order.**

### 5.4 Episode reset draws (`GameState.reset` → `FoodManager.reset` → `_spawn_initial`)
`reset()` (game_state.py l.133-175): on soft-reset it repositions each existing snake
via `_get_non_overlapping_snake_position` (l.207-218): first `get_random_position()`
(game_state.py l.504-532: **2 `randint`, x then y**, then snap), and if it overlaps
already-placed snakes, `find_empty_position` (5.2). Snakes placed in list order.
Then `food_manager.reset(initial_food, self.snakes)` → `_spawn_initial(count, snakes)`
(l.79-91): clears food; loops `count` times, `_find_spawn_position(snakes)` (nested
rejection, 5.2). **On the very first construction** `FoodManager.__init__` calls
`_spawn_initial(initial_food)` with `snakes=None` → routes to `_get_random_position`
(5.1) — but `GameState` then calls `reset()` which re-spawns food WITH snakes, so the
init-time food is discarded. A parity harness should model the `reset()`-time food
(with snakes), not the constructor-time food.

### 5.5 Circular arena draws (`get_random_circular_position`, game_logic.py l.80-96)
Per call, **in order**: `random.uniform(0, 2π)` → angle; `random.uniform(0,1)` →
(sqrt'd) radius. Then snap. **See §7 — recommend rectangular-only for first parity.**

### 5.6 Snapping
Every rectangular draw is `snap_to_cell((x,y), s) = ((x//s)*s, (y//s)*s)`. The batched
sim draws the same two ints from a Python-`random`-compatible stream and applies the
same floor-snap. **Match Python's `random.Random` Mersenne-Twister exactly** (see §8):
`randint(a,b)` = `a + int(random()*(b-a+1))` semantics via `_randbelow`; simplest is to
call an actual `random.Random(seed)` from the harness rather than reimplement.

---

## 6. REWARD

### 6a. Action decode (before move; `AISnake.update`, ai_snake.py l.464-481)
Given integer `action ∈ [0,5]` (clamped l.465):
- `is_boost = action >= 3` (l.468); `direction_action = action % 3` (l.469).
- `new_direction = relative_to_absolute_direction(self.direction, direction_action)`
  (l.472) — CARDINAL=`[(0,-1),(1,0),(0,1),(-1,0)]`; left=idx-1, right=idx+1, straight
  same (game_logic.py l.98-128). Set `self.direction = new_direction` (l.473).
- `is_boosting = is_boost and length >= MIN_BOOST_LENGTH` else False (l.476-479).
- `self.move()`.
This is the entire deterministic action→world path a replayed-action harness needs.

### 6b. Reward v2 (`snake_reward.calculate_reward` l.46-91 → `_calculate_reward_v2`
l.178-246 → `reward_events.compute_reward_v2` l.76-122)
When `REWARD_VERSION == 2`, dispatch to the pure event function. Event summary
(`RewardEvents`, reward_events.py l.42-61) fields, built per snake in
`_calculate_reward_v2`:
- `prev_length` = `self._reward_prev_length` (baseline before this step; reset to 1 on
  respawn/soft_reset).
- `new_length` = `self.length` (post-step, post-growth).
- `died` = `collided` (= `snake.id in frame_collisions` this frame).
- `gamma` = `GameConfig.APEX_GAMMA` (= 0.99).
- `kills` = tuple of `victim._logical_length()` for each `victim_id in
  frame_kills.get(self.id, [])`, looked up in `other_snakes` (l.214-222).

`compute_reward_v2` (pure):
```
new_potential = 0.0 if died else phi(new_length)     # phi(L) = L / 10
potential = gamma * new_potential - phi(prev_length)
death     = -3.0 if died else 0.0
kill      = sum(0.3 * victim_length for victim_length in kills)   # UNCLAMPED
total     = potential + death + kill                  # accum order fixed
```
Constants (reward_events.py l.29-35): `PHI_LENGTH_DIVISOR=10.0`,
`KILL_REWARD_PER_VICTIM_LENGTH=0.3`, `DEATH_REWARD=-3.0`. **No clamping at v2.**
Breakdown keys `("potential","death","kill")` sum bit-exactly to total.

Counter upkeep (l.239-244): on death, `frames_since_food` and `_reward_prev_length`
are left as-is (respawn resets them); otherwise if `ate_food` set
`frames_since_food=0` else `+=1`, and `_reward_prev_length = self.length`. **A
dynamics-only harness that doesn't compute rewards can ignore `frames_since_food`;
`_reward_prev_length` is only a reward baseline (not a world quantity).**

### 6c. Reward v1 (context; `calculate_reward` l.93-176)
Legacy path (many shaping terms: death-with-length-scale, food base, food-distance
shaping from the state tensor, wall-awareness, per-action danger, starvation,
survival, kill via `_calculate_interaction_reward`, boost-segment penalty), clamped
to `[REWARD_MIN, REWARD_MAX]`. **v1 reward depends on the state tensor** (food-distance,
wall, danger indices) — reproducing it requires the full observation build, out of
scope for a pure-dynamics harness. **Recommend: parity target is mechanics-v2 +
reward-v2** (`configs/mechanics_v2.yaml`), whose reward is the pure event function and
needs no state tensor.

---

## 7. Determinism hazards & parity constraints

1. **dict/set iteration order.** `frame_kills`, `frame_collisions`,
   `_corpse_positions` are dicts/sets. `frame_kills`/`frame_collisions` are built by
   ordered appends and only *read by key* (never iterated for world effect), so order
   doesn't affect dynamics. `_corpse_positions` is a `set` but is only membership-
   tested (never iterated to produce positions), EXCEPT `trim_ambient` (l.234-264),
   `corpse_count` (l.285-294) and `evict_oldest_corpse` (§4.7), which all iterate
   `self.food` (a list, ordered) and test set membership — safe. **No load-bearing
   set/dict iteration in the hot path** — but note this is a *consequence* of those
   three scanning the list rather than the set. A reimplementation that iterates
   `_corpse_positions` directly to pick an eviction victim would be
   **nondeterministic and break parity**.

   The **complete inventory of `self.food` mutations** (the ordered list is the
   load-bearing structure — mirror it exactly):

   | Mutation | Site | Order effect |
   |---|---|---|
   | append | `add_food` (§4.2) | tail append, after cell-exact de-dup |
   | **remove oldest corpse** | **`evict_oldest_corpse` via `add_food` (§4.7)** | **removes the FIRST corpse-class cell in list order; v2 only, hot path** |
   | remove eaten | `consume_at` (§4.1) | removes every cell sharing the head's cell |
   | remove newest ambient | `trim_ambient` (l.234-264) | back-to-front, ambient only; external callers (web food-target control), not the core loop |
   | clear | `_spawn_initial` / `clear` | resets list + `_corpse_positions` |

   Append order is therefore **load-bearing in the core loop under v2**: it selects the
   corpse-cap eviction victim (§4.7). (Before the corpse cap landed, order affected only
   `trim_ambient` victims outside the core loop — that is no longer true.)

2. **Floating point in circular arena.** Wall check (`dx²+dy² > radius²`),
   `_segment_inside_arena`, `get_circular_arena` (scaled center/radius), and
   `get_random_circular_position` (`uniform`, `sqrt`, `cos`, `sin`) all use floats.
   Radius scaling multiplies by `min(width/base_width, height/base_height)`. These are
   FP-sensitive and hard to match bit-exactly across implementations.
   **Constraint: fix `arena_type = rectangular` for the first parity milestone.**
   Rectangular wall/spawn math is pure integer + `snap_to_cell` (floor div), which is
   trivially reproducible. Add circular parity only after rectangular is green.

3. **Two remaining float paths in rectangular collision** — both reduce to exact cell
   logic on the snapped lattice but literally call `GameLogic.distance`:
   (a) `_head_paths_crossed` head-swap (§3.2); (b) `position_overlaps_snakes` used in
   spawn rejection (§5.2). On the lattice, `distance < s` ⇔ same cell, so the batched
   sim can use cell equality — but it must apply it to the SAME candidate cells in the
   SAME rejection order to keep the RNG stream aligned.

4. **Snake id assignment order.** `create_snakes_for_game` (snake_factory.py l.324-356)
   assigns `id = i` for `i in range(num_snakes)` in list order; `actor_id` increments
   only for AI snakes (human at index 0 if `human_mode`). Colors =
   `SNAKE_COLORS[i % len]`. **List order == id order** (ids are 0..n-1 contiguous).
   Collision detection iterates by list index (= id). The batched sim should index
   snakes 0..n-1 in the same order; ids are just the index. Spawn positions are drawn
   in this same order at reset (§5.4), so id↔position pairing is RNG-order-dependent.

5. **`train_mode` branches** change RNG draw structure (§5.3) and death terminality
   (§1 step 4). The harness must set `train_mode` / `allow_respawn` to match the run
   being replayed.

6. **`_logical_length` vs body length.** v2 head-on/kill math uses
   `max(1, int(self.length))`, the *rule* length, which can exceed
   `len(segments)` for one frame right after eating (body hasn't filled in yet, §2).
   Use `self.length`, not `len(segments)`, for size comparisons and kill victim
   length. Kill reward victim length also uses `_logical_length` (§6b).

7. **max_length / clamping.** `grow()` has no cap; `length` grows unbounded in the
   sim itself. `MAX_LENGTH` only appears in reward-v1 death scaling and state
   normalization — not a movement/collision constraint. **No length cap in dynamics.**

---

## 8. Recommended parity harness contract

To make bit-exact parity achievable, impose:

1. **Rectangular arena only** (`arena_type=rectangular`) for milestone 1 (§7.2).
2. **Mechanics v2 + reward v2** as the parity target (`configs/mechanics_v2.yaml`):
   reward is the pure `compute_reward_v2`; no observation tensor needed.
3. **Seed `random` exactly once per env at reset**, then let the recorded run and the
   batched sim both draw from a Python-`random.Random(seed)`-compatible stream. The
   simplest correct approach: the batched sim, for spawn positions, calls a real
   `random.Random(seed)` and reproduces the draw ORDER in §5 (respawns → maintain →
   per-eat spawns), each draw being `randint(x)` then `randint(y)` with the same
   rejection loops (snake-overlap then food-overlap). Do NOT try to vectorize the RNG
   away — the rejection sampling makes draw counts data-dependent.
4. **Replay a recorded action log**: for each frame, per snake, the integer action
   0–5 (post-mask, as actually executed). This removes the neural net, epsilon, and
   action-mask nondeterminism from the parity surface. Feed the same action to §6a.
5. **Terminal deaths** (`allow_respawn=False`, i.e. `train_mode=True`) to avoid the
   respawn RNG branch, unless the recorded run used respawns (then reproduce §5.3.4).
6. **Match Python's Mersenne-Twister**: because rejection-sampling draw counts are
   world-dependent, the cleanest guarantee is to keep an actual CPython `random.Random`
   in the harness and consume it in the documented order, rather than reimplementing
   the PRNG in NumPy (whose stream differs).

**Validation ladder:** (a) single snake, no eating, straight moves — head/tail/wall;
(b) add boost (2-step, burn cadence, trail pellets); (c) add food (consume, grow
timing, spawn RNG); (d) two snakes (head-on size resolution, body kill attribution,
v1-vs-v2 body de-dup); (e) full 6-snake episode vs a recorded live-sim trace,
asserting per-frame equality of every snake's `segments`, `length`, `is_alive`,
`frame_kills`, `frame_collisions`, and the full `food` list (ordered).
