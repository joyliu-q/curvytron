#!/usr/bin/env python3
"""
Curvytron 1v1 Position Evaluator

Deterministic game-tree search for evaluating positions in Curvytron.
For every game state, enumerates all 3x3 action combinations (left/straight/right
for each player), recursively builds the complete game tree to a given depth,
and propagates death outcomes backward using minimax for the simultaneous-move game.

Physics replicated from the JS server (BaseAvatar, Game, World, AvatarBody):
  - Angle:    angle += direction * angularVelocityBase * step
  - Velocity: vx = cos(angle) * velocity/1000;  vy = sin(angle) * velocity/1000
  - Position: x += vx * step;  y += vy * step
  - Trail points emitted when dist(head, lastTrail) > radius
  - Self-trail immunity for most recent trailLatency (3) points
  - Wall death when head ± radius exceeds [0, boardSize]

Optimised for tree search via:
  - In-place state mutation with snapshot/restore (avoids O(T) clone per node)
  - Spatial hash grid for trail collision (O(1) amortised vs O(T) brute-force)
  - Maximin row-pruning (skips branches once a row can't beat current best)

Usage:
    python scripts/position_evaluator.py --depth 5
    python scripts/position_evaluator.py --depth 6 --mid-game --prune
    python scripts/position_evaluator.py --server http://localhost:8080 --session <id> --depth 6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum

# ---------------------------------------------------------------------------
# Constants — match src/shared/model/BaseAvatar.js, BaseGame.js, etc.
# ---------------------------------------------------------------------------

class Action(IntEnum):
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

ALL_ACTIONS: list[Action] = [Action.LEFT, Action.STRAIGHT, Action.RIGHT]
ACTION_NAMES: dict[Action, str] = {
    Action.LEFT: "left",
    Action.STRAIGHT: "straight",
    Action.RIGHT: "right",
}

VELOCITY = 16                              # BaseAvatar.prototype.velocity
VELOCITY_PER_MS = VELOCITY / 1000          # 0.016 units/ms
ANGULAR_VEL_BASE = 2.8 / 1000             # rad/ms  (BaseAvatar.prototype.angularVelocityBase)
RADIUS = 0.6                               # BaseAvatar.prototype.radius
TRAIL_LATENCY = 3                          # BaseAvatar.prototype.trailLatency
TICK_MS = 16                               # BaseGame.prototype.fixedStep default
ACTION_REPEAT = 4                          # default ticks per RL decision

MOVE_PER_TICK = VELOCITY_PER_MS * TICK_MS              # 0.256 units
ANGLE_PER_TICK = ANGULAR_VEL_BASE * TICK_MS            # 0.0448 rad ≈ 2.57°
COLLISION_DIST = RADIUS * 2                             # 1.2  (sum of radii)
COLLISION_DIST_SQ = COLLISION_DIST ** 2                 # 1.44

# Board for 2 players: sqrt(80^2 + 80^2/5) ≈ 87.6 → 88
DEFAULT_BOARD_SIZE = 88.0

# Spatial grid cell size — must be >= COLLISION_DIST so a 3×3 neighbourhood
# around the query cell covers all possible colliders.
GRID_CELL = 2.5

# ---------------------------------------------------------------------------
# Lightweight trail point (immutable, many instances)
# ---------------------------------------------------------------------------

TrailPoint = namedtuple("TrailPoint", ["x", "y", "owner", "index"])

# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    x: float
    y: float
    angle: float              # radians — 0=right, π/2=down
    alive: bool = True
    velocity: float = VELOCITY
    radius: float = RADIUS
    body_count: int = 0       # next trail index (= head's num in JS)
    last_trail_x: float = 0.0
    last_trail_y: float = 0.0
    printing: bool = True

    def __post_init__(self):
        if self.last_trail_x == 0.0 and self.last_trail_y == 0.0:
            self.last_trail_x = self.x
            self.last_trail_y = self.y

    def copy(self) -> PlayerState:
        return PlayerState(
            self.x, self.y, self.angle, self.alive, self.velocity, self.radius,
            self.body_count, self.last_trail_x, self.last_trail_y, self.printing,
        )

# ---------------------------------------------------------------------------
# Game state (clean, immutable-style — used for external API / demos)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    players: list[PlayerState]
    trails: list[TrailPoint]
    board_size: float
    tick: int = 0

    def clone(self) -> GameState:
        return GameState(
            [p.copy() for p in self.players],
            list(self.trails),
            self.board_size,
            self.tick,
        )

    @property
    def game_over(self) -> bool:
        return not self.players[0].alive or not self.players[1].alive

    def outcome(self, perspective: int) -> float | None:
        me = self.players[perspective]
        opp = self.players[1 - perspective]
        if me.alive and opp.alive:
            return None
        if not me.alive and not opp.alive:
            return 0.0
        return -1.0 if not me.alive else 1.0


# ---------------------------------------------------------------------------
# Clean simulation (for external use / testing)
# ---------------------------------------------------------------------------

def simulate_tick(state: GameState, actions: tuple[Action, Action]) -> GameState:
    """Advance by one tick. Returns a new GameState."""
    s = state.clone()
    for i in range(2):
        p = s.players[i]
        if not p.alive:
            continue
        ang_vel = float(actions[i]) * ANGULAR_VEL_BASE
        if ang_vel != 0.0:
            p.angle = (p.angle + ang_vel * TICK_MS) % (2.0 * math.pi)
        vel = p.velocity / 1000.0
        p.x += math.cos(p.angle) * vel * TICK_MS
        p.y += math.sin(p.angle) * vel * TICK_MS
        if p.printing:
            dx, dy = p.x - p.last_trail_x, p.y - p.last_trail_y
            if dx * dx + dy * dy > p.radius * p.radius:
                s.trails.append(TrailPoint(p.x, p.y, i, p.body_count))
                p.body_count += 1
                p.last_trail_x, p.last_trail_y = p.x, p.y
    s.tick += 1
    for i in range(2):
        p = s.players[i]
        if not p.alive:
            continue
        if (p.x - p.radius < 0 or p.x + p.radius > s.board_size
                or p.y - p.radius < 0 or p.y + p.radius > s.board_size):
            p.alive = False
            continue
        bc = p.body_count
        for t in s.trails:
            if t.owner == i and bc - t.index <= TRAIL_LATENCY:
                continue
            dx, dy = p.x - t.x, p.y - t.y
            if dx * dx + dy * dy < COLLISION_DIST_SQ:
                p.alive = False
                break
    return s


def simulate_step(
    state: GameState, actions: tuple[Action, Action], ticks: int = ACTION_REPEAT,
) -> GameState:
    s = state
    for _ in range(ticks):
        s = simulate_tick(s, actions)
        if s.game_over:
            break
    return s


# ---------------------------------------------------------------------------
# Fast in-place search engine (spatial grid + snapshot/restore)
# ---------------------------------------------------------------------------

def _save_player(p: PlayerState) -> tuple:
    return (p.x, p.y, p.angle, p.alive, p.velocity, p.radius,
            p.body_count, p.last_trail_x, p.last_trail_y, p.printing)


def _restore_player(p: PlayerState, s: tuple):
    (p.x, p.y, p.angle, p.alive, p.velocity, p.radius,
     p.body_count, p.last_trail_x, p.last_trail_y, p.printing) = s


class _Engine:
    """Mutable game state + spatial grid optimised for depth-first tree search."""

    __slots__ = (
        "p", "trails", "bs", "tick", "perspective", "ar",
        "grid", "nodes",
    )

    def __init__(self, state: GameState, perspective: int, action_repeat: int):
        self.p = [state.players[0].copy(), state.players[1].copy()]
        self.trails: list[TrailPoint] = list(state.trails)
        self.bs = state.board_size
        self.tick = state.tick
        self.perspective = perspective
        self.ar = action_repeat
        self.nodes = 0

        # Build spatial hash
        self.grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, t in enumerate(self.trails):
            self.grid[_gcell(t.x, t.y)].append(i)

    # -- snapshot / restore --

    def save(self) -> tuple:
        return (_save_player(self.p[0]), _save_player(self.p[1]),
                len(self.trails), self.tick)

    def restore(self, snap: tuple):
        _restore_player(self.p[0], snap[0])
        _restore_player(self.p[1], snap[1])
        tl = snap[2]
        self.tick = snap[3]
        # Remove grid entries for trails added since snapshot (reverse order)
        for i in range(len(self.trails) - 1, tl - 1, -1):
            t = self.trails[i]
            cell = self.grid.get(_gcell(t.x, t.y))
            if cell:
                cell.pop()
        del self.trails[tl:]

    # -- physics (in-place) --

    def step(self, actions: tuple[Action, Action]):
        for _ in range(self.ar):
            self._tick(actions)
            if not self.p[0].alive or not self.p[1].alive:
                break

    def _tick(self, actions: tuple[Action, Action]):
        bs = self.bs
        trails = self.trails
        grid = self.grid
        p0, p1 = self.p

        for i, (p, act) in enumerate(((p0, actions[0]), (p1, actions[1]))):
            if not p.alive:
                continue
            ang_vel = float(act) * ANGULAR_VEL_BASE
            if ang_vel != 0.0:
                p.angle = (p.angle + ang_vel * TICK_MS) % (2.0 * math.pi)
            vel = p.velocity / 1000.0
            p.x += math.cos(p.angle) * vel * TICK_MS
            p.y += math.sin(p.angle) * vel * TICK_MS
            if p.printing:
                dx = p.x - p.last_trail_x
                dy = p.y - p.last_trail_y
                if dx * dx + dy * dy > p.radius * p.radius:
                    tp = TrailPoint(p.x, p.y, i, p.body_count)
                    idx = len(trails)
                    trails.append(tp)
                    grid[_gcell(tp.x, tp.y)].append(idx)
                    p.body_count += 1
                    p.last_trail_x = p.x
                    p.last_trail_y = p.y

        self.tick += 1

        for i, p in enumerate((p0, p1)):
            if not p.alive:
                continue
            if (p.x - p.radius < 0 or p.x + p.radius > bs
                    or p.y - p.radius < 0 or p.y + p.radius > bs):
                p.alive = False
                continue
            if _grid_collision(grid, trails, p.x, p.y, i, p.body_count):
                p.alive = False

    # -- terminal / heuristic --

    def outcome(self) -> float | None:
        per = self.perspective
        me_alive = self.p[per].alive
        opp_alive = self.p[1 - per].alive
        if me_alive and opp_alive:
            return None
        if not me_alive and not opp_alive:
            return 0.0
        return -1.0 if not me_alive else 1.0

    def heuristic(self) -> float:
        per = self.perspective
        me = self.p[per]
        opp = self.p[1 - per]
        bs = self.bs
        s = 0.0

        # Wall proximity
        my_w = min(me.x, me.y, bs - me.x, bs - me.y)
        op_w = min(opp.x, opp.y, bs - opp.x, bs - opp.y)
        ref = bs * 0.25
        s += 0.10 * (_clamp01(my_w / ref) - _clamp01(op_w / ref))

        # Nearest dangerous trail
        my_n = self._nearest_trail(me, per)
        op_n = self._nearest_trail(opp, 1 - per)
        tref = 5.0
        s += 0.15 * (
            (_clamp01(my_n / tref) if my_n < 1e9 else 1.0)
            - (_clamp01(op_n / tref) if op_n < 1e9 else 1.0)
        )

        # Forward clearance
        my_f = self._forward_clear(me, per)
        op_f = self._forward_clear(opp, 1 - per)
        fref = 20.0
        s += 0.15 * (_clamp01(my_f / fref) - _clamp01(op_f / fref))

        # Safe immediate moves
        my_sm = self._safe_moves(me, per)
        op_sm = self._safe_moves(opp, 1 - per)
        s += 0.10 * (my_sm - op_sm) / 3.0

        return max(-0.5, min(0.5, s))

    def _nearest_trail(self, p: PlayerState, pidx: int) -> float:
        """Distance to closest dangerous trail point (using grid)."""
        best = 1e18
        px, py = p.x, p.y
        bc = p.body_count
        gx, gy = int(px / GRID_CELL), int(py / GRID_CELL)
        trails = self.trails
        grid = self.grid
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = grid.get((gx + dx, gy + dy))
                if not cell:
                    continue
                for ti in cell:
                    t = trails[ti]
                    if t.owner == pidx and bc - t.index <= TRAIL_LATENCY:
                        continue
                    ddx = px - t.x
                    ddy = py - t.y
                    d = math.sqrt(ddx * ddx + ddy * ddy) - COLLISION_DIST
                    if d < best:
                        best = d
        return best

    def _forward_clear(self, p: PlayerState, pidx: int, max_d: float = 30.0) -> float:
        step = 0.5
        vx = math.cos(p.angle) * step
        vy = math.sin(p.angle) * step
        px, py = p.x, p.y
        bc = p.body_count
        bs = self.bs
        grid = self.grid
        trails = self.trails
        d = 0.0
        while d < max_d:
            px += vx
            py += vy
            d += step
            if px - RADIUS < 0 or px + RADIUS > bs or py - RADIUS < 0 or py + RADIUS > bs:
                return d
            if _grid_collision(grid, trails, px, py, pidx, bc):
                return d
        return max_d

    def _safe_moves(self, p: PlayerState, pidx: int) -> int:
        safe = 0
        bs = self.bs
        ar = self.ar
        bc = p.body_count
        grid = self.grid
        trails = self.trails
        for act in ALL_ACTIONS:
            a = p.angle + float(act) * ANGLE_PER_TICK * ar
            px = p.x + math.cos(a) * MOVE_PER_TICK * ar
            py = p.y + math.sin(a) * MOVE_PER_TICK * ar
            if px - RADIUS < 0 or px + RADIUS > bs or py - RADIUS < 0 or py + RADIUS > bs:
                continue
            if not _grid_collision(grid, trails, px, py, pidx, bc):
                safe += 1
        return safe

    # -- tree search --

    def search(self, depth: int) -> dict:
        self.nodes += 1
        oc = self.outcome()
        if oc is not None:
            lab = "win" if oc > 0 else ("loss" if oc < 0 else "draw")
            return {"score": oc, "terminal": True, "label": lab}
        if depth == 0:
            return {"score": self.heuristic(), "terminal": True, "label": "ongoing"}

        snap = self.save()
        per = self.perspective
        payoff = [[0.0] * 3 for _ in range(3)]
        counts: dict[str, int] = {"win": 0, "loss": 0, "draw": 0, "ongoing": 0}

        for i, my_a in enumerate(ALL_ACTIONS):
            for j, op_a in enumerate(ALL_ACTIONS):
                actions = (my_a, op_a) if per == 0 else (op_a, my_a)
                self.step(actions)
                sub = self.search(depth - 1)
                payoff[i][j] = sub["score"]
                _tally(counts, sub)
                self.restore(snap)

        return _minimax(payoff, counts)

    def search_pruned(self, depth: int) -> dict:
        self.nodes += 1
        oc = self.outcome()
        if oc is not None:
            lab = "win" if oc > 0 else ("loss" if oc < 0 else "draw")
            return {"score": oc, "terminal": True, "label": lab}
        if depth == 0:
            return {"score": self.heuristic(), "terminal": True, "label": "ongoing"}

        snap = self.save()
        per = self.perspective
        payoff = [[0.0] * 3 for _ in range(3)]
        counts: dict[str, int] = {"win": 0, "loss": 0, "draw": 0, "ongoing": 0}
        best_row_min = -2.0

        for i, my_a in enumerate(ALL_ACTIONS):
            row_min = 2.0
            pruned = False
            for j, op_a in enumerate(ALL_ACTIONS):
                actions = (my_a, op_a) if per == 0 else (op_a, my_a)
                self.step(actions)
                sub = self.search_pruned(depth - 1)
                payoff[i][j] = sub["score"]
                _tally(counts, sub)
                self.restore(snap)

                row_min = min(row_min, sub["score"])
                if row_min <= best_row_min and j < 2:
                    for jj in range(j + 1, 3):
                        payoff[i][jj] = row_min
                    pruned = True
                    break

            if not pruned and row_min > best_row_min:
                best_row_min = row_min
            elif pruned and row_min > best_row_min:
                best_row_min = row_min

        return _minimax(payoff, counts)


# -- free functions used by the engine --

def _gcell(x: float, y: float) -> tuple[int, int]:
    return (int(x / GRID_CELL), int(y / GRID_CELL))


def _grid_collision(
    grid: dict, trails: list, px: float, py: float, owner: int, body_count: int,
) -> bool:
    gx = int(px / GRID_CELL)
    gy = int(py / GRID_CELL)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cell = grid.get((gx + dx, gy + dy))
            if not cell:
                continue
            for ti in cell:
                t = trails[ti]
                if t.owner == owner and body_count - t.index <= TRAIL_LATENCY:
                    continue
                ddx = px - t.x
                ddy = py - t.y
                if ddx * ddx + ddy * ddy < COLLISION_DIST_SQ:
                    return True
    return False


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _tally(counts: dict, sub: dict):
    if sub.get("terminal"):
        lab = sub.get("label", "ongoing")
        counts[lab] = counts.get(lab, 0) + 1
    else:
        for k, v in sub.get("counts", {}).items():
            counts[k] = counts.get(k, 0) + v


def _minimax(payoff: list[list[float]], counts: dict) -> dict:
    row_mins = [min(payoff[i]) for i in range(3)]
    best_i = max(range(3), key=lambda i: row_mins[i])
    return {
        "score": row_mins[best_i],
        "best_action": ALL_ACTIONS[best_i],
        "payoff": payoff,
        "action_stats": {
            ALL_ACTIONS[i]: {
                "min": min(payoff[i]),
                "max": max(payoff[i]),
                "avg": sum(payoff[i]) / 3.0,
            }
            for i in range(3)
        },
        "counts": counts,
        "terminal": False,
    }


# ---------------------------------------------------------------------------
# Public evaluator API
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    score: float
    best_action: Action
    payoff_matrix: list[list[float]]
    action_stats: dict
    outcome_counts: dict
    nodes_explored: int
    depth: int


class PositionEvaluator:
    """Evaluate a 1v1 Curvytron position via complete game-tree search."""

    def __init__(self, perspective: int = 0, action_repeat: int = ACTION_REPEAT):
        self.perspective = perspective
        self.action_repeat = action_repeat

    def evaluate(self, state: GameState, depth: int, prune: bool = False) -> EvalResult:
        eng = _Engine(state, self.perspective, self.action_repeat)
        tree = eng.search_pruned(depth) if prune else eng.search(depth)
        return self._pack(tree, depth, eng.nodes)

    def all_outcomes(self, state: GameState, depth: int) -> list[dict]:
        """Enumerate every leaf (action-path → outcome). Small depths only!"""
        results: list[dict] = []
        self._enum(state, depth, [], results)
        return results

    # -- internals --

    def _enum(self, state, depth, path, results):
        oc = state.outcome(self.perspective)
        if oc is not None:
            lab = "win" if oc > 0 else ("loss" if oc < 0 else "draw")
            results.append({"path": list(path), "outcome": lab, "score": oc})
            return
        if depth == 0:
            eng = _Engine(state, self.perspective, self.action_repeat)
            h = eng.heuristic()
            results.append({"path": list(path), "outcome": "ongoing", "score": h})
            return
        for my_a in ALL_ACTIONS:
            for op_a in ALL_ACTIONS:
                acts = (my_a, op_a) if self.perspective == 0 else (op_a, my_a)
                ns = simulate_step(state, acts, self.action_repeat)
                path.append((my_a, op_a))
                self._enum(ns, depth - 1, path, results)
                path.pop()

    @staticmethod
    def _pack(tree: dict, depth: int, nodes: int) -> EvalResult:
        if tree.get("terminal"):
            s = tree["score"]
            return EvalResult(
                score=s, best_action=Action.STRAIGHT,
                payoff_matrix=[[s] * 3] * 3,
                action_stats={a: {"min": s, "max": s, "avg": s} for a in ALL_ACTIONS},
                outcome_counts={tree.get("label", "ongoing"): 1},
                nodes_explored=nodes, depth=depth,
            )
        return EvalResult(
            score=tree["score"],
            best_action=tree["best_action"],
            payoff_matrix=tree["payoff"],
            action_stats=tree["action_stats"],
            outcome_counts=tree["counts"],
            nodes_explored=nodes,
            depth=depth,
        )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_result(r: EvalResult) -> str:
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("  CURVYTRON 1v1 POSITION EVALUATION")
    lines.append("=" * 62)
    lines.append(f"  Minimax score : {r.score:+.4f}")
    lines.append(f"  Best action   : {ACTION_NAMES[r.best_action]}")
    lines.append(f"  Search depth  : {r.depth}")
    lines.append(f"  Nodes explored: {r.nodes_explored:,}")

    names = ["left", "straight", "right"]
    lines.append("")
    lines.append("  Payoff matrix (rows = us, cols = opponent):")
    hdr = "              " + "  ".join(f"{'opp_' + n:>12}" for n in names)
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for i, n in enumerate(names):
        row = f"  {n:>10}  " + "  ".join(
            f"{r.payoff_matrix[i][j]:>12.4f}" for j in range(3)
        )
        lines.append(row)

    lines.append("")
    lines.append("  Per-action analysis:")
    for act in ALL_ACTIONS:
        s = r.action_stats[act]
        lines.append(
            f"    {ACTION_NAMES[act]:>8}:  "
            f"min={s['min']:+.4f}  max={s['max']:+.4f}  avg={s['avg']:+.4f}"
        )

    total = sum(r.outcome_counts.values())
    if total > 0:
        lines.append("")
        lines.append(f"  Outcome distribution ({total:,} leaves):")
        for k in ("win", "loss", "draw", "ongoing"):
            c = r.outcome_counts.get(k, 0)
            pct = 100.0 * c / total
            bar = "#" * int(pct / 2.5)
            lines.append(f"    {k:>7}: {c:>10,} ({pct:5.1f}%)  {bar}")

    lines.append("=" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API state import
# ---------------------------------------------------------------------------

def state_from_api(api: dict) -> GameState:
    bs = api["board"]["size"]
    players: list[PlayerState] = []
    for p in api["players"][:2]:
        players.append(PlayerState(
            x=p["x"], y=p["y"], angle=p["angle"],
            alive=p["alive"],
            velocity=p.get("velocity", VELOCITY),
            radius=p.get("radius", RADIUS),
            printing=p.get("printing", True),
        ))
    # Reconstruct trails from occupancy grid (owner unknown → always dangerous)
    trails: list[TrailPoint] = []
    occ = api.get("occupancy")
    if occ:
        cells, w, h = occ["cells"], occ["width"], occ["height"]
        cw, ch = bs / w, bs / h
        for row in range(h):
            for col in range(w):
                if cells[row][col] == 1:
                    trails.append(TrailPoint((col + 0.5) * cw, (row + 0.5) * ch, -1, 0))
    return GameState(players, trails, bs, api.get("tick", 0))


def fetch_state(server: str, session_id: str, token: str = "") -> dict:
    import requests as req
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = req.get(f"{server}/api/rl/sessions/{session_id}/state", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Demo states
# ---------------------------------------------------------------------------

def demo_opening(bs: float = DEFAULT_BOARD_SIZE) -> GameState:
    m = bs * 0.2
    return GameState(
        [PlayerState(m, bs / 2, 0.0), PlayerState(bs - m, bs / 2, math.pi)],
        [], bs,
    )


def demo_mid_game(bs: float = DEFAULT_BOARD_SIZE) -> GameState:
    p0 = PlayerState(50.0, 44.0, 0.3)
    p1 = PlayerState(38.0, 44.0, math.pi + 0.3)
    trails: list[TrailPoint] = []
    for i in range(80):
        t = i * 0.4
        trails.append(TrailPoint(10.0 + t * 0.7, 44.0 + 8 * math.sin(t * 0.08), 0, i))
    p0.body_count = 80
    p0.last_trail_x, p0.last_trail_y = trails[79].x, trails[79].y
    for i in range(80):
        t = i * 0.4
        trails.append(TrailPoint(78.0 - t * 0.7, 44.0 - 8 * math.sin(t * 0.08), 1, i))
    p1.body_count = 80
    p1.last_trail_x, p1.last_trail_y = trails[-1].x, trails[-1].y
    return GameState([p0, p1], trails, bs)


def demo_near_wall(bs: float = DEFAULT_BOARD_SIZE) -> GameState:
    return GameState(
        [PlayerState(bs - 3.0, bs / 2, 0.0), PlayerState(bs / 2, bs / 2, math.pi)],
        [], bs,
    )


def demo_corridor(bs: float = DEFAULT_BOARD_SIZE) -> GameState:
    """Player 0 in a narrow corridor between two trail walls."""
    p0 = PlayerState(44.0, 44.0, 0.0)
    p1 = PlayerState(20.0, 20.0, 0.0)
    trails: list[TrailPoint] = []
    # Upper wall
    for i in range(120):
        trails.append(TrailPoint(5.0 + i * 0.7, 41.0, -1, 0))
    # Lower wall
    for i in range(120):
        trails.append(TrailPoint(5.0 + i * 0.7, 47.0, -1, 0))
    return GameState([p0, p1], trails, bs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Curvytron 1v1 Position Evaluator")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--perspective", type=int, default=0, choices=[0, 1])
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--action-repeat", type=int, default=ACTION_REPEAT)
    ap.add_argument("--board-size", type=float, default=DEFAULT_BOARD_SIZE)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="opening (default)")
    g.add_argument("--mid-game", action="store_true")
    g.add_argument("--near-wall", action="store_true")
    g.add_argument("--corridor", action="store_true")
    g.add_argument("--server", type=str, help="RL API URL")

    ap.add_argument("--session", type=str)
    ap.add_argument("--token", type=str, default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--enumerate", action="store_true",
                    help="list every leaf (small depths only)")

    args = ap.parse_args()

    if args.server and args.session:
        api = fetch_state(args.server, args.session, args.token)
        state = state_from_api(api)
        print(f"Loaded session {args.session}  tick={state.tick}", file=sys.stderr)
    elif args.mid_game:
        state = demo_mid_game(args.board_size)
        print("Demo: mid-game", file=sys.stderr)
    elif args.near_wall:
        state = demo_near_wall(args.board_size)
        print("Demo: near wall", file=sys.stderr)
    elif args.corridor:
        state = demo_corridor(args.board_size)
        print("Demo: corridor", file=sys.stderr)
    else:
        state = demo_opening(args.board_size)
        print("Demo: opening", file=sys.stderr)

    print(
        f"Board {state.board_size:.0f}x{state.board_size:.0f}  "
        f"trails={len(state.trails)}  tick={state.tick}",
        file=sys.stderr,
    )
    for i, pl in enumerate(state.players):
        print(
            f"  P{i}: ({pl.x:.1f},{pl.y:.1f}) "
            f"{math.degrees(pl.angle):.1f}° "
            f"alive={pl.alive} bodies={pl.body_count}",
            file=sys.stderr,
        )
    ticks_ahead = args.depth * args.action_repeat
    print(
        f"depth={args.depth}  (~{ticks_ahead} ticks / {ticks_ahead * TICK_MS}ms)  "
        f"max_nodes={9**args.depth:,}"
        f"{'  (pruning ON)' if args.prune else ''}",
        file=sys.stderr,
    )

    ev = PositionEvaluator(args.perspective, args.action_repeat)
    t0 = time.time()
    result = ev.evaluate(state, args.depth, prune=args.prune)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s ({result.nodes_explored:,} nodes)\n", file=sys.stderr)

    if args.json:
        print(json.dumps({
            "score": result.score,
            "best_action": ACTION_NAMES[result.best_action],
            "payoff_matrix": result.payoff_matrix,
            "action_stats": {
                ACTION_NAMES[a]: s for a, s in result.action_stats.items()
            },
            "outcome_counts": result.outcome_counts,
            "nodes_explored": result.nodes_explored,
            "depth": result.depth,
            "elapsed_s": round(elapsed, 3),
        }, indent=2))
    else:
        print(format_result(result))

    if args.enumerate:
        if args.depth > 3:
            print(
                f"\nWARN: --enumerate at depth {args.depth} → {9**args.depth:,} leaves",
                file=sys.stderr,
            )
        outcomes = ev.all_outcomes(state, args.depth)
        wins = sum(1 for o in outcomes if o["outcome"] == "win")
        losses = sum(1 for o in outcomes if o["outcome"] == "loss")
        draws = sum(1 for o in outcomes if o["outcome"] == "draw")
        ongoing = len(outcomes) - wins - losses - draws
        print(f"\nFull enumeration: {len(outcomes)} leaves")
        print(f"  Wins: {wins}  Losses: {losses}  Draws: {draws}  Ongoing: {ongoing}")
        terminal = [o for o in outcomes if o["outcome"] in ("win", "loss", "draw")]
        if terminal:
            print(f"\n  Sample terminal paths (first 10):")
            for o in terminal[:10]:
                path_str = " -> ".join(
                    f"({ACTION_NAMES[a]},{ACTION_NAMES[b]})" for a, b in o["path"]
                )
                print(f"    {o['outcome']:>5} [{o['score']:+.1f}]  {path_str}")


if __name__ == "__main__":
    main()
