"""Self-play agent system for curvytron multi-agent GRPO training.

Two agents (both the training model) play against each other in a full game
episode. Each per-turn generation is recorded as a SLIME Sample with logprobs.
Reward per step: escalating survival bonus + reachable-space fraction.
"""

import asyncio
import collections
import itertools
import json
import os
import uuid
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

from slime.utils.http_utils import post
from slime.utils.types import Sample

from .game_client import AsyncGameClient, MAX_STEPS
from .prompts import SYSTEM_PROMPT

ACTIONS = ["left", "straight", "right"]

# ── Game log for debugging ─────────────────────────────────────────────────

# Rolling buffer of recent game summaries, printed periodically
_game_log: list[dict] = []
_game_count = 0
_LOG_EVERY = 8  # print a digest every N games

# ── Trace recording (for replay viewer) ────────────────────────────────────
# Every Nth game is saved step-by-step to /data/game_traces/*.json on the
# curvytron-data Modal volume, to be rendered by slime/game_viewer.py.
TRACE_EVERY = 10
TRACE_DIR = os.environ.get("CURVYTRON_TRACE_DIR", "/data/game_traces")
_trace_counter = itertools.count()
_trace_volume = None


def _log_game(summary: dict):
    """Record a game summary and periodically print a digest."""
    global _game_count
    _game_log.append(summary)
    if len(_game_log) > 50:
        _game_log.pop(0)
    _game_count += 1

    if _game_count % _LOG_EVERY == 0:
        _print_digest()


def _print_digest():
    """Print a digest of recent games to stdout (shows up in training logs)."""
    recent = _game_log[-_LOG_EVERY:]
    if not recent:
        return

    avg_steps = sum(g["steps"] for g in recent) / len(recent)
    outcomes = collections.Counter(g["outcome"] for g in recent)
    action_counts = collections.Counter()
    for g in recent:
        action_counts.update(g["actions_a"])
        action_counts.update(g["actions_b"])
    total_actions = sum(action_counts.values()) or 1

    print(
        f"\n[curvytron-log] ──── Game digest (last {len(recent)} games, total {_game_count}) ────"
    )
    print(f"  Avg steps: {avg_steps:.1f}")
    print(f"  Outcomes: {dict(outcomes)}")
    print(
        "  Action distribution: "
        + ", ".join(f"{a}={action_counts[a] / total_actions:.0%}" for a in ACTIONS)
    )

    # Print the last game's final board as a sample
    last = recent[-1]
    print(
        f"  Last game (seed={last['seed']}, {last['steps']} steps, {last['outcome']}):"
    )
    if last.get("final_board"):
        # Indent and truncate board for readability
        board_lines = last["final_board"].split("\n")
        for line in board_lines[:15]:
            print(f"    {line}")
        if len(board_lines) > 15:
            print(f"    ... ({len(board_lines) - 15} more rows)")
    print(f"  A actions: {last['actions_a']}")
    print(f"  B actions: {last['actions_b']}")
    print("[curvytron-log] ─────────────────────────────────────────────────────\n")


def _commit_trace_volume():
    """Best-effort commit of the curvytron-data volume so the viewer can read."""
    global _trace_volume
    try:
        if _trace_volume is None:
            import modal

            _trace_volume = modal.Volume.from_name(
                "curvytron-data", create_if_missing=True
            )
        _trace_volume.commit()
    except Exception as e:
        print(f"[curvytron-trace] commit failed: {e}")


def _snapshot_players(state: dict) -> list[dict]:
    out = []
    for p in state.get("players", []):
        out.append(
            {
                "player_id": p.get("player_id"),
                "marker": p.get("marker"),
                "name": p.get("name"),
                "x": p.get("x"),
                "y": p.get("y"),
                "angle": p.get("angle"),
                "alive": p.get("alive"),
            }
        )
    return out


def _make_step_record(
    step_idx: int,
    state: dict,
    action_a,
    action_b,
    reward_a,
    reward_b,
) -> dict:
    occ = state.get("occupancy", {}) or {}
    return {
        "step": step_idx,
        "tick": state.get("tick"),
        "board": occ.get("ascii"),
        "players": _snapshot_players(state),
        "action_a": action_a,
        "action_b": action_b,
        "reward_a": reward_a,
        "reward_b": reward_b,
    }


def _save_trace(
    seed: str,
    outcome: str,
    total_steps: int,
    trace_steps: list[dict],
    marker_a: str,
    marker_b: str,
):
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_seed = "".join(c if c.isalnum() or c in "-_" else "_" for c in seed)
        fname = f"{ts}_{safe_seed}_{uuid.uuid4().hex[:6]}.json"
        path = os.path.join(TRACE_DIR, fname)
        payload = {
            "seed": seed,
            "recorded_at": ts,
            "outcome": outcome,
            "total_steps": total_steps,
            "markers": {"a": marker_a, "b": marker_b},
            "steps": trace_steps,
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        print(
            f"[curvytron-trace] saved {fname} ({total_steps} steps, outcome={outcome})"
        )
        _commit_trace_volume()
    except Exception as e:
        print(f"[curvytron-trace] save failed for seed={seed}: {e}")


# Constrained decoding regex — forces SGLang to output exactly one valid action
ACTION_REGEX = r"(left|straight|right)"
MAX_ACTION_TOKENS = 5

# Per-step rewards
DEAD_REWARD = 0.0
INVALID_REWARD = -10.0
SPACE_WEIGHT = 0.5  # weight for reachable-space bonus component


# ── Prompt building ─────────────────────────────────────────────────────────


def build_turn_prompt(state: dict, my_player_id: str, my_marker: str) -> str:
    """Build the user-turn message describing the current board state."""
    tick = state["tick"]
    occ = state["occupancy"]
    ascii_board = occ["ascii"]
    board_w = occ["width"]
    board_h = occ["height"]

    me, opponents = None, []
    for p in state["players"]:
        if p["player_id"] == my_player_id:
            me = p
        else:
            opponents.append(p)

    board_info = state.get("board", {})
    borderless = board_info.get("borderless", False)
    border_note = " (BORDERLESS — edges wrap around!)" if borderless else ""
    lines = [f"## Tick {tick}  |  Board {board_w}x{board_h}{border_note}", ""]

    if me:
        alive_str = "ALIVE" if me["alive"] else "DEAD"
        lines.append(
            f"**You** are player '{my_marker}' | "
            f"Position: ({me['x']:.1f}, {me['y']:.1f}) | "
            f"Angle: {me['angle']:.2f} rad | Status: {alive_str}"
        )

    lines.append("")
    for op in opponents:
        alive_str = "ALIVE" if op["alive"] else "DEAD"
        lines.append(
            f"Opponent '{op['marker']}' ({op['name']}) | "
            f"Position: ({op['x']:.1f}, {op['y']:.1f}) | "
            f"Angle: {op['angle']:.2f} rad | Status: {alive_str}"
        )

    lines += [
        "",
        "**Board:**",
        "```",
        ascii_board,
        "```",
        "",
        "Choose your action: left, straight, or right?",
    ]
    return "\n".join(lines)


def format_chat_prompt(
    tokenizer,
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> str:
    """Apply the chat template to produce a full prompt string.

    If history is provided, prior user/assistant turns are included so the
    model can learn from the trajectory of the game.  History is a list of
    {"role": "user"|"assistant", "content": ...} dicts.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# Maximum number of past turns to keep per player (user+assistant = 1 turn).
# Each board state is ~780 tokens; with context_len=4096 we can fit ~4 turns.
MAX_HISTORY_TURNS = 3


# ── Action parsing ──────────────────────────────────────────────────────────


def parse_action(reply):
    """Parse action from model output. Returns None if no valid action found."""
    if not reply:
        return None
    import re

    clean = reply
    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    # Strip stop tokens like <|im_end|>
    clean = clean.split("<|")[0].strip().lower()
    if clean in ACTIONS:
        return clean
    # High-temperature outputs may append garbage after the action word
    for action in ACTIONS:
        if clean.startswith(action):
            return action
    return None


# ── SGLang generation ───────────────────────────────────────────────────────


async def generate_response(args, prompt: str, key: str):
    """Generate a single action response via SGLang with constrained decoding."""
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        sample.prompt = prompt
        prompt_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)

        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            MAX_ACTION_TOKENS,
            max_context_length - prompt_length,
        )

        if current_sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": current_sampling_params,
            "return_logprob": True,
            "regex": ACTION_REGEX,
        }

        output = await post(url, payload)

        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
        else:
            new_response_tokens = []

        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        args.results_dict[key].append(sample)
        return output["text"]

    except Exception as e:
        print(f"[curvytron-ma] generate_response error: {e}")
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────


def is_player_alive(state: dict, player_id: str) -> bool:
    for p in state.get("players", []):
        if p.get("player_id") == player_id:
            return p.get("alive", False)
    return False


def _get_player_pos(state: dict, player_id: str) -> tuple[float, float] | None:
    """Return (x, y) world position of a player, or None if not found."""
    for p in state.get("players", []):
        if p.get("player_id") == player_id:
            x, y = p.get("x"), p.get("y")
            if x is not None and y is not None:
                return (x, y)
    return None


def reachable_fraction(state: dict, player_id: str) -> float:
    """BFS flood fill from a player's position on the occupancy grid.

    Returns the fraction of empty cells reachable from the player's current
    position (0.0 = completely trapped, 1.0 = entire board reachable).
    Runs in O(grid_size^2) which is ~7700 cells for an 88x88 board.
    """
    pos = _get_player_pos(state, player_id)
    if pos is None:
        return 0.0

    occ = state.get("occupancy", {})
    cells = occ.get("cells")
    if not cells:
        return 0.0

    height = len(cells)
    width = len(cells[0]) if height > 0 else 0
    if width == 0:
        return 0.0

    # Map world coords to grid coords
    board_size = state.get("board", {}).get("size")
    if board_size and board_size > 0:
        gx = int(pos[0] / board_size * width)
        gy = int(pos[1] / board_size * height)
    else:
        gx = int(pos[0] / width * width)
        gy = int(pos[1] / height * height)
    gx = max(0, min(width - 1, gx))
    gy = max(0, min(height - 1, gy))

    # Count total empty cells for normalization
    total_empty = sum(1 for row in cells for c in row if c == 0)
    if total_empty == 0:
        return 0.0

    # BFS from player position
    visited = set()
    queue = deque()
    queue.append((gy, gx))
    visited.add((gy, gx))
    reachable = 0

    while queue:
        cy, cx = queue.popleft()
        if cells[cy][cx] == 0:
            reachable += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in visited:
                    if cells[ny][nx] == 0:
                        visited.add((ny, nx))
                        queue.append((ny, nx))

    return reachable / total_empty


def compute_step_reward(step: int, state: dict, player_id: str) -> float:
    """Compute per-step reward: escalating survival base + reachable space bonus.

    - Escalating base: step / MAX_STEPS (0 → 1 over the game)
    - Space bonus: SPACE_WEIGHT * fraction of empty cells reachable via BFS
    """
    base = step / MAX_STEPS
    space = reachable_fraction(state, player_id)
    return base + SPACE_WEIGHT * space


# ── Main self-play loop ─────────────────────────────────────────────────────


async def run_selfplay_game(args, sample: Sample):
    """Play a full self-play game and return scored Samples for both players.

    Two agents (both the training model) play against each other.
    Each action generates a Sample; reward = escalating survival base +
    reachable-space bonus if alive, 0.0 if dead.
    """
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"player_a": [], "player_b": []}

    game_seed = sample.prompt
    tokenizer = args.tokenizer

    client = AsyncGameClient()
    session_id = None

    # Decide whether to record this game step-by-step for the replay viewer.
    record = next(_trace_counter) % TRACE_EVERY == 0
    trace_steps: list[dict] = []

    try:
        # ── Setup game ──────────────────────────────────────────────────
        session = await client.create_session(game_seed)
        session_id = session["session_id"]

        actor_a = await client.add_bot(session_id, "Agent-A", "#ff4444")
        actor_b = await client.add_bot(session_id, "Agent-B", "#4444ff")
        state = await client.start_game(session_id)

        marker_a, marker_b = "?", "?"
        for p in state.get("players", []):
            if p.get("player_id") == actor_a["player_id"]:
                marker_a = p.get("marker", "?")
            elif p.get("player_id") == actor_b["player_id"]:
                marker_b = p.get("marker", "?")

        if record:
            trace_steps.append(
                _make_step_record(0, state, None, None, None, None)
            )

        # ── Game loop ───────────────────────────────────────────────────
        step = 0
        a_alive, b_alive = True, True
        actions_a, actions_b = [], []
        # Per-player conversation history for stateful prompts
        history_a: list[dict] = []
        history_b: list[dict] = []

        while step < MAX_STEPS and not state.get("done", False):
            # Generate actions for living players
            gen_tasks = []
            gen_keys = []
            if a_alive:
                turn_text_a = build_turn_prompt(state, actor_a["player_id"], marker_a)
                prompt_a = format_chat_prompt(
                    tokenizer, SYSTEM_PROMPT, turn_text_a, history=history_a
                )
                gen_tasks.append(generate_response(args, prompt_a, key="player_a"))
                gen_keys.append("a")
            if b_alive:
                turn_text_b = build_turn_prompt(state, actor_b["player_id"], marker_b)
                prompt_b = format_chat_prompt(
                    tokenizer, SYSTEM_PROMPT, turn_text_b, history=history_b
                )
                gen_tasks.append(generate_response(args, prompt_b, key="player_b"))
                gen_keys.append("b")

            responses = await asyncio.gather(*gen_tasks)
            response_map = dict(zip(gen_keys, responses))

            action_a = parse_action(response_map.get("a")) if a_alive else "straight"
            action_b = parse_action(response_map.get("b")) if b_alive else "straight"

            # Update conversation history for each player
            if a_alive and action_a is not None:
                history_a.append({"role": "user", "content": turn_text_a})
                history_a.append({"role": "assistant", "content": action_a})
                # Keep only the last N turns
                if len(history_a) > MAX_HISTORY_TURNS * 2:
                    history_a[:] = history_a[-(MAX_HISTORY_TURNS * 2) :]
            if b_alive and action_b is not None:
                history_b.append({"role": "user", "content": turn_text_b})
                history_b.append({"role": "assistant", "content": action_b})
                if len(history_b) > MAX_HISTORY_TURNS * 2:
                    history_b[:] = history_b[-(MAX_HISTORY_TURNS * 2) :]

            # If constrained decoding failed, penalize and bail
            if a_alive and action_a is None:
                for s in args.results_dict["player_a"]:
                    s.reward = INVALID_REWARD
                for s in args.results_dict["player_b"]:
                    if s.reward is None:
                        s.reward = DEAD_REWARD
                print(
                    f"[curvytron-ma] action_a parse failed for seed {game_seed}, step {step}, raw_a={repr(response_map.get('a'))}"
                )
                _log_game(
                    {
                        "seed": game_seed,
                        "steps": step,
                        "outcome": "parse_fail_a",
                        "actions_a": actions_a,
                        "actions_b": actions_b,
                        "final_board": None,
                    }
                )
                return args.results_dict["player_a"] + args.results_dict["player_b"]
            if b_alive and action_b is None:
                for s in args.results_dict["player_b"]:
                    s.reward = INVALID_REWARD
                for s in args.results_dict["player_a"]:
                    if s.reward is None:
                        s.reward = DEAD_REWARD
                print(
                    f"[curvytron-ma] action_b parse failed for seed {game_seed}, step {step}"
                )
                _log_game(
                    {
                        "seed": game_seed,
                        "steps": step,
                        "outcome": "parse_fail_b",
                        "actions_a": actions_a,
                        "actions_b": actions_b,
                        "final_board": None,
                    }
                )
                return args.results_dict["player_a"] + args.results_dict["player_b"]

            if a_alive:
                actions_a.append(action_a)
            if b_alive:
                actions_b.append(action_b)

            # Submit both actions and advance the game synchronously
            state = await client.step(
                session_id,
                {actor_a["id"]: action_a, actor_b["id"]: action_b},
            )
            step += 1

            # Per-step reward: escalating survival + space bonus, or 0 on death
            a_still_alive = is_player_alive(state, actor_a["player_id"])
            b_still_alive = is_player_alive(state, actor_b["player_id"])

            if a_alive:
                r = (
                    compute_step_reward(step, state, actor_a["player_id"])
                    if a_still_alive
                    else DEAD_REWARD
                )
                args.results_dict["player_a"][-1].reward = r
                if not a_still_alive:
                    a_alive = False
            if b_alive:
                r = (
                    compute_step_reward(step, state, actor_b["player_id"])
                    if b_still_alive
                    else DEAD_REWARD
                )
                args.results_dict["player_b"][-1].reward = r
                if not b_still_alive:
                    b_alive = False

            a_rewards = [s.reward for s in args.results_dict["player_a"]]
            b_rewards = [s.reward for s in args.results_dict["player_b"]]
            if None in a_rewards or None in b_rewards:
                print(
                    f"[curvytron-ma] DEBUG step={step} seed={game_seed} a_rewards={a_rewards} b_rewards={b_rewards} a_alive={a_alive} b_alive={b_alive}"
                )

            if record:
                trace_steps.append(
                    _make_step_record(
                        step,
                        state,
                        action_a,
                        action_b,
                        a_rewards[-1] if a_rewards else None,
                        b_rewards[-1] if b_rewards else None,
                    )
                )

        all_samples = args.results_dict["player_a"] + args.results_dict["player_b"]
        if not all_samples:
            print(f"[curvytron-ma] Warning: no samples generated for seed {game_seed}")
            return []

        # Safety net: ensure no sample has None reward
        for s in all_samples:
            if s.reward is None:
                print(
                    f"[curvytron-ma] BUG: sample with None reward for seed {game_seed}, step {step}"
                )
                s.reward = DEAD_REWARD

        # ── Log game summary ───────────────────────────────────────────
        if a_alive and not b_alive:
            outcome = "a_wins"
        elif b_alive and not a_alive:
            outcome = "b_wins"
        elif not a_alive and not b_alive:
            outcome = "both_dead"
        else:
            outcome = "timeout"

        final_board = state.get("occupancy", {}).get("ascii")
        _log_game(
            {
                "seed": game_seed,
                "steps": step,
                "outcome": outcome,
                "actions_a": actions_a,
                "actions_b": actions_b,
                "final_board": final_board,
            }
        )

        if record and trace_steps:
            _save_trace(game_seed, outcome, step, trace_steps, marker_a, marker_b)

        return all_samples

    except Exception as e:
        print(f"[curvytron-ma] Game error for seed {game_seed}: {e}")
        return []

    finally:
        if session_id:
            await client.delete_session(session_id)
