"""Self-play agent system for curvytron GRPO training.

Two agents (both the training model) play against each other in a full game
episode. Each per-turn generation is recorded as a SLIME Sample with logprobs.
After the game ends, rewards are assigned retroactively to all turns based on
survival steps and win/loss outcome.
"""

import asyncio
from copy import deepcopy

from slime.utils.http_utils import post
from slime.utils.types import Sample

from .game_client import AsyncGameClient, MAX_STEPS
from .prompts import SYSTEM_PROMPT

ACTIONS = ["left", "straight", "right"]
WIN_BONUS = 10.0  # bonus seconds for winning
TICK_MS = 16.0    # ms per game tick

# Constrained decoding regex — forces SGLang to output exactly one valid action
ACTION_REGEX = r"(left|straight|right)"

# Max action tokens (safety cap, regex should terminate well before this)
MAX_ACTION_TOKENS = 5

# Reward penalty for invalid/failed generations
INVALID_REWARD = -10.0


# ── Prompt building ─────────────────────────────────────────────────────────


def build_turn_prompt(state: dict, my_player_id: str, my_marker: str) -> str:
    """Build the user-turn message describing the current board state.

    State is always complete here (synchronous HTTP from game server),
    so no fallback defaults needed unlike the WebSocket-based play scripts.
    """
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

    lines += ["", "**Board:**", "```", ascii_board, "```", "",
              "Choose your action: left, straight, or right?"]
    return "\n".join(lines)


def format_chat_prompt(tokenizer, system_prompt: str, user_message: str) -> str:
    """Apply the chat template to produce a full prompt string."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Tokenizer doesn't support enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ── Action parsing ──────────────────────────────────────────────────────────


def parse_action(reply: str | None) -> str | None:
    """Parse action from constrained-decoded output.

    With regex-guided decoding, the reply is guaranteed to be one of
    "left", "straight", "right". Returns None only if generation failed
    entirely (reply is None or empty), which the caller must handle
    explicitly rather than silently falling back to random.
    """
    if not reply:
        return None
    clean = reply.strip().lower()
    if clean in ACTIONS:
        return clean
    return None


# ── SGLang generation (adapted from SLIME multi-agent example) ──────────────


async def generate_response(args, prompt: str, key: str):
    """Generate a single action response via SGLang.

    Mirrors the SLIME multi-agent generate_response pattern:
    - Tokenizes prompt, sends to SGLang, collects logprobs
    - Creates a Sample and appends to args.results_dict[key]
    - Returns the response text
    """
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
        # Cap response length — we only need a single action word
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

        # Extract response tokens
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
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
        print(f"[curvytron] generate_response error: {e}")
        return None


# ── Reward computation ──────────────────────────────────────────────────────


def is_player_alive(state: dict, player_id: str) -> bool:
    for p in state.get("players", []):
        if p.get("player_id") == player_id:
            return p.get("alive", False)
    return False


def compute_rewards(
    results_dict: dict,
    final_state: dict,
    actor_a: dict,
    actor_b: dict,
    death_tick_a: int | None,
    death_tick_b: int | None,
    final_tick: int,
):
    """Assign rewards retroactively to all per-turn Samples.

    Reward = survival_seconds + WIN_BONUS if won.
    All turns for a given player receive the same episode-level reward.
    """
    winner_pid = final_state.get("round_winner_player_id")
    a_won = winner_pid == actor_a["player_id"] if winner_pid else False
    b_won = winner_pid == actor_b["player_id"] if winner_pid else False

    survival_tick_a = death_tick_a if death_tick_a is not None else final_tick
    survival_tick_b = death_tick_b if death_tick_b is not None else final_tick

    survival_sec_a = survival_tick_a * TICK_MS / 1000.0
    survival_sec_b = survival_tick_b * TICK_MS / 1000.0

    reward_a = survival_sec_a + (WIN_BONUS if a_won else 0)
    reward_b = survival_sec_b + (WIN_BONUS if b_won else 0)

    for sample in results_dict["player_a"]:
        sample.reward = reward_a
    for sample in results_dict["player_b"]:
        sample.reward = reward_b


def _penalize_samples(samples: list):
    """Mark all samples from a player as penalized due to invalid generation."""
    for s in samples:
        s.reward = INVALID_REWARD


# ── Main self-play loop ─────────────────────────────────────────────────────


async def run_selfplay_game(args, sample: Sample):
    """Play a full self-play game and return scored Samples for both players.

    This is the core function called by the SLIME rollout entry point.
    """
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"player_a": [], "player_b": []}

    game_seed = sample.prompt  # e.g. "rl-seed-42"
    tokenizer = args.tokenizer

    client = AsyncGameClient()

    session_id = None
    try:
        # ── Setup game ──────────────────────────────────────────────────
        session = await client.create_session(game_seed)
        session_id = session["session_id"]

        actor_a = await client.add_bot(session_id, "Agent-A", "#ff4444")
        actor_b = await client.add_bot(session_id, "Agent-B", "#4444ff")

        # No wait_for_players needed — we just added both bots synchronously
        state = await client.start_game(session_id)

        # Find markers (A, B, etc.)
        marker_a, marker_b = "?", "?"
        for p in state.get("players", []):
            if p.get("player_id") == actor_a["player_id"]:
                marker_a = p.get("marker", "?")
            elif p.get("player_id") == actor_b["player_id"]:
                marker_b = p.get("marker", "?")

        # ── Game loop ───────────────────────────────────────────────────
        step = 0
        a_alive, b_alive = True, True
        death_tick_a, death_tick_b = None, None

        while step < MAX_STEPS and not state.get("done", False):
            # Only generate actions for living players — skip dead ones
            # to avoid wasting SGLang compute and creating junk Samples
            gen_tasks = []
            gen_keys = []
            if a_alive:
                turn_text_a = build_turn_prompt(state, actor_a["player_id"], marker_a)
                prompt_a = format_chat_prompt(tokenizer, SYSTEM_PROMPT, turn_text_a)
                gen_tasks.append(generate_response(args, prompt_a, key="player_a"))
                gen_keys.append("a")
            if b_alive:
                turn_text_b = build_turn_prompt(state, actor_b["player_id"], marker_b)
                prompt_b = format_chat_prompt(tokenizer, SYSTEM_PROMPT, turn_text_b)
                gen_tasks.append(generate_response(args, prompt_b, key="player_b"))
                gen_keys.append("b")

            responses = await asyncio.gather(*gen_tasks)
            response_map = dict(zip(gen_keys, responses))

            action_a = parse_action(response_map.get("a")) if a_alive else "straight"
            action_b = parse_action(response_map.get("b")) if b_alive else "straight"

            # If constrained decoding failed, mark invalid and bail —
            # don't silently inject random actions into training data
            if a_alive and action_a is None:
                _penalize_samples(args.results_dict["player_a"])
                return args.results_dict["player_a"] + args.results_dict["player_b"]
            if b_alive and action_b is None:
                _penalize_samples(args.results_dict["player_b"])
                return args.results_dict["player_a"] + args.results_dict["player_b"]

            # Send both actions to game server
            await asyncio.gather(
                client.send_action(session_id, actor_a["id"], action_a),
                client.send_action(session_id, actor_b["id"], action_b),
            )

            # Get updated state
            state = await client.get_state(session_id)
            step += 1

            # Track death ticks
            if a_alive and not is_player_alive(state, actor_a["player_id"]):
                a_alive = False
                death_tick_a = state.get("tick", step)
            if b_alive and not is_player_alive(state, actor_b["player_id"]):
                b_alive = False
                death_tick_b = state.get("tick", step)

        # ── Compute and assign rewards ──────────────────────────────────
        final_tick = state.get("tick", step)
        compute_rewards(
            args.results_dict, state,
            actor_a, actor_b,
            death_tick_a, death_tick_b,
            final_tick,
        )

        all_samples = args.results_dict["player_a"] + args.results_dict["player_b"]

        # If no samples were generated (game failed immediately), return empty
        if not all_samples:
            print(f"[curvytron] Warning: no samples generated for seed {game_seed}")
            return []

        return all_samples

    except Exception as e:
        print(f"[curvytron] Game error for seed {game_seed}: {e}")
        return []

    finally:
        if session_id:
            await client.delete_session(session_id)
