"""Self-play agent system for curvytron multi-agent GRPO training.

Two agents (both the training model) play against each other in a full game
episode. Each per-turn generation is recorded as a SLIME Sample with logprobs.
Reward per step: 1.0 if the agent is still alive after the action, 0.0 if dead.
"""

import asyncio
from copy import deepcopy

from slime.utils.http_utils import post
from slime.utils.types import Sample

from .game_client import AsyncGameClient, MAX_STEPS
from .prompts import SYSTEM_PROMPT

ACTIONS = ["left", "straight", "right"]

# Constrained decoding regex — forces SGLang to output exactly one valid action
ACTION_REGEX = r"(left|straight|right)"
MAX_ACTION_TOKENS = 5

# Per-step rewards
ALIVE_REWARD = 1.0
DEAD_REWARD = 0.0
INVALID_REWARD = -10.0


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
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ── Action parsing ──────────────────────────────────────────────────────────


def parse_action(reply):
    """Parse action from constrained-decoded output. Returns None if failed."""
    if not reply:
        return None
    clean = reply.strip().lower()
    if clean in ACTIONS:
        return clean
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
        print(f"[curvytron-ma] generate_response error: {e}")
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────


def is_player_alive(state: dict, player_id: str) -> bool:
    for p in state.get("players", []):
        if p.get("player_id") == player_id:
            return p.get("alive", False)
    return False


# ── Main self-play loop ─────────────────────────────────────────────────────


async def run_selfplay_game(args, sample: Sample):
    """Play a full self-play game and return scored Samples for both players.

    Two agents (both the training model) play against each other.
    Each action generates a Sample; reward = 1.0 if alive after that step,
    0.0 if dead.
    """
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"player_a": [], "player_b": []}

    game_seed = sample.prompt
    tokenizer = args.tokenizer

    client = AsyncGameClient()
    session_id = None

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

        # ── Game loop ───────────────────────────────────────────────────
        step = 0
        a_alive, b_alive = True, True

        while step < MAX_STEPS and not state.get("done", False):
            # Generate actions for living players
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

            # If constrained decoding failed, penalize and bail
            if a_alive and action_a is None:
                for s in args.results_dict["player_a"]:
                    s.reward = INVALID_REWARD
                for s in args.results_dict["player_b"]:
                    if s.reward is None:
                        s.reward = DEAD_REWARD
                print(f"[curvytron-ma] action_a parse failed for seed {game_seed}, step {step}, raw_a={repr(response_map.get('a'))}")
                return args.results_dict["player_a"] + args.results_dict["player_b"]
            if b_alive and action_b is None:
                for s in args.results_dict["player_b"]:
                    s.reward = INVALID_REWARD
                for s in args.results_dict["player_a"]:
                    if s.reward is None:
                        s.reward = DEAD_REWARD
                print(f"[curvytron-ma] action_b parse failed for seed {game_seed}, step {step}")
                return args.results_dict["player_a"] + args.results_dict["player_b"]

            # Send actions
            await asyncio.gather(
                client.send_action(session_id, actor_a["id"], action_a),
                client.send_action(session_id, actor_b["id"], action_b),
            )

            # Get updated state
            state = await client.get_state(session_id)
            step += 1

            # Per-step reward: alive → 1.0, dead → 0.0
            a_still_alive = is_player_alive(state, actor_a["player_id"])
            b_still_alive = is_player_alive(state, actor_b["player_id"])

            if a_alive:
                r = ALIVE_REWARD if a_still_alive else DEAD_REWARD
                args.results_dict["player_a"][-1].reward = r
                if not a_still_alive:
                    a_alive = False
            if b_alive:
                r = ALIVE_REWARD if b_still_alive else DEAD_REWARD
                args.results_dict["player_b"][-1].reward = r
                if not b_still_alive:
                    b_alive = False

            a_rewards = [s.reward for s in args.results_dict["player_a"]]
            b_rewards = [s.reward for s in args.results_dict["player_b"]]
            if None in a_rewards or None in b_rewards:
                print(f"[curvytron-ma] DEBUG step={step} seed={game_seed} a_rewards={a_rewards} b_rewards={b_rewards} a_alive={a_alive} b_alive={b_alive}")

        all_samples = args.results_dict["player_a"] + args.results_dict["player_b"]
        if not all_samples:
            print(f"[curvytron-ma] Warning: no samples generated for seed {game_seed}")
            return []

        # Safety net: ensure no sample has None reward
        for s in all_samples:
            if s.reward is None:
                print(f"[curvytron-ma] BUG: sample with None reward for seed {game_seed}, step {step}")
                s.reward = DEAD_REWARD

        return all_samples

    except Exception as e:
        print(f"[curvytron-ma] Game error for seed {game_seed}: {e}")
        return []

    finally:
        if session_id:
            await client.delete_session(session_id)
