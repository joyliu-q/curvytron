#!/usr/bin/env python3
"""
Play a curvytron game using an LLM endpoint to choose moves.
Each instance controls exactly 1 player. Run two instances with the same room
to get a 2-player game (or pair with play_random.py for an LLM-vs-random match).

Usage:
    # Terminal 1 (LLM player):
    uv run python scripts/play_llm.py --room myroom

    # Terminal 2 (random opponent):
    uv run python scripts/play_random.py --room myroom
"""

import argparse
import json
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from urllib.parse import urlencode

import requests
import websocket  # websocket-client package

from common import (
    ACTIONS, MAX_STEPS, ESC, CLEAR_SCREEN, HIDE_CURSOR, SHOW_CURSOR,
    BOLD, DIM, RESET,
    render_frame, get_state, setup_session, add_common_args,
)

DEFAULT_LLM_ENDPOINT = "https://joyliu-q--curvytron-player-curvytronplayer.us-east.modal.direct/v1/chat/completions"

# How many game ticks to hold each LLM action before asking again
DEFAULT_HOLD_TICKS = 3


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are playing Curvytron, a multiplayer Snake/Tron-like game on a 2D grid.

## Rules
- You control a continuously moving avatar that leaves a trail behind it.
- Each tick you choose one of three actions: "left", "straight", or "right".
  - "left" turns your avatar left relative to its current heading.
  - "straight" keeps your current heading.
  - "right" turns your avatar right relative to its current heading.
- If your avatar collides with any trail (yours or an opponent's), a wall, or \
the border, you die.
- The last player alive wins the round.

## Board notation
The board is an ASCII grid where:
- "." = empty space (safe to move into)
- "#" = wall / border / trail segment (deadly on contact)
- Uppercase letters (A, B, …) = player head positions
- Lowercase letters = bonuses (power-ups) on the board

PLAY DEFENSIVELY. TRY TO SURVIVE.

## Response format

Respond with ONLY one word: left, straight, or right
"""


def build_turn_prompt(state, my_player_id, my_marker, llm_board=None):
    """Build the user message for the current game tick."""
    tick = state.get("tick", "?")
    players = state.get("players", [])

    # Use the low-res LLM board if available, otherwise fall back to state occupancy
    if llm_board:
        ascii_board = llm_board.get("ascii", "(no board)")
        board_w = llm_board.get("width", "?")
        board_h = llm_board.get("height", "?")
    else:
        occ = state.get("occupancy", {})
        ascii_board = occ.get("ascii", "(no board)")
        board_w = occ.get("width", "?")
        board_h = occ.get("height", "?")

    me = None
    opponents = []
    for p in players:
        if p.get("player_id") == my_player_id:
            me = p
        else:
            opponents.append(p)

    board_info = state.get("board", {})
    borderless = board_info.get("borderless", False)

    border_note = " (BORDERLESS — edges wrap around!)" if borderless else ""
    lines = [f"## Tick {tick}  |  Board {board_w}x{board_h}{border_note}"]
    lines.append("")

    if me:
        alive_str = "ALIVE" if me.get("alive") else "DEAD"
        x, y = me.get('x'), me.get('y')
        ang = me.get('angle')
        pos_str = f"({x:.1f}, {y:.1f})" if x is not None and y is not None else "(?, ?)"
        ang_str = f"{ang:.2f} rad" if ang is not None else "?"
        lines.append(
            f"**You** are player '{my_marker}' | "
            f"Position: {pos_str} | "
            f"Angle: {ang_str} | "
            f"Status: {alive_str}"
        )
        if me.get("printing") is False:
            lines.append("  !! You are currently in a GAP — not leaving a trail")
        if me.get("inverse"):
            lines.append("  !! Controls are INVERTED — left/right are swapped!")
        if me.get("invincible"):
            lines.append("  !! You are INVINCIBLE — you can pass through trails!")
        active = me.get("active_bonuses", [])
        if active:
            for ab in active:
                rem = ab.get("remaining_ms")
                rem_str = f"{rem / 1000:.1f}s left" if rem is not None else "permanent"
                lines.append(f"  Active buff: {ab['type']} ({rem_str})")

    lines.append("")
    for op in opponents:
        alive_str = "ALIVE" if op.get("alive") else "DEAD"
        x, y = op.get('x'), op.get('y')
        ang = op.get('angle')
        pos_str = f"({x:.1f}, {y:.1f})" if x is not None and y is not None else "(?, ?)"
        ang_str = f"{ang:.2f} rad" if ang is not None else "?"
        lines.append(
            f"Opponent '{op.get('marker', '?')}' ({op.get('name', '?')}) | "
            f"Position: {pos_str} | "
            f"Angle: {ang_str} | "
            f"Status: {alive_str}"
        )
        op_active = op.get("active_bonuses", [])
        if op_active:
            for ab in op_active:
                rem = ab.get("remaining_ms")
                rem_str = f"{rem / 1000:.1f}s left" if rem is not None else "permanent"
                lines.append(f"  Their active debuff: {ab['type']} ({rem_str})")

    bonuses = state.get("bonuses", [])
    if bonuses:
        lines.append("")
        lines.append("**Active bonuses on the board:**")
        for b in bonuses:
            lines.append(f"  - {b['type']} at ({b['x']:.1f}, {b['y']:.1f}) [marker: '{b['marker']}']")

    lines.append("")
    lines.append("**Board:**")
    lines.append("```")
    lines.append(ascii_board)
    lines.append("```")
    lines.append("")
    lines.append("Choose your action: left, straight, or right?")

    return "\n".join(lines)


# ── LLM decision-making ─────────────────────────────────────────────────────

def strip_thinking(text):
    """Remove <think>...</think> blocks from Qwen3 responses."""
    # Remove complete thinking blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove unclosed thinking block (truncated output)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def parse_action(reply):
    """Extract an action from the LLM's response, with fallback."""
    # Strip thinking blocks first — only parse the actual answer
    clean = strip_thinking(reply)
    clean_lower = clean.lower().strip()

    # Direct match (constrained decoding or single-word answer)
    if clean_lower in ACTIONS:
        return clean_lower

    # Try JSON parse
    try:
        data = json.loads(clean)
        action = data.get("action", "").lower().strip()
        if action in ACTIONS:
            return action
    except (json.JSONDecodeError, AttributeError):
        pass

    # Look for action word in the clean (non-thinking) text
    for action in ACTIONS:
        if action in clean_lower:
            return action

    # Last resort: look in the full reply (including thinking)
    reply_lower = reply.lower()
    # Find the LAST action word mentioned (more likely to be the conclusion)
    last_action = None
    last_pos = -1
    for action in ACTIONS:
        pos = reply_lower.rfind(action)
        if pos > last_pos:
            last_pos = pos
            last_action = action
    if last_action:
        return last_action

    return random.choice(ACTIONS)


def choose_action_llm(llm_endpoint, state, my_player_id, my_marker, conversation_history, llm_board=None):
    """Ask the LLM endpoint (OpenAI-compatible) to choose an action.

    Uses guided_choice for constrained decoding (single token output) and
    a compact prompt to minimize input tokens.
    """
    turn_msg = build_turn_prompt(state, my_player_id, my_marker, llm_board)

    # Keep a short sliding window to give the model context of recent moves
    max_history = 5
    if len(conversation_history) > max_history * 2:
        conversation_history[:] = conversation_history[-(max_history * 2):]

    conversation_history.append({"role": "user", "content": turn_msg})

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

        payload = {
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.0, # no randomness
            "guided_choice": ACTIONS,
            # Disable Qwen3 thinking mode — we just need one word
            "chat_template_kwargs": {"enable_thinking": False},
        }

        resp = requests.post(llm_endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        reply = data["choices"][0]["message"]["content"].strip()
        conversation_history.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        return action, reply
    except Exception as e:
        if conversation_history and conversation_history[-1].get("role") == "user":
            conversation_history.pop()
        return random.choice(ACTIONS), f"(fallback: {e})"


# ── WebSocket helpers ────────────────────────────────────────────────────────

def connect_ws(base, session_id, token=None):
    """Open a WebSocket to the RL session and return (ws, initial_state)."""
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://")
    params = {"session": session_id}
    if token:
        params["token"] = token
    ws_url += f"/api/rl/ws?{urlencode(params)}"
    ws = websocket.create_connection(ws_url, timeout=30)
    ack = json.loads(ws.recv())
    if ack.get("type") != "connected":
        raise RuntimeError(f"WS handshake failed: {ack}")
    return ws


def drain_ws(ws):
    """Read all pending WS messages, return the latest state (or None)."""
    latest = None
    ws.settimeout(0)
    try:
        while True:
            raw = ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "state":
                latest = msg["data"]
    except (websocket.WebSocketTimeoutException, BlockingIOError):
        pass
    finally:
        ws.settimeout(30)
    return latest



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play curvytron with an LLM (1 player per script)")
    add_common_args(parser)
    parser.add_argument("--llm-endpoint", default=DEFAULT_LLM_ENDPOINT, help="LLM chat endpoint URL")
    parser.add_argument("--hold-ticks", type=int, default=DEFAULT_HOLD_TICKS,
                        help="Game steps to hold each LLM action")
    args = parser.parse_args()

    hold_ticks = max(1, args.hold_ticks)

    base, headers, session_id, is_creator, my_actor_id, my_player_id, bot_name, spectate_url = setup_session(args)

    # Figure out our marker (A, B, etc.)
    state = get_state(base, headers, session_id)
    my_marker = "?"
    for p in state.get("players", []):
        if p.get("player_id") == my_player_id:
            my_marker = p.get("marker", "?")
            break

    conversation_history = []
    executor = ThreadPoolExecutor(max_workers=1)

    # Action log: rolling window of recent LLM decisions
    MAX_ACTION_LOG = 10
    action_log = []  # list of (step, action, raw_reply)

    def log_action(step_num, action, raw_reply):
        action_log.append((step_num, action, raw_reply))
        if len(action_log) > MAX_ACTION_LOG:
            action_log.pop(0)

    def action_log_lines():
        """Build extra lines showing recent LLM actions."""
        lines = [f"{DIM}{'─' * 60}{RESET}", f"{BOLD}LLM Action Log:{RESET}"]
        if not action_log:
            lines.append("  (waiting for first LLM response...)")
        for s, act, raw in action_log:
            lines.append(f"  step {s:>4}: {BOLD}{act:<8}{RESET}  raw: {raw!r}")
        return lines

    # --- hide cursor, clear screen for ANSI rendering ---
    sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
    sys.stdout.flush()

    # Try WebSocket for fast action/state communication; fall back to HTTP
    token = args.token or None
    ws = None
    try:
        ws = connect_ws(base, session_id, token)
    except Exception as e:
        print(f"WebSocket connection failed ({e}), using HTTP fallback")

    # Shared mutable state for the receiver thread
    latest_state = {"state": state, "lock": threading.Lock()}

    if ws:
        def ws_receiver():
            """Background thread: receive state pushes and keep latest_state up to date."""
            try:
                while True:
                    raw = ws.recv()
                    if raw is None:
                        break
                    msg = json.loads(raw)
                    if msg.get("type") == "state":
                        with latest_state["lock"]:
                            latest_state["state"] = msg["data"]
            except Exception:
                pass

        receiver_thread = threading.Thread(target=ws_receiver, daemon=True)
        receiver_thread.start()

    def get_latest_state():
        if ws:
            with latest_state["lock"]:
                return latest_state["state"]
        return get_state(base, headers, session_id)

    def send_action(action):
        """Send an action over WebSocket, or fall back to HTTP."""
        if ws:
            ws.send(json.dumps({
                "type": "action",
                "actor_id": my_actor_id,
                "action": action,
            }))
        else:
            requests.post(
                f"{base}/api/rl/sessions/{session_id}/actors/{my_actor_id}/action",
                json={"action": action},
                headers=headers,
            )

    def request_llm(st):
        """Submit an LLM request in background. Uses state occupancy directly
        instead of a separate HTTP board fetch to minimize latency."""
        def _do_llm():
            occ = st.get("occupancy")
            return choose_action_llm(
                args.llm_endpoint, st, my_player_id, my_marker,
                conversation_history, occ,
            )
        return executor.submit(_do_llm)

    try:
        if is_creator:
            # Start the episode via HTTP (reliable even without WS)
            resp = requests.post(f"{base}/api/rl/sessions/{session_id}/start", headers=headers)
            if resp.status_code != 200:
                print(f"Failed to start episode: {resp.status_code} {resp.text}")
                sys.exit(1)
            state = resp.json()
            with latest_state["lock"]:
                latest_state["state"] = state
            render_frame(state, my_player_id, 0, "creator (llm) waiting...", action_log_lines(), spectate_url=spectate_url)

            step = 0
            next_future = request_llm(state)

            while step < MAX_STEPS and not state.get("done", False):
                render_frame(state, my_player_id, step, "llm [thinking...]", action_log_lines(), spectate_url=spectate_url)
                current_action, raw_reply = next_future.result()
                log_action(step, current_action, raw_reply)

                # Send action and immediately start next LLM call in parallel
                send_action(current_action)
                state = get_latest_state()
                step += 1

                if not state.get("done", False):
                    next_future = request_llm(state)

                render_frame(state, my_player_id, step, f"llm [{current_action}]", action_log_lines(), spectate_url=spectate_url)

            render_frame(state, my_player_id, step, "llm [done]", action_log_lines(), spectate_url=spectate_url)
            requests.delete(f"{base}/api/rl/sessions/{session_id}", headers=headers)

        else:
            step = 0
            next_future = request_llm(state)

            while step < MAX_STEPS and not state.get("done", False):
                render_frame(state, my_player_id, step, "llm [thinking...]", action_log_lines(), spectate_url=spectate_url)
                current_action, raw_reply = next_future.result()
                log_action(step, current_action, raw_reply)

                send_action(current_action)
                state = get_latest_state()
                step += 1

                if not state.get("done", False):
                    next_future = request_llm(state)

                render_frame(state, my_player_id, step, f"llm [{current_action}]", action_log_lines(), spectate_url=spectate_url)

            render_frame(state, my_player_id, step, "llm [done]", action_log_lines(), spectate_url=spectate_url)

    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        executor.shutdown(wait=False)
        sys.stdout.write(SHOW_CURSOR + f"\n{ESC}[{shutil.get_terminal_size((100, 40)).lines}H\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
