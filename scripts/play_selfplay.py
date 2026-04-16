#!/usr/bin/env python3
"""
Self-play: two LLM bots play each other in a single process.
Both players block and wait for the other before advancing, just like training.

Usage:
    uv run python scripts/play_selfplay.py --room myroom
    uv run python scripts/play_selfplay.py --room myroom --llm-endpoint https://...
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import (
    ACTIONS, MAX_STEPS, ESC, CLEAR_SCREEN, HIDE_CURSOR, SHOW_CURSOR,
    BOLD, DIM, RESET, PLAYER_COLORS,
    render_frame, get_state, wait_for_server, find_or_create_session,
    add_bot, delete_session, add_common_args,
)
from prompts import SYSTEM_PROMPT

DEFAULT_LLM_ENDPOINT = "https://modal-labs-joy-dev--curvytron-bot-curvytronbot.us-east.modal.direct/generate"


# ── Action parsing ─────────────────────────────────────────────────────────

def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def parse_action(reply):
    clean = strip_thinking(reply)
    clean_lower = clean.lower().strip()
    if clean_lower in ACTIONS:
        return clean_lower
    for action in ACTIONS:
        if action in clean_lower:
            return action
    return random.choice(ACTIONS)


# ── LLM call ───────────────────────────────────────────────────────────────

def build_turn_prompt(state, my_player_id, my_marker):
    tick = state.get("tick", "?")
    occ = state.get("occupancy", {})
    ascii_board = occ.get("ascii", "(no board)")
    board_w = occ.get("width", "?")
    board_h = occ.get("height", "?")

    me, opponents = None, []
    for p in state.get("players", []):
        if p.get("player_id") == my_player_id:
            me = p
        else:
            opponents.append(p)

    board_info = state.get("board", {})
    borderless = board_info.get("borderless", False)
    border_note = " (BORDERLESS — edges wrap around!)" if borderless else ""
    lines = [f"## Tick {tick}  |  Board {board_w}x{board_h}{border_note}", ""]

    if me:
        alive_str = "ALIVE" if me.get("alive") else "DEAD"
        lines.append(
            f"**You** are player '{my_marker}' | "
            f"Position: ({me.get('x', 0):.1f}, {me.get('y', 0):.1f}) | "
            f"Angle: {me.get('angle', 0):.2f} rad | Status: {alive_str}"
        )

    lines.append("")
    for op in opponents:
        alive_str = "ALIVE" if op.get("alive") else "DEAD"
        lines.append(
            f"Opponent '{op.get('marker', '?')}' ({op.get('name', '?')}) | "
            f"Position: ({op.get('x', 0):.1f}, {op.get('y', 0):.1f}) | "
            f"Angle: {op.get('angle', 0):.2f} rad | Status: {alive_str}"
        )

    lines += ["", "**Board:**", "```", ascii_board, "```", "",
              "Choose your action: left, straight, or right?"]
    return "\n".join(lines)


# Match training: keep last N turns of history (each board ~780 tokens, context=4096)
MAX_HISTORY_TURNS = 3


def choose_action(llm_endpoint, state, player_id, marker, history):
    """Ask the LLM for an action. Stateful — includes conversation history
    to match training in multi_agent_system.py."""
    turn_msg = build_turn_prompt(state, player_id, marker)

    # Build prompt with history, matching training's format_chat_prompt
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    for msg in history:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

    # Empty think block = enable_thinking=False (same as training)
    prompt += f"<|im_start|>user\n{turn_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    try:
        payload = {
            "text": prompt,
            "sampling_params": {"temperature": 0.1, "max_new_tokens": 5},
            "regex": "(left|straight|right)",
        }
        resp = requests.post(llm_endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        reply = resp.json().get("text", "").strip()

        # Update history
        history.append({"role": "user", "content": turn_msg})
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history[:] = history[-(MAX_HISTORY_TURNS * 2):]

        action = reply.lower() if reply.lower() in ACTIONS else parse_action(reply)
        return action, reply
    except Exception as e:
        return random.choice(ACTIONS), f"(error: {e})"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-play: two LLM bots in one process")
    add_common_args(parser)
    parser.add_argument("--llm-endpoint", default=DEFAULT_LLM_ENDPOINT, help="LLM chat endpoint URL")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"}

    wait_for_server(base)

    session, _ = find_or_create_session(base, headers, args.room, args.grid, auto_advance=False, map_size=getattr(args, 'map_size', None))
    session_id = session["session_id"]
    print(f"Session: {session_id}")

    room_name = session.get("room_name", "")
    if not room_name and session_id.startswith("rl:"):
        room_name = f"rl-session-{session_id.split(':')[1]}"
    spectate_url = f"{base}/#/room/{room_name}" if room_name else None
    if spectate_url:
        print(f"Spectate: {spectate_url}")

    # Add two bots
    color_a = f"#{random.randint(0, 0xFFFFFF):06x}"
    color_b = f"#{random.randint(0, 0xFFFFFF):06x}"
    actor_a = add_bot(base, headers, session_id, "LLM-A", color_a)
    actor_b = add_bot(base, headers, session_id, "LLM-B", color_b)
    if not actor_a or not actor_b:
        print("Failed to add bots — deleting stale session and retrying")
        delete_session(base, headers, session_id)
        session, _ = find_or_create_session(base, headers, args.room, args.grid, auto_advance=True)
        session_id = session["session_id"]
        actor_a = add_bot(base, headers, session_id, "LLM-A", color_a)
        actor_b = add_bot(base, headers, session_id, "LLM-B", color_b)

    pid_a, aid_a = actor_a["player_id"], actor_a["id"]
    pid_b, aid_b = actor_b["player_id"], actor_b["id"]

    # Find markers
    state = get_state(base, headers, session_id)
    marker_a = marker_b = "?"
    for p in state.get("players", []):
        if p.get("player_id") == pid_a:
            marker_a = p.get("marker", "?")
        elif p.get("player_id") == pid_b:
            marker_b = p.get("marker", "?")

    print(f"Player A: {marker_a} ({pid_a})")
    print(f"Player B: {marker_b} ({pid_b})")

    # Start game
    resp = requests.post(f"{base}/api/rl/sessions/{session_id}/start", headers=headers)
    if resp.status_code != 200:
        print(f"Failed to start: {resp.status_code} {resp.text}")
        sys.exit(1)
    state = resp.json()

    history_a, history_b = [], []
    executor = ThreadPoolExecutor(max_workers=2)

    action_log = []
    MAX_ACTION_LOG = 10

    def log_action(step, marker, action, raw):
        action_log.append((step, marker, action, raw))
        if len(action_log) > MAX_ACTION_LOG:
            action_log.pop(0)

    def action_log_lines():
        lines = [f"{DIM}{'─' * 60}{RESET}", f"{BOLD}Self-Play Action Log:{RESET}"]
        if not action_log:
            lines.append("  (waiting...)")
        for s, m, act, raw in action_log:
            color = PLAYER_COLORS[0] if m == marker_a else PLAYER_COLORS[1]
            lines.append(f"  step {s:>4} {color}{m}{RESET}: {BOLD}{act:<8}{RESET}  raw: {raw!r}")
        return lines

    sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
    sys.stdout.flush()

    try:
        step = 0
        while step < MAX_STEPS and not state.get("done", False):
            a_alive = any(p.get("alive") for p in state.get("players", []) if p.get("player_id") == pid_a)
            b_alive = any(p.get("alive") for p in state.get("players", []) if p.get("player_id") == pid_b)

            render_frame(state, pid_a, step, "self-play [thinking...]", action_log_lines(), spectate_url=spectate_url)

            # Both players think in parallel, then both send actions
            futures = {}
            if a_alive:
                futures["a"] = executor.submit(choose_action, args.llm_endpoint, state, pid_a, marker_a, history_a)
            if b_alive:
                futures["b"] = executor.submit(choose_action, args.llm_endpoint, state, pid_b, marker_b, history_b)

            action_a = "straight"
            action_b = "straight"

            if "a" in futures:
                action_a, raw_a = futures["a"].result()
                log_action(step, marker_a, action_a, raw_a)
            if "b" in futures:
                action_b, raw_b = futures["b"].result()
                log_action(step, marker_b, action_b, raw_b)

            # Send both actions
            requests.post(f"{base}/api/rl/sessions/{session_id}/actors/{aid_a}/action",
                          json={"action": action_a}, headers=headers)
            requests.post(f"{base}/api/rl/sessions/{session_id}/actors/{aid_b}/action",
                          json={"action": action_b}, headers=headers)

            # Advance the game tick (auto_advance=False means we drive it)
            requests.post(f"{base}/api/rl/sessions/{session_id}/step", headers=headers)

            # Get new state
            state = get_state(base, headers, session_id)
            step += 1

            status = f"self-play [{marker_a}={action_a} {marker_b}={action_b}]"
            render_frame(state, pid_a, step, status, action_log_lines(), spectate_url=spectate_url)

        render_frame(state, pid_a, step, "self-play [DONE]", action_log_lines(), spectate_url=spectate_url)

    finally:
        delete_session(base, headers, session_id)
        executor.shutdown(wait=False)
        sys.stdout.write(SHOW_CURSOR + f"\n{ESC}[{shutil.get_terminal_size((100, 40)).lines}H\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
