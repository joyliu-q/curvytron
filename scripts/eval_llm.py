#!/usr/bin/env python3
"""
Evaluation harness: pit two (LLM endpoint, system prompt) configurations
against each other across multiple maps and score them.

Reward per player per match:
    survival_seconds + 10  (if the player won the round and survived for longer than 10 seconds)

Where survival_seconds = death_tick * tick_ms / 1000  (default tick_ms=16).

Usage:
    uv run python scripts/eval_llm.py --maps 10
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from urllib.parse import urlencode

import requests
import websocket

from common import (
    ACTIONS, MAX_STEPS,
    wait_for_server, find_or_create_session, add_bot, delete_session,
    get_state, wait_for_players,
)

# ── Default configs ──────────────────────────────────────────────────────────

WIN_BONUS = 10  # seconds added to reward for winning

DEFAULT_SERVER = os.environ.get(
    "CURVYTRON_URL",
    "https://joyliu-q--curvytron-curvytron.us-east.modal.direct",
)

DEFAULT_ENDPOINT_A = os.environ.get(
    "EVAL_ENDPOINT_A",
    "https://joyliu-q--curvytron-player-curvytronplayer.us-east.modal.direct/v1/chat/completions",
)
DEFAULT_ENDPOINT_B = os.environ.get(
    "EVAL_ENDPOINT_B",
    "https://joyliu-q--curvytron-player-curvytronplayer.us-east.modal.direct/v1/chat/completions",
)

from prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B


# ── LLM helpers (copied/adapted from play_llm.py) ───────────────────────────

def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def parse_action(reply):
    clean = strip_thinking(reply).lower().strip()
    if clean in ACTIONS:
        return clean
    try:
        data = json.loads(clean)
        action = data.get("action", "").lower().strip()
        if action in ACTIONS:
            return action
    except (json.JSONDecodeError, AttributeError):
        pass
    for action in ACTIONS:
        if action in clean:
            return action
    reply_lower = reply.lower()
    last_action, last_pos = None, -1
    for action in ACTIONS:
        pos = reply_lower.rfind(action)
        if pos > last_pos:
            last_pos, last_action = pos, action
    if last_action:
        return last_action
    return random.choice(ACTIONS)


def build_turn_prompt(state, my_player_id, my_marker, llm_board=None):
    tick = state.get("tick", "?")
    players = state.get("players", [])

    if llm_board:
        ascii_board = llm_board.get("ascii", "(no board)")
        board_w = llm_board.get("width", "?")
        board_h = llm_board.get("height", "?")
    else:
        occ = state.get("occupancy", {})
        ascii_board = occ.get("ascii", "(no board)")
        board_w = occ.get("width", "?")
        board_h = occ.get("height", "?")

    me, opponents = None, []
    for p in players:
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
        x, y = me.get('x'), me.get('y')
        ang = me.get('angle')
        pos_str = f"({x:.1f}, {y:.1f})" if x is not None and y is not None else "(?, ?)"
        ang_str = f"{ang:.2f} rad" if ang is not None else "?"
        lines.append(
            f"**You** are player '{my_marker}' | "
            f"Position: {pos_str} | Angle: {ang_str} | Status: {alive_str}"
        )

    lines.append("")
    for op in opponents:
        alive_str = "ALIVE" if op.get("alive") else "DEAD"
        x, y = op.get('x'), op.get('y')
        ang = op.get('angle')
        pos_str = f"({x:.1f}, {y:.1f})" if x is not None and y is not None else "(?, ?)"
        ang_str = f"{ang:.2f} rad" if ang is not None else "?"
        lines.append(
            f"Opponent '{op.get('marker', '?')}' ({op.get('name', '?')}) | "
            f"Position: {pos_str} | Angle: {ang_str} | Status: {alive_str}"
        )

    lines += ["", "**Board:**", "```", ascii_board, "```", "",
              "Choose your action: left, straight, or right?"]
    return "\n".join(lines)


def choose_action(endpoint, system_prompt, state, my_player_id, my_marker, history):
    turn_msg = build_turn_prompt(state, my_player_id, my_marker,
                                  state.get("occupancy"))
    max_history = 5
    if len(history) > max_history * 2:
        history[:] = history[-(max_history * 2):]
    history.append({"role": "user", "content": turn_msg})

    try:
        messages = [{"role": "system", "content": system_prompt}] + history
        payload = {
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.0,
            "guided_choice": ACTIONS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = requests.post(endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": reply})
        return parse_action(reply)
    except Exception:
        if history and history[-1].get("role") == "user":
            history.pop()
        return random.choice(ACTIONS)


# ── WebSocket helpers ────────────────────────────────────────────────────────

def connect_ws(base, session_id, token=None):
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


# ── Single-player game loop (runs in a thread) ──────────────────────────────

def _is_player_alive(state, player_id):
    """Check if a specific player is alive in the given state."""
    for p in state.get("players", []):
        if p.get("player_id") == player_id:
            return p.get("alive", False)
    return False


def run_player(
    base, headers, session_id, actor_id, player_id,
    endpoint, system_prompt, is_creator, result_holder, player_label,
    token=None, verbose=False,
):
    """
    Play one full round. Writes into result_holder dict when done.
    Tracks death_tick: the game tick when this player died (None if survived).
    """
    state = get_state(base, headers, session_id)
    my_marker = "?"
    for p in state.get("players", []):
        if p.get("player_id") == player_id:
            my_marker = p.get("marker", "?")
            break

    history = []
    executor = ThreadPoolExecutor(max_workers=1)

    # Connect WebSocket
    ws = None
    latest_state = {"state": state, "lock": threading.Lock()}
    try:
        ws = connect_ws(base, session_id, token)
    except Exception:
        pass

    if ws:
        def ws_receiver():
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
        threading.Thread(target=ws_receiver, daemon=True).start()

    def get_latest():
        if ws:
            with latest_state["lock"]:
                return latest_state["state"]
        return get_state(base, headers, session_id)

    def send_action(action):
        if ws:
            ws.send(json.dumps({
                "type": "action",
                "actor_id": actor_id,
                "action": action,
            }))
        else:
            requests.post(
                f"{base}/api/rl/sessions/{session_id}/actors/{actor_id}/action",
                json={"action": action}, headers=headers,
            )

    def request_llm(st):
        def _do():
            return choose_action(endpoint, system_prompt, st, player_id, my_marker, history)
        return executor.submit(_do)

    try:
        if is_creator:
            resp = requests.post(f"{base}/api/rl/sessions/{session_id}/start", headers=headers)
            if resp.status_code != 200:
                result_holder["error"] = f"Failed to start: {resp.status_code}"
                return
            state = resp.json()
            with latest_state["lock"]:
                latest_state["state"] = state

        step = 0
        was_alive = True
        death_tick = None
        fut = request_llm(state)

        while step < MAX_STEPS and not state.get("done", False):
            action = fut.result()
            send_action(action)
            state = get_latest()
            step += 1

            # Track the tick at which this player died
            if was_alive and not _is_player_alive(state, player_id):
                was_alive = False
                death_tick = state.get("tick", step)

            if verbose:
                print(f"  [{player_label}] step {step}: {action}")
            if not state.get("done", False):
                fut = request_llm(state)

        # Record result
        result_holder["final_state"] = state
        result_holder["steps"] = step
        result_holder["death_tick"] = death_tick  # None means survived to end
        result_holder["final_tick"] = state.get("tick", step)

    except Exception as e:
        result_holder["error"] = str(e)
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        executor.shutdown(wait=False)


# ── Run one match ────────────────────────────────────────────────────────────

def run_match(base, token, room_seed, tick_ms, win_bonus,
              endpoint_a, prompt_a, endpoint_b, prompt_b, verbose=False):
    """
    Run a single match between player A and player B.
    Returns a dict with per-player rewards and match metadata.
    """
    headers = {"Authorization": f"Bearer {token}"}

    # Create session with default grid/map size (matches a real game)
    session, _ = find_or_create_session(
        base, headers, room_seed, 88, auto_advance=True,
    )
    session_id = session["session_id"]

    try:
        # Add both bots
        actor_a = add_bot(base, headers, session_id, "Player-A", "#ff4444")
        actor_b = add_bot(base, headers, session_id, "Player-B", "#4444ff")

        if actor_a is None or actor_b is None:
            return {"error": "Failed to add bots"}

        # Wait for both players
        wait_for_players(base, headers, session_id)

        result_a = {}
        result_b = {}

        thread_a = threading.Thread(target=run_player, kwargs=dict(
            base=base, headers=headers, session_id=session_id,
            actor_id=actor_a["id"], player_id=actor_a["player_id"],
            endpoint=endpoint_a, system_prompt=prompt_a,
            is_creator=True, result_holder=result_a, player_label="A",
            token=token, verbose=verbose,
        ))
        thread_b = threading.Thread(target=run_player, kwargs=dict(
            base=base, headers=headers, session_id=session_id,
            actor_id=actor_b["id"], player_id=actor_b["player_id"],
            endpoint=endpoint_b, system_prompt=prompt_b,
            is_creator=False, result_holder=result_b, player_label="B",
            token=token, verbose=verbose,
        ))

        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=300)
        thread_b.join(timeout=300)

        if "error" in result_a:
            return {"error": result_a["error"]}
        if "error" in result_b:
            return {"error": result_b["error"]}

        # Determine winner from final state
        final = result_a.get("final_state") or result_b.get("final_state")
        if final is None:
            return {"error": "No final state"}

        final_tick = max(
            result_a.get("final_tick", 0),
            result_b.get("final_tick", 0),
        )
        steps = max(result_a.get("steps", 0), result_b.get("steps", 0))

        winner_pid = final.get("round_winner_player_id")

        # Determine winner label
        a_won = winner_pid == actor_a["player_id"] if winner_pid else False
        b_won = winner_pid == actor_b["player_id"] if winner_pid else False

        if a_won:
            winner = "A"
        elif b_won:
            winner = "B"
        else:
            winner = "draw"

        # Compute survival ticks per player.
        # In a 2-player game the round ends the instant one dies, so both
        # threads see the same final_tick. Use death_tick if the thread
        # captured it (explicit None check — 0 is a valid tick).
        death_tick_a = result_a.get("death_tick")
        death_tick_b = result_b.get("death_tick")
        survival_tick_a = death_tick_a if death_tick_a is not None else final_tick
        survival_tick_b = death_tick_b if death_tick_b is not None else final_tick

        survival_sec_a = survival_tick_a * tick_ms / 1000.0
        survival_sec_b = survival_tick_b * tick_ms / 1000.0

        reward_a = survival_sec_a + (win_bonus if a_won else 0)
        reward_b = survival_sec_b + (win_bonus if b_won else 0)

        return {
            "winner": winner,
            "steps": steps,
            "final_tick": final_tick,
            "survival_sec_a": survival_sec_a,
            "survival_sec_b": survival_sec_b,
            "reward_a": reward_a,
            "reward_b": reward_b,
            "a_won": a_won,
            "b_won": b_won,
            "error": None,
        }

    finally:
        delete_session(base, headers, session_id)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two LLM configs head-to-head across multiple maps",
    )
    parser.add_argument("--maps", type=int, default=10, help="Number of maps (seeds) to play")
    parser.add_argument("--url", default=DEFAULT_SERVER, help="Game server URL")
    parser.add_argument("--token", default=os.environ.get("CURVYTRON_RL_API_TOKEN", ""),
                        help="API token")
    parser.add_argument("--tick-ms", type=float, default=16.0,
                        help="Milliseconds per game tick (default: 16, i.e. ~62.5 ticks/sec)")
    parser.add_argument("--win-bonus", type=float, default=WIN_BONUS,
                        help=f"Bonus seconds added to reward for winning (default: {WIN_BONUS})")
    parser.add_argument("--endpoint-a", default=DEFAULT_ENDPOINT_A, help="LLM endpoint for player A")
    parser.add_argument("--endpoint-b", default=DEFAULT_ENDPOINT_B, help="LLM endpoint for player B")
    parser.add_argument("--prompt-a", default=None,
                        help="Path to a text file with system prompt for A (default: built-in defensive)")
    parser.add_argument("--prompt-b", default=None,
                        help="Path to a text file with system prompt for B (default: built-in aggressive)")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of matches to run in parallel (default: 1)")
    parser.add_argument("--seed-prefix", default="eval", help="Prefix for room seeds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-step actions")
    parser.add_argument("--output", "-o", default=None, help="Write JSON results to file")
    args = parser.parse_args()

    win_bonus = args.win_bonus
    base = args.url.rstrip("/")
    wait_for_server(base)

    prompt_a = SYSTEM_PROMPT_A
    prompt_b = SYSTEM_PROMPT_B
    if args.prompt_a:
        with open(args.prompt_a) as f:
            prompt_a = f.read()
    if args.prompt_b:
        with open(args.prompt_b) as f:
            prompt_b = f.read()

    print("=" * 70)
    print("CURVYTRON LLM EVAL")
    print("=" * 70)
    print(f"  Server:     {base}")
    print(f"  Maps:       {args.maps}")
    print(f"  Tick:       {args.tick_ms}ms   Win bonus: {win_bonus}s")
    print(f"  Endpoint A: {args.endpoint_a}")
    print(f"  Endpoint B: {args.endpoint_b}")
    print(f"  Reward:     survival_seconds + {win_bonus} if won")
    print("=" * 70)
    print()

    results = [None] * args.maps  # pre-allocate to preserve map order
    wins = {"A": 0, "B": 0, "draw": 0, "error": 0}
    total_reward_a = 0.0
    total_reward_b = 0.0
    print_lock = threading.Lock()
    completed = [0]  # mutable counter

    def run_one(i):
        seed = f"{args.seed_prefix}-map-{i}"
        m = run_match(
            base=base, token=args.token, room_seed=seed,
            tick_ms=args.tick_ms, win_bonus=win_bonus,
            endpoint_a=args.endpoint_a, prompt_a=prompt_a,
            endpoint_b=args.endpoint_b, prompt_b=prompt_b,
            verbose=args.verbose,
        )
        return i, seed, m

    parallel = max(1, args.parallel)
    print(f"  Running {args.maps} matches with parallelism={parallel}\n")

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run_one, i): i for i in range(args.maps)}

        for fut in as_completed(futures):
            i, seed, m = fut.result()
            completed[0] += 1
            n = completed[0]

            if "error" in m and m["error"] and "winner" not in m:
                wins["error"] += 1
                results[i] = {
                    "map": i, "seed": seed, "winner": "error",
                    "steps": 0, "reward_a": 0, "reward_b": 0,
                    "survival_sec_a": 0, "survival_sec_b": 0,
                    "error": m["error"],
                }
                with print_lock:
                    print(f"[{n}/{args.maps}] seed={seed}  ERROR ({m['error']})")
                continue

            winner = m["winner"]
            wins[winner] += 1
            total_reward_a += m["reward_a"]
            total_reward_b += m["reward_b"]

            results[i] = {
                "map": i, "seed": seed, "winner": winner,
                "steps": m["steps"], "final_tick": m["final_tick"],
                "survival_sec_a": m["survival_sec_a"],
                "survival_sec_b": m["survival_sec_b"],
                "reward_a": m["reward_a"],
                "reward_b": m["reward_b"],
                "error": None,
            }

            with print_lock:
                print(
                    f"[{n}/{args.maps}] seed={seed}  "
                    f"{winner:>5}  "
                    f"A: {m['survival_sec_a']:6.1f}s (r={m['reward_a']:6.1f})  "
                    f"B: {m['survival_sec_b']:6.1f}s (r={m['reward_b']:6.1f})  "
                    f"[{m['steps']} steps]"
                )

    # ── Summary ──────────────────────────────────────────────────────────
    played = args.maps - wins["error"]
    avg_a = total_reward_a / played if played else 0
    avg_b = total_reward_b / played if played else 0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Player A wins: {wins['A']:>3} / {played}")
    print(f"  Player B wins: {wins['B']:>3} / {played}")
    print(f"  Draws:         {wins['draw']:>3} / {played}")
    if wins["error"]:
        print(f"  Errors:        {wins['error']:>3} / {args.maps}")
    print()
    print(f"  Total reward A: {total_reward_a:8.1f}   avg: {avg_a:6.1f}")
    print(f"  Total reward B: {total_reward_b:8.1f}   avg: {avg_b:6.1f}")
    print()
    delta = avg_a - avg_b
    if abs(delta) < 0.5:
        print(f"  Verdict: TIE (delta = {delta:+.1f}s)")
    elif delta > 0:
        print(f"  Verdict: Player A is better by {delta:+.1f}s avg reward")
    else:
        print(f"  Verdict: Player B is better by {-delta:+.1f}s avg reward")
    print("=" * 70)

    # Per-map breakdown
    print()
    hdr = f"  {'Map':<4} {'Seed':<20} {'Win':<5} {'SurvA':>7} {'RwdA':>7} {'SurvB':>7} {'RwdB':>7} {'Steps':>6}"
    print(hdr)
    print(f"  {'─'*4} {'─'*20} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")
    for r in results:
        if r.get("error"):
            print(f"  {r['map']:<4} {r['seed']:<20} ERROR  {r['error']}")
        else:
            print(
                f"  {r['map']:<4} {r['seed']:<20} {r['winner']:<5} "
                f"{r['survival_sec_a']:>6.1f}s {r['reward_a']:>6.1f} "
                f"{r['survival_sec_b']:>6.1f}s {r['reward_b']:>6.1f} "
                f"{r['steps']:>6}"
            )

    if args.output:
        out = {
            "config": {
                "server": base,
                "maps": args.maps,
                "tick_ms": args.tick_ms,
                "win_bonus": win_bonus,
                "endpoint_a": args.endpoint_a,
                "endpoint_b": args.endpoint_b,
                "prompt_a_label": args.prompt_a or "defensive",
                "prompt_b_label": args.prompt_b or "aggressive",
            },
            "summary": {
                **wins,
                "total_reward_a": total_reward_a,
                "total_reward_b": total_reward_b,
                "avg_reward_a": avg_a,
                "avg_reward_b": avg_b,
            },
            "matches": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
