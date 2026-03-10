"""
Shared helpers for curvytron player scripts.
"""

import hashlib
import math
import os
import random
import shutil
import sys
import time

import requests

ACTIONS = ["left", "straight", "right"]
MAX_STEPS = 2000
POLL_INTERVAL = 0.5
NUM_PLAYERS = 2

DEFAULT_URL = "https://joyliu-q--curvytron-curvytron.us-east.modal.direct"

# ── ANSI helpers ──────────────────────────────────────────────────────────────

ESC = "\033"
CLEAR_SCREEN = f"{ESC}[2J"
HOME = f"{ESC}[H"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
HIDE_CURSOR = f"{ESC}[?25l"
SHOW_CURSOR = f"{ESC}[?25h"

PLAYER_COLORS = [
    f"{ESC}[38;5;196m",  # red
    f"{ESC}[38;5;51m",   # cyan
    f"{ESC}[38;5;226m",  # yellow
    f"{ESC}[38;5;46m",   # green
]
WALL_COLOR = f"{ESC}[38;5;240m"
BONUS_COLOR = f"{ESC}[38;5;214m"
EMPTY_COLOR = f"{ESC}[38;5;236m"
HEAD_STYLE = f"{ESC}[1m"


# ── Rendering ─────────────────────────────────────────────────────────────────

def colorize_board(ascii_board, players):
    """Convert plain ASCII board to ANSI-colored output."""
    marker_color = {}
    for i, p in enumerate(players):
        marker = p.get("marker", chr(65 + i))
        marker_color[marker] = PLAYER_COLORS[i % len(PLAYER_COLORS)]

    lines = ascii_board.split("\n")
    colored = []
    for line in lines:
        parts = []
        for ch in line:
            if ch == "#":
                parts.append(f"{WALL_COLOR}#{RESET}")
            elif ch == ".":
                parts.append(f"{EMPTY_COLOR}.{RESET}")
            elif ch in marker_color:
                parts.append(f"{marker_color[ch]}{HEAD_STYLE}{ch}{RESET}")
            elif ch.isalpha():
                parts.append(f"{BONUS_COLOR}{ch}{RESET}")
            else:
                parts.append(ch)
        colored.append("".join(parts))
    return "\n".join(colored)


def build_status_bar(state, my_player_id, step, role):
    """Build a status bar showing game observations."""
    tick = state.get("tick", 0)
    done = state.get("done", False)
    players = state.get("players", [])
    bonuses = state.get("bonuses", [])
    board = state.get("board", {})
    board_size = board.get("size")
    borderless = board.get("borderless", False)

    lines = []

    term_w = shutil.get_terminal_size((100, 40)).columns
    lines.append(f"{DIM}{'─' * term_w}{RESET}")

    status = f"{BOLD}{'GAME OVER' if done else 'PLAYING'}{RESET}"
    info = (
        f" {status}  "
        f"step {BOLD}{step}{RESET}  "
        f"tick {BOLD}{tick}{RESET}  "
        f"board {board_size or '?'}  "
        f"{'borderless' if borderless else 'walled'}  "
        f"bonuses {len(bonuses)}  "
        f"role: {role}"
    )
    lines.append(info)

    for i, p in enumerate(players):
        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        marker = p.get("marker", "?")
        name = p.get("name", "?")
        alive = p.get("alive", False)
        alive_str = f"{ESC}[32m ALIVE{RESET}" if alive else f"{ESC}[31m DEAD{RESET}"
        me_tag = f" {BOLD}(you){RESET}" if p.get("player_id") == my_player_id else ""

        x = p.get("x")
        y = p.get("y")
        pos = f"({x:.1f},{y:.1f})" if x is not None else "(-,-)"

        angle = p.get("angle")
        angle_deg = f"{math.degrees(angle):.0f}deg" if angle is not None else "-"

        vel = p.get("velocity")
        vel_str = f"{vel:.2f}" if vel is not None else "-"

        radius = p.get("radius")
        rad_str = f"{radius:.1f}" if radius is not None else "-"

        printing = p.get("printing")
        print_str = "ink" if printing else "gap" if printing is not None else "-"

        score = p.get("score", 0)
        rscore = p.get("round_score", 0)

        active = p.get("active_bonuses", [])
        buff_parts = []
        for b in active:
            btype = b.get("type", "?")
            label = btype.replace("Bonus", "")
            remaining = b.get("remaining_ms")
            if remaining is not None:
                buff_parts.append(f"{label}({remaining / 1000:.1f}s)")
            else:
                buff_parts.append(label)
        buffs_str = ", ".join(buff_parts) if buff_parts else "none"

        lines.append(
            f"  {color}{marker}{RESET} {name}{me_tag}{alive_str}  "
            f"pos={pos}  ang={angle_deg}  vel={vel_str}  "
            f"r={rad_str}  {print_str}  "
            f"score={score} round={rscore}"
        )
        lines.append(
            f"      buffs: {buffs_str}"
        )

    winner = state.get("round_winner_player_id")
    if done and winner is not None:
        winner_name = "?"
        for p in players:
            if p.get("player_id") == winner:
                winner_name = p.get("name", "?")
        me_tag = " (you!)" if winner == my_player_id else ""
        lines.append(f"  {BOLD}Winner: {winner_name}{me_tag}{RESET}")

    return "\n".join(lines)


def render_frame(state, my_player_id, step, role, extra_lines=None):
    """Clear screen and draw the board + status bar."""
    buf = HOME

    ascii_board = ""
    if "occupancy" in state and "ascii" in state["occupancy"]:
        ascii_board = state["occupancy"]["ascii"]

    players = state.get("players", [])
    board_lines = colorize_board(ascii_board, players) if ascii_board else ""
    status = build_status_bar(state, my_player_id, step, role)

    buf += board_lines + "\n" + status

    if extra_lines:
        buf += "\n" + "\n".join(extra_lines)

    term_h = shutil.get_terminal_size((100, 40)).lines
    extra_count = len(extra_lines) if extra_lines else 0
    used = ascii_board.count("\n") + 1 + status.count("\n") + 1 + extra_count
    if used < term_h:
        blank_line = " " * shutil.get_terminal_size((100, 40)).columns
        buf += "\n" + "\n".join(blank_line for _ in range(term_h - used - 1))

    sys.stdout.write(buf)
    sys.stdout.flush()


# ── Network helpers ───────────────────────────────────────────────────────────

def wait_for_server(base):
    print("Waiting for server...", end="", flush=True)
    for _ in range(30):
        try:
            requests.get(base, timeout=2)
            break
        except requests.ConnectionError:
            print(".", end="", flush=True)
            time.sleep(1)
    else:
        print("\nServer not reachable. Is it running?")
        sys.exit(1)
    print(" connected!")


def find_or_create_session(base, headers, room, grid):
    """Create a session, or return the existing one if one already exists for this room."""
    resp = requests.post(f"{base}/api/rl/sessions", json={
        "seed": room,
        "grid_width": grid,
        "grid_height": grid,
        "max_score": 1,
        "warmup_ms": 0,
        "warmdown_ms": 0,
        "print_delay_ms": 0,
    }, headers=headers)
    if resp.status_code not in (200, 201):
        print(f"Failed to create/find session: {resp.status_code} {resp.text}")
        sys.exit(1)
    created = resp.status_code == 201
    return resp.json(), created


def add_bot(base, headers, session_id, name, color):
    resp = requests.post(f"{base}/api/rl/sessions/{session_id}/bots", json={
        "name": name,
        "color": color,
    }, headers=headers)
    if resp.status_code == 201:
        return resp.json()
    return None


def delete_session(base, headers, session_id):
    requests.delete(f"{base}/api/rl/sessions/{session_id}", headers=headers)


def get_state(base, headers, session_id):
    resp = requests.get(f"{base}/api/rl/sessions/{session_id}/state", headers=headers)
    if resp.status_code != 200:
        print(f"Failed to get state: {resp.status_code} {resp.text}")
        sys.exit(1)
    return resp.json()


def wait_for_players(base, headers, session_id):
    """Poll state until NUM_PLAYERS players are present."""
    print(f"Waiting for {NUM_PLAYERS} players to join...", end="", flush=True)
    while True:
        state = get_state(base, headers, session_id)
        players = state.get("players", [])
        if len(players) >= NUM_PLAYERS:
            print(f" {len(players)} players ready!")
            return state
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


# ── Session setup ─────────────────────────────────────────────────────────────

def setup_session(args):
    """
    Common session setup: connect, create/join session, add bot, wait for players.
    Returns (base, headers, session_id, is_creator, my_actor_id, my_player_id, bot_name).
    """
    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"}

    wait_for_server(base)

    session, is_creator = find_or_create_session(base, headers, args.room, args.grid)
    session_id = session["session_id"]
    if is_creator:
        print(f"Session created: {session_id}")
    else:
        print(f"Joined existing session: {session_id}")

    room_name = session.get("room_name", "")
    if not room_name and session_id.startswith("rl:"):
        room_name = f"rl-session-{session_id.split(':')[1]}"
    if room_name:
        print(f"Spectate live: {base}/#/room/{room_name}")

    bot_name = args.name
    if not bot_name:
        tag = hashlib.md5(f"{session_id}-{os.getpid()}-{time.time()}".encode()).hexdigest()[:4]
        bot_name = f"bot-{tag}"

    color = f"#{random.randint(0, 0xFFFFFF):06x}"
    actor = add_bot(base, headers, session_id, bot_name, color)

    # If adding a bot failed (stale session from a previous run), clean up and retry
    if actor is None and not is_creator:
        print(f"Stale session detected — deleting {session_id} and creating a fresh one...")
        delete_session(base, headers, session_id)
        session, is_creator = find_or_create_session(base, headers, args.room, args.grid)
        session_id = session["session_id"]
        print(f"Session created: {session_id}")
        actor = add_bot(base, headers, session_id, bot_name, color)

    if actor is None:
        print(f"Failed to add bot to session {session_id}")
        sys.exit(1)

    my_actor_id = actor["id"]
    my_player_id = actor["player_id"]
    print(f"Joined as '{bot_name}' (actor {my_actor_id})")

    wait_for_players(base, headers, session_id)

    return base, headers, session_id, is_creator, my_actor_id, my_player_id, bot_name


def add_common_args(parser):
    """Add CLI arguments shared by all player scripts."""
    parser.add_argument("--room", required=True, help="Room name (also used as RNG seed)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument("--token", default=os.environ.get("CURVYTRON_RL_API_TOKEN", ""), help="API token")
    parser.add_argument("--name", default=None, help="Bot name (auto-generated if omitted)")
    parser.add_argument("--grid", type=int, default=88, help="Grid width and height")
