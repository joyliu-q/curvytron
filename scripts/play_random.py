#!/usr/bin/env python3
"""
Play a curvytron game with random moves via the RL API.
Each instance controls exactly 1 player. Run two instances with the same room
to get a 2-player game.

Usage:
    # Terminal 1:
    python scripts/play_random.py --room myroom

    # Terminal 2 (auto-joins the existing session for "myroom"):
    python scripts/play_random.py --room myroom
"""

import argparse
import random
import shutil
import sys

import requests

from common import (
    ACTIONS, MAX_STEPS, ESC, CLEAR_SCREEN, HIDE_CURSOR, SHOW_CURSOR,
    render_frame, get_state, setup_session, add_common_args,
)


def main():
    parser = argparse.ArgumentParser(description="Play curvytron with random moves (1 player per script)")
    add_common_args(parser)
    args = parser.parse_args()

    base, headers, session_id, is_creator, my_actor_id, my_player_id, bot_name, spectate_url = setup_session(args)

    # --- hide cursor, clear screen for ANSI rendering ---
    sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
    sys.stdout.flush()

    auto_advance = not args.no_auto_advance

    try:
        if is_creator:
            # --- creator: start episode ---
            resp = requests.post(f"{base}/api/rl/sessions/{session_id}/start", headers=headers)
            if resp.status_code != 200:
                print(f"Failed to start episode: {resp.status_code} {resp.text}")
                sys.exit(1)
            state = resp.json()
            render_frame(state, my_player_id, 0, "creator", spectate_url=spectate_url)

            step = 0
            if auto_advance:
                # Auto-advance mode: server ticks the game; we just set actions and poll
                while step < MAX_STEPS and not state.get("done", False):
                    my_action = random.choice(ACTIONS)
                    requests.post(
                        f"{base}/api/rl/sessions/{session_id}/actors/{my_actor_id}/action",
                        json={"action": my_action},
                        headers=headers,
                    )
                    time.sleep(0.05)
                    state = get_state(base, headers, session_id)
                    step += 1
                    render_frame(state, my_player_id, step, "creator", spectate_url=spectate_url)
            else:
                # Manual step mode: we drive the game tick by tick
                while step < MAX_STEPS and not state.get("done", False):
                    my_action = random.choice(ACTIONS)
                    resp = requests.post(
                        f"{base}/api/rl/sessions/{session_id}/step",
                        json={"actions": {str(my_actor_id): my_action}},
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        break
                    state = resp.json()
                    step += 1
                    render_frame(state, my_player_id, step, "creator", spectate_url=spectate_url)

            render_frame(state, my_player_id, step, "creator", spectate_url=spectate_url)

            # --- cleanup ---
            requests.delete(f"{base}/api/rl/sessions/{session_id}", headers=headers)

        else:
            # --- joiner: set our action each tick, poll state until done ---
            import time
            state = get_state(base, headers, session_id)
            step = 0
            while step < MAX_STEPS and not state.get("done", False):
                my_action = random.choice(ACTIONS)
                requests.post(
                    f"{base}/api/rl/sessions/{session_id}/actors/{my_actor_id}/action",
                    json={"action": my_action},
                    headers=headers,
                )
                time.sleep(0.01)
                state = get_state(base, headers, session_id)
                step += 1
                render_frame(state, my_player_id, step, "joiner", spectate_url=spectate_url)

            render_frame(state, my_player_id, step, "joiner", spectate_url=spectate_url)

    finally:
        sys.stdout.write(SHOW_CURSOR + f"\n{ESC}[{shutil.get_terminal_size((100, 40)).lines}H\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
