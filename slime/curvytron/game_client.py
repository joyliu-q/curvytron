"""Async HTTP client for the curvytron game server.

Wraps synchronous requests in asyncio.to_thread so the self-play loop
can await game server calls alongside SGLang inference calls.
"""

import asyncio
import os
import time

import requests

DEFAULT_SERVER = os.environ.get(
    "CURVYTRON_URL",
    "https://joyliu-q--curvytron-curvytron.us-east.modal.direct",
)
DEFAULT_TOKEN = os.environ.get("CURVYTRON_RL_API_TOKEN", "")

GRID_SIZE = 88
MAX_STEPS = 2000


def _headers(token: str | None = None) -> dict:
    t = token or DEFAULT_TOKEN
    return {"Authorization": f"Bearer {t}"} if t else {}


# ── Synchronous helpers (called via asyncio.to_thread) ──────────────────────


def _create_session(base: str, headers: dict, seed: str) -> dict:
    """Create a game session with the given seed. Returns session dict."""
    body = {
        "seed": seed,
        "grid_width": GRID_SIZE,
        "grid_height": GRID_SIZE,
        "max_score": 1,
        "warmup_ms": 0,
        "warmdown_ms": 0,
        "print_delay_ms": 0,
        "auto_advance": True,
    }
    resp = requests.post(f"{base}/api/rl/sessions", json=body, headers=headers, timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create session: {resp.status_code} {resp.text}")
    return resp.json()


def _add_bot(base: str, headers: dict, session_id: str, name: str, color: str) -> dict:
    """Add a bot player. Returns {"id": actor_id, "player_id": player_id}."""
    resp = requests.post(
        f"{base}/api/rl/sessions/{session_id}/bots",
        json={"name": name, "color": color},
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"Failed to add bot {name}: {resp.status_code} {resp.text}")
    return resp.json()


def _wait_for_players(base: str, headers: dict, session_id: str, n: int = 2) -> dict:
    """Poll until n players are present. Returns the state."""
    for _ in range(60):
        state = _get_state(base, headers, session_id)
        if len(state.get("players", [])) >= n:
            return state
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {n} players")


def _start_game(base: str, headers: dict, session_id: str) -> dict:
    """Start the game episode. Returns initial state."""
    resp = requests.post(f"{base}/api/rl/sessions/{session_id}/start", headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to start: {resp.status_code} {resp.text}")
    return resp.json()


def _get_state(base: str, headers: dict, session_id: str) -> dict:
    """Get current game state."""
    resp = requests.get(
        f"{base}/api/rl/sessions/{session_id}/state",
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get state: {resp.status_code} {resp.text}")
    return resp.json()


def _send_action(base: str, headers: dict, session_id: str, actor_id: str, action: str):
    """Send an action for a specific actor."""
    requests.post(
        f"{base}/api/rl/sessions/{session_id}/actors/{actor_id}/action",
        json={"action": action},
        headers=headers,
        timeout=30,
    )


def _delete_session(base: str, headers: dict, session_id: str):
    """Delete (cleanup) a game session."""
    try:
        requests.delete(f"{base}/api/rl/sessions/{session_id}", headers=headers, timeout=10)
    except Exception:
        pass


# ── Async wrappers ──────────────────────────────────────────────────────────


class AsyncGameClient:
    """Async wrapper around the curvytron game server HTTP API."""

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base = (base_url or DEFAULT_SERVER).rstrip("/")
        self.headers = _headers(token)

    async def create_session(self, seed: str) -> dict:
        return await asyncio.to_thread(_create_session, self.base, self.headers, seed)

    async def add_bot(self, session_id: str, name: str, color: str) -> dict:
        return await asyncio.to_thread(_add_bot, self.base, self.headers, session_id, name, color)

    async def wait_for_players(self, session_id: str, n: int = 2) -> dict:
        return await asyncio.to_thread(_wait_for_players, self.base, self.headers, session_id, n)

    async def start_game(self, session_id: str) -> dict:
        return await asyncio.to_thread(_start_game, self.base, self.headers, session_id)

    async def get_state(self, session_id: str) -> dict:
        return await asyncio.to_thread(_get_state, self.base, self.headers, session_id)

    async def send_action(self, session_id: str, actor_id: str, action: str):
        await asyncio.to_thread(_send_action, self.base, self.headers, session_id, actor_id, action)

    async def delete_session(self, session_id: str):
        await asyncio.to_thread(_delete_session, self.base, self.headers, session_id)
