"""Evaluate a curvytron checkpoint by playing it against a baseline.

Spins up two SGLang engines (current vs baseline), plays N games via the
curvytron game server, and reports win rates.

Usage:
    # Eval latest checkpoint vs base model
    modal run slime/eval_selfplay.py::eval_vs_baseline \
        --current-path curvytron-selfplay-Qwen3-4B-20260329-040710/iter_0000099_hf \
        --baseline-path Qwen/Qwen3-4B \
        --num-games 20

    # Eval two checkpoints against each other
    modal run slime/eval_selfplay.py::eval_vs_baseline \
        --current-path curvytron-selfplay-Qwen3-4B-20260402-074456/iter_0000799 \
        --baseline-path curvytron-selfplay-Qwen3-4B-20260416-162623/iter_0000009 \
        --num-games 1
"""

import os
import re
import subprocess
import time
from pathlib import Path

import modal

MINUTES = 60

# Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("curvytron-checkpoints", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
CHECKPOINTS_PATH = Path("/checkpoints")

CURVYTRON_URL = os.environ.get(
    "CURVYTRON_URL",
    "https://modal-labs-joy-dev--curvytron-curvytron.us-east.modal.direct",
)

ACTION_REGEX = r"(left|straight|right)"
ACTIONS = ["left", "straight", "right"]
MAX_GAME_STEPS = 500

SYSTEM_PROMPT = """\
You are playing a multiplayer Snake/Tron-like game on a 2D grid.

## Rules
- You control a continuously moving avatar that leaves a trail behind it.
- Each tick you choose one of three actions: "left", "straight", or "right".
  - "left" turns your avatar left relative to its current heading.
  - "straight" keeps your current heading.
  - "right" turns your avatar right relative to its current heading.
- If your avatar collides with any trail (yours or an opponent's), a wall, or the border, you die.
- The last player alive wins the round.

## Board notation
The board is an ASCII grid where:
- "." = empty space (safe to move into)
- "#" = wall / border / trail segment (deadly on contact)
- Uppercase letters (A, B, …) = player head positions

## Response format
Respond with ONLY one word: left, straight, or right
"""

eval_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime")
    .entrypoint([])
    .pip_install("huggingface-hub==0.36.0", "requests")
)

app = modal.App(name="curvytron-eval")


# =============================================================================
# Prompt building (same as multi_agent_system.py)
# =============================================================================


def build_turn_prompt(state: dict, my_player_id: str, my_marker: str) -> str:
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

    lines = [f"## Tick {tick}  |  Board {board_w}x{board_h}", ""]
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


def build_prompt_text(state, player_id, marker):
    turn = build_turn_prompt(state, player_id, marker)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{turn}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# =============================================================================
# SGLang engine helpers
# =============================================================================


def start_sglang(model_path: str, port: int, gpu_id: int = 0) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp", "1",
        "--mem-fraction-static", "0.80",
        "--context-length", "8192",
    ]
    print(f"Starting SGLang on port {port} (GPU {gpu_id}): {model_path}")
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def wait_sglang(port: int, timeout: int = 10 * MINUTES):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{port}/health").raise_for_status()
            print(f"SGLang on port {port} ready")
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang on port {port} not ready")


def query_action(port: int, prompt: str) -> str:
    """Query SGLang /generate with regex constraint. Returns action string."""
    import requests
    import random

    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0.1, "max_new_tokens": 5},
        "regex": ACTION_REGEX,
    }
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json().get("text", "").strip().lower()
        # Strip any thinking tokens
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().lower()
        if text in ACTIONS:
            return text
    except Exception as e:
        print(f"[eval] query error on port {port}: {e}")
    return random.choice(ACTIONS)


# =============================================================================
# Game client (sync, minimal)
# =============================================================================


def game_request(method, path, **kwargs):
    import requests
    url = f"{CURVYTRON_URL}{path}"
    resp = getattr(requests, method)(url, timeout=30, **kwargs)
    return resp


def play_one_game(seed: str, port_current: int, port_baseline: int) -> dict:
    """Play one game. Returns {"winner": "current"|"baseline"|"draw", "steps": N}."""
    import random

    # Create session
    body = {
        "seed": seed, "grid_width": 88, "grid_height": 88,
        "max_score": 1, "warmup_ms": 0, "warmdown_ms": 0,
        "print_delay_ms": 0, "auto_advance": False,
    }
    resp = game_request("post", "/api/rl/sessions", json=body)
    session = resp.json()
    session_id = session["session_id"]

    try:
        # Add bots
        resp_a = game_request("post", f"/api/rl/sessions/{session_id}/bots",
                              json={"name": "Current", "color": "#ff4444"})
        resp_b = game_request("post", f"/api/rl/sessions/{session_id}/bots",
                              json={"name": "Baseline", "color": "#4444ff"})
        actor_a = resp_a.json()
        actor_b = resp_b.json()

        # Randomly assign which model plays which side
        if random.random() < 0.5:
            port_a, port_b = port_current, port_baseline
            label_a, label_b = "current", "baseline"
        else:
            port_a, port_b = port_baseline, port_current
            label_a, label_b = "baseline", "current"

        # Start
        resp = game_request("post", f"/api/rl/sessions/{session_id}/start")
        state = resp.json()

        marker_a = marker_b = "?"
        for p in state.get("players", []):
            if p.get("player_id") == actor_a["player_id"]:
                marker_a = p.get("marker", "?")
            elif p.get("player_id") == actor_b["player_id"]:
                marker_b = p.get("marker", "?")

        # Game loop
        step = 0
        a_alive = b_alive = True

        while step < MAX_GAME_STEPS and not state.get("done", False):
            # Query both models
            action_a = "straight"
            action_b = "straight"
            if a_alive:
                prompt_a = build_prompt_text(state, actor_a["player_id"], marker_a)
                action_a = query_action(port_a, prompt_a)
            if b_alive:
                prompt_b = build_prompt_text(state, actor_b["player_id"], marker_b)
                action_b = query_action(port_b, prompt_b)

            # Send actions
            game_request("post", f"/api/rl/sessions/{session_id}/actors/{actor_a['id']}/action",
                         json={"action": action_a})
            game_request("post", f"/api/rl/sessions/{session_id}/actors/{actor_b['id']}/action",
                         json={"action": action_b})

            # Step
            game_request("post", f"/api/rl/sessions/{session_id}/step")
            resp = game_request("get", f"/api/rl/sessions/{session_id}/state")
            state = resp.json()
            step += 1

            # Check alive
            for p in state.get("players", []):
                if p.get("player_id") == actor_a["player_id"]:
                    a_alive = p.get("alive", False)
                elif p.get("player_id") == actor_b["player_id"]:
                    b_alive = p.get("alive", False)

        # Determine winner
        if a_alive and not b_alive:
            winner = label_a
        elif b_alive and not a_alive:
            winner = label_b
        else:
            winner = "draw"

        print(f"  Game {seed}: {winner} wins in {step} steps "
              f"({label_a}={marker_a} vs {label_b}={marker_b})")
        return {"winner": winner, "steps": step, "seed": seed}

    finally:
        try:
            game_request("delete", f"/api/rl/sessions/{session_id}")
        except Exception:
            pass


# =============================================================================
# Modal eval function
# =============================================================================


@app.function(
    image=eval_image,
    gpu="H100:2",
    volumes={HF_CACHE_PATH: hf_cache_vol, CHECKPOINTS_PATH.as_posix(): checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 60 * MINUTES,
)
def eval_vs_baseline(
    current_path: str,
    baseline_path: str = "Qwen/Qwen3-4B",
    num_games: int = 20,
):
    """Evaluate current checkpoint vs baseline by playing curvytron games.

    Args:
        current_path: Path relative to /checkpoints (e.g. "experiment/iter_0000099_hf"),
                      or a HuggingFace model ID.
        baseline_path: Same format as current_path.
        num_games: Number of games to play.
    """
    convert_checkpoint = modal.Function.from_name("curvytron", "convert_checkpoint")

    # Resolve paths
    def resolve(p):
        full = CHECKPOINTS_PATH / p
        if full.exists():
            return str(full)
        # If it is not suffixed with _hf, convert it via the deployed curvytron app.
        if not p.endswith("_hf"):
            # p looks like "<model_path>/iter_XXXXXXX"; split into model_path and iter_dir.
            parts = p.split("/")
            if len(parts) >= 2:
                model_path = "/".join(parts[:-1])
                iter_dir = parts[-1]
                # origin_hf_dir is the base HF repo name embedded in model_path
                # (e.g. "curvytron-selfplay-Qwen3-4B-..." -> "Qwen3-4B").
                match = re.search(r"(Qwen[^-/]*-[^-/]+)", model_path)
                origin_hf_dir = match.group(1) if match else "Qwen3-4B"
                convert_checkpoint.remote(
                    model_path=model_path,
                    iter_dir=iter_dir,
                    origin_hf_dir=origin_hf_dir,
                )
                checkpoints_vol.reload()
                return str(CHECKPOINTS_PATH / model_path / f"{iter_dir}_hf")
        # Might be an HF model ID — SGLang will download it
        return p

    current_resolved = resolve(current_path)
    baseline_resolved = resolve(baseline_path)

    print(f"Eval: {current_path} vs {baseline_path}")
    print(f"  Current:  {current_resolved}")
    print(f"  Baseline: {baseline_resolved}")
    print(f"  Games:    {num_games}")

    # Start two SGLang engines on separate GPUs
    PORT_CURRENT = 8000
    PORT_BASELINE = 8001

    proc_current = start_sglang(current_resolved, PORT_CURRENT, gpu_id=0)
    proc_baseline = start_sglang(baseline_resolved, PORT_BASELINE, gpu_id=1)

    try:
        wait_sglang(PORT_CURRENT)
        wait_sglang(PORT_BASELINE)

        # Play games
        results = []
        for i in range(num_games):
            seed = f"eval-{i}-{int(time.time())}"
            result = play_one_game(seed, PORT_CURRENT, PORT_BASELINE)
            results.append(result)

        # Summarize
        current_wins = sum(1 for r in results if r["winner"] == "current")
        baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
        draws = sum(1 for r in results if r["winner"] == "draw")
        avg_steps = sum(r["steps"] for r in results) / len(results)

        print(f"\n{'='*60}")
        print(f"Results: {current_path} vs {baseline_path}")
        print(f"  Current wins:  {current_wins}/{num_games} ({100*current_wins/num_games:.0f}%)")
        print(f"  Baseline wins: {baseline_wins}/{num_games} ({100*baseline_wins/num_games:.0f}%)")
        print(f"  Draws:         {draws}/{num_games}")
        print(f"  Avg steps:     {avg_steps:.1f}")
        print(f"{'='*60}")

        return {
            "current_path": current_path,
            "baseline_path": baseline_path,
            "current_wins": current_wins,
            "baseline_wins": baseline_wins,
            "draws": draws,
            "num_games": num_games,
            "avg_steps": avg_steps,
            "win_rate": current_wins / num_games,
        }

    finally:
        proc_current.terminate()
        proc_baseline.terminate()
        proc_current.wait()
        proc_baseline.wait()
