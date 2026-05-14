"""Evaluate a curvytron checkpoint vs a baseline using training-gym.

Deploys both models via DeploymentConfig, then uses EvalConfig with a
custom eval_fn that plays games on the curvytron game server.

Usage:
    python slime/eval_gym.py --training-run-id "Qwen/Qwen3-0.6B.546ca7ad-..." --num-games 10

    # Or specify checkpoint index directly
    python slime/eval_gym.py --training-run-id "Qwen/Qwen3-0.6B.546ca7ad-..." --checkpoint-idx -1

    # Custom baseline
    python slime/eval_gym.py --training-run-id "..." --baseline Qwen/Qwen3-4B
"""

import argparse
import os
import random
import re
import uuid

import modal
import requests

from modal_training_gym import (
    DeploymentConfig,
    EvalConfig,
    EvalRowResult,
    ModelDeployment,
    Qwen3_0_6B,
    Qwen3_4B,
    Qwen3_8B,
    list_checkpoints,
)
from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType

from curvytron.dataset import CurvytronSeedDataset
from curvytron.prompts import SYSTEM_PROMPT

CURVYTRON_URL = os.environ.get(
    "CURVYTRON_URL",
    "https://modal-labs-joy-dev--curvytron-curvytron.us-east.modal.direct",
)

ACTIONS = ["left", "straight", "right"]
MAX_GAME_STEPS = 500
GRID_SIZE = 88


# ── Model lookup ────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "Qwen/Qwen3-0.6B": Qwen3_0_6B,
    "Qwen/Qwen3-4B": Qwen3_4B,
    "Qwen/Qwen3-8B": Qwen3_8B,
}


def _model_config_for(model_name: str):
    for prefix, cls in MODEL_CONFIGS.items():
        if model_name.startswith(prefix):
            return cls()
    return Qwen3_4B()


# ── Game helpers ────────────────────────────────────────────────────────────

def _game_headers(session_id: str | None = None) -> dict:
    h: dict = {}
    if session_id:
        h["Modal-Session-Id"] = session_id
    return h


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


def query_action(deployment: ModelDeployment, state: dict, player_id: str, marker: str) -> str:
    turn = build_turn_prompt(state, player_id, marker)
    prompt = f"{SYSTEM_PROMPT}\n\n{turn}"
    try:
        response = deployment.generate(
            prompt,
            ensure_ready=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip().lower()
        for action in ACTIONS:
            if clean.startswith(action):
                return action
    except Exception as e:
        print(f"[eval] generate error: {e}")
    return random.choice(ACTIONS)


def play_one_game(
    seed: str,
    current: ModelDeployment,
    baseline: ModelDeployment,
) -> dict:
    modal_sid = uuid.uuid4().hex
    headers = _game_headers(modal_sid)

    body = {
        "seed": seed, "grid_width": GRID_SIZE, "grid_height": GRID_SIZE,
        "max_score": 1, "warmup_ms": 0, "warmdown_ms": 0,
        "print_delay_ms": 0, "auto_advance": False,
    }
    resp = requests.post(f"{CURVYTRON_URL}/api/rl/sessions", json=body, headers=headers, timeout=30)
    session = resp.json()
    session_id = session["session_id"]

    try:
        resp_a = requests.post(
            f"{CURVYTRON_URL}/api/rl/sessions/{session_id}/bots",
            json={"name": "Current", "color": "#ff4444"}, headers=headers, timeout=30,
        )
        resp_b = requests.post(
            f"{CURVYTRON_URL}/api/rl/sessions/{session_id}/bots",
            json={"name": "Baseline", "color": "#4444ff"}, headers=headers, timeout=30,
        )
        actor_a = resp_a.json()
        actor_b = resp_b.json()

        if random.random() < 0.5:
            deploy_a, deploy_b = current, baseline
            label_a, label_b = "current", "baseline"
        else:
            deploy_a, deploy_b = baseline, current
            label_a, label_b = "baseline", "current"

        resp = requests.post(
            f"{CURVYTRON_URL}/api/rl/sessions/{session_id}/start",
            headers=headers, timeout=30,
        )
        state = resp.json()

        marker_a = marker_b = "?"
        for p in state.get("players", []):
            if p.get("player_id") == actor_a["player_id"]:
                marker_a = p.get("marker", "?")
            elif p.get("player_id") == actor_b["player_id"]:
                marker_b = p.get("marker", "?")

        step = 0
        a_alive = b_alive = True

        while step < MAX_GAME_STEPS and not state.get("done", False):
            action_a = query_action(deploy_a, state, actor_a["player_id"], marker_a) if a_alive else "straight"
            action_b = query_action(deploy_b, state, actor_b["player_id"], marker_b) if b_alive else "straight"

            resp = requests.post(
                f"{CURVYTRON_URL}/api/rl/sessions/{session_id}/step",
                json={"actions": {str(actor_a["id"]): action_a, str(actor_b["id"]): action_b}},
                headers=headers, timeout=30,
            )
            state = resp.json()
            step += 1

            for p in state.get("players", []):
                if p.get("player_id") == actor_a["player_id"]:
                    a_alive = p.get("alive", False)
                elif p.get("player_id") == actor_b["player_id"]:
                    b_alive = p.get("alive", False)

        if a_alive and not b_alive:
            winner = label_a
        elif b_alive and not a_alive:
            winner = label_b
        else:
            winner = "draw"

        print(f"  Game {seed}: {winner} wins in {step} steps")
        return {"winner": winner, "steps": step, "seed": seed}

    finally:
        try:
            requests.delete(f"{CURVYTRON_URL}/api/rl/sessions/{session_id}", headers=headers, timeout=10)
        except Exception:
            pass


# ── EvalConfig integration ──────────────────────────────────────────────────

class CurvytronEvalDataset(CurvytronSeedDataset):
    """Eval dataset — just game seeds, loaded for EvalConfig.evaluate()."""

    def load(self):
        return [{"prompt": f"eval-seed-{i}", "label": ""} for i in range(self._num_games)]

    def __init__(self, num_games: int = 10):
        self._num_games = num_games
        super().__init__(num_seeds=num_games)


def make_curvytron_eval_fn(baseline: ModelDeployment):
    """Return an eval_fn that plays a game: current (from EvalConfig) vs baseline."""

    def curvytron_eval_fn(
        deployment: ModelDeployment,
        example: dict,
    ) -> EvalRowResult:
        seed = example.get("prompt", f"eval-{uuid.uuid4().hex[:8]}")
        result = play_one_game(seed, current=deployment, baseline=baseline)
        score = 1.0 if result["winner"] == "current" else (0.5 if result["winner"] == "draw" else 0.0)
        return EvalRowResult(
            score=score,
            response=f"{result['winner']} in {result['steps']} steps",
            metadata=result,
        )

    return curvytron_eval_fn


def summarize_eval(eval_result):
    rows = eval_result.rows
    wins = sum(1 for r in rows if r.metadata.get("winner") == "current")
    losses = sum(1 for r in rows if r.metadata.get("winner") == "baseline")
    draws = sum(1 for r in rows if r.metadata.get("winner") == "draw")
    avg_steps = sum(r.metadata.get("steps", 0) for r in rows) / max(len(rows), 1)
    return {
        "win_rate": wins / max(len(rows), 1),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_steps": avg_steps,
        "total": len(rows),
    }


# ── Checkpoint fallback ──────────────────────────────────────────────────────

VOLUME_NAME = "slime-slimerecipe-checkpoints"
MOUNT_PATH = "/checkpoints"


def _scan_checkpoints_volume(training_run_id: str) -> list[Checkpoint]:
    """Scan the checkpoints volume directly when TrainResult metadata is missing."""
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    prefix = "iter_"
    checkpoints = []

    try:
        entries = list(vol.iterdir(training_run_id, recursive=False))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []

    for entry in sorted(entries, key=lambda e: getattr(e, "path", "")):
        name = getattr(entry, "path", "").rstrip("/").rsplit("/", 1)[-1]
        if not name.startswith(prefix):
            continue
        ckpt_type = CheckpointType.hf if name.endswith("_hf") else CheckpointType.megatron
        checkpoints.append(Checkpoint(
            checkpoint_type=ckpt_type,
            name=name,
            path=f"{MOUNT_PATH}/{training_run_id}/{name}",
            timestamp=float(getattr(entry, "mtime", 0.0)),
            training_run_id=training_run_id,
            app_name="slime-slimerecipe",
            checkpoints_volume_name=VOLUME_NAME,
            checkpoints_mount_path=MOUNT_PATH,
        ))

    return checkpoints


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Curvytron eval via training-gym")
    parser.add_argument("--training-run-id", required=True, help="Training run ID from train_gym.py")
    parser.add_argument("--checkpoint-idx", type=int, default=-1, help="Checkpoint index (-1 = latest)")
    parser.add_argument("--baseline", default=None, help="Baseline model name (default: same base model)")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    args = parser.parse_args()

    print(f"Loading checkpoints for: {args.training_run_id}")
    try:
        checkpoints = list_checkpoints(args.training_run_id)
    except KeyError:
        print("  TrainResult not found in metadata, scanning volume directly...")
        checkpoints = _scan_checkpoints_volume(args.training_run_id)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found for {args.training_run_id}")

    checkpoint = checkpoints[args.checkpoint_idx]
    print(f"Using checkpoint: {checkpoint.name} (type={checkpoint.checkpoint_type})")

    model_name = args.training_run_id.rsplit(".", 2)[0]
    model_config = _model_config_for(model_name)
    baseline_name = args.baseline or model_name
    baseline_config = _model_config_for(baseline_name)

    print(f"Deploying current: {model_name} @ {checkpoint.name}")
    current_deployment = DeploymentConfig(
        model=model_config,
        checkpoint=checkpoint,
        app_name=f"curvytron-eval-current",
        served_model_name="current",
    ).serve()
    print(f"  Current URL: {current_deployment.url}")

    print(f"Deploying baseline: {baseline_name}")
    baseline_deployment = DeploymentConfig(
        model=baseline_config,
        app_name=f"curvytron-eval-baseline",
        served_model_name="baseline",
    ).serve()
    print(f"  Baseline URL: {baseline_deployment.url}")

    eval_dataset = CurvytronEvalDataset(num_games=args.num_games)
    eval_config = EvalConfig(
        dataset=eval_dataset,
        eval_fn=make_curvytron_eval_fn(baseline_deployment),
    )

    print(f"\nPlaying {args.num_games} games...")
    eval_result = eval_config.evaluate(current_deployment, debug=True)

    summary = summarize_eval(eval_result)
    print(f"\n{'='*60}")
    print(f"Results: {checkpoint.name} vs {baseline_name}")
    print(f"  Win rate:  {summary['win_rate']:.0%}")
    print(f"  Wins:      {summary['wins']}/{summary['total']}")
    print(f"  Losses:    {summary['losses']}/{summary['total']}")
    print(f"  Draws:     {summary['draws']}/{summary['total']}")
    print(f"  Avg steps: {summary['avg_steps']:.1f}")
    print(f"  Mean score:{eval_result.mean:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
